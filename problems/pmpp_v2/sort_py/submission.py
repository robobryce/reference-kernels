import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor segment_sort(torch::Tensor input, torch::Tensor output);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Phase 1: per-row BlockRadixSort in shared memory
// 256 threads x 40 items = 10240 capacity (covers 100M's ~10K-item rows)
// ---------------------------------------------------------------------------
constexpr int ROW_BLOCK_THREADS = 256;
constexpr int ROW_ITEMS_PER_THREAD = 40;

template <int BLOCK_DIM, int IPT>
__global__ void sort_rows_kernel(
    float*       data,
    const int*   row_offsets,
    int          rows,
    int          max_cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int start = row_offsets[row];
    int end   = row_offsets[row + 1];
    int count = end - start;

    using BlockRS = cub::BlockRadixSort<float, BLOCK_DIM, IPT>;
    __shared__ typename BlockRS::TempStorage temp_storage;

    // Per-thread register arrays — BlockRadixSort::Sort operates on these
    float thread_keys[IPT];

    float* row_data = data + start;

    // Load keys into per-thread arrays (pad tail with INFINITY)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        thread_keys[i] = (idx < count) ? row_data[idx] : INFINITY;
    }
    __syncthreads();

    // Radix sort the per-thread keys (in registers, using shared temp storage)
    BlockRS(temp_storage).Sort(thread_keys);
    __syncthreads();

    // Write back (excluding padding)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < count) {
            row_data[idx] = thread_keys[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: batched merge kernel — 2D grid (chunks_per_pair, num_pairs)
// Each block handles one chunk of one pair's output.
// ---------------------------------------------------------------------------
constexpr int MERGE_BLOCK_THREADS = 256;
constexpr int MERGE_ITEMS_PER_BLOCK = 1536;  // 6 items/thread

__global__ void merge_level_kernel(
    const float* src,
    const int*   pair_info,    // [num_pairs*4]: {a_start, lenA, b_start, lenB}
    float*       dst,
    const int*   dst_offsets,  // [num_pairs]
    int          num_pairs,
    int          max_chunks)
{
    int pair_idx = blockIdx.y;
    if (pair_idx >= num_pairs) return;

    int a_start   = pair_info[4 * pair_idx];
    int lenA      = pair_info[4 * pair_idx + 1];
    int b_start   = pair_info[4 * pair_idx + 2];
    int lenB      = pair_info[4 * pair_idx + 3];
    int total_len = lenA + lenB;
    int dst_start = dst_offsets[pair_idx];

    int chunk = blockIdx.x;
    int block_start = chunk * MERGE_ITEMS_PER_BLOCK;
    if (block_start >= total_len) return;
    int block_len = min(MERGE_ITEMS_PER_BLOCK, total_len - block_start);

    int tid = threadIdx.x;
    int items_per_thread = MERGE_ITEMS_PER_BLOCK / MERGE_BLOCK_THREADS;

    int t_start = block_start + tid * items_per_thread;
    int t_end   = min(t_start + items_per_thread, block_start + block_len);
    if (t_start >= t_end) return;
    int t_len = t_end - t_start;

    // binary search on merge path for this thread's start
    int diag = t_start;
    int low  = max(0, diag - lenB);
    int high = min(diag, lenA);
    while (low < high) {
        int mid = (low + high) / 2;
        int midB = diag - 1 - mid;
        if (midB < 0) {
            high = mid;
        } else if (midB >= lenB) {
            low = mid + 1;
        } else if (src[a_start + mid] <= src[b_start + midB]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int posA = a_start + low;
    int posB = b_start + t_start - low;
    int a_end_ptr = a_start + lenA;
    int b_end_ptr = b_start + lenB;

    float* out = dst + dst_start + t_start;
    #pragma unroll 1
    for (int i = 0; i < t_len; i++) {
        if (posA < a_end_ptr && (posB >= b_end_ptr || src[posA] <= src[posB])) {
            out[i] = src[posA++];
        } else {
            out[i] = src[posB++];
        }
    }
}

// ---------------------------------------------------------------------------
// Simple copy kernel (for odd leftover segments)
// ---------------------------------------------------------------------------
__global__ void copy_segment_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Host-side segment layout builder
// ---------------------------------------------------------------------------
struct SegLayout {
    std::vector<int> starts;
    std::vector<int> lengths;
};

static SegLayout build_initial_layout(const int* row_offsets, int rows) {
    SegLayout layout;
    layout.starts.reserve(rows + 1);
    for (int i = 0; i <= rows; i++) layout.starts.push_back(row_offsets[i]);
    layout.lengths.reserve(rows);
    for (int i = 0; i < rows; i++)
        layout.lengths.push_back(row_offsets[i + 1] - row_offsets[i]);
    return layout;
}

// Build the next level's layout from current level.
// Returns false when only one segment remains (fully sorted).
static bool next_merge_level(
    const SegLayout& cur,
    SegLayout&       next,
    std::vector<int>& pair_info,     // [num_pairs*4] for merge kernel
    std::vector<int>& dst_offsets,    // [num_pairs]
    int&              max_chunks)
{
    int n = (int)cur.lengths.size();
    if (n <= 1) return false;

    int pairs = n / 2;
    int pos = 0;

    next.starts.clear();
    next.lengths.clear();
    pair_info.clear();
    dst_offsets.clear();
    max_chunks = 0;

    for (int p = 0; p < pairs; p++) {
        int a_start = cur.starts[2 * p];
        int lenA    = cur.lengths[2 * p];
        int b_start = cur.starts[2 * p + 1];
        int lenB    = cur.lengths[2 * p + 1];

        pair_info.push_back(a_start);
        pair_info.push_back(lenA);
        pair_info.push_back(b_start);
        pair_info.push_back(lenB);
        dst_offsets.push_back(pos);

        int total = lenA + lenB;
        next.starts.push_back(pos);
        next.lengths.push_back(total);
        pos += total;

        int chunks = (total + MERGE_ITEMS_PER_BLOCK - 1) / MERGE_ITEMS_PER_BLOCK;
        if (chunks > max_chunks) max_chunks = chunks;
    }

    // Odd leftover
    if (n % 2 == 1) {
        int last_start = cur.starts[n - 1];
        int last_len   = cur.lengths[n - 1];
        next.starts.push_back(pos);
        next.lengths.push_back(last_len);
        pos += last_len;
    }

    next.starts.push_back(pos);  // sentinel
    return true;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
torch::Tensor segment_sort(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda()  && output.is_cuda(),  "must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 &&
                output.dtype() == torch::kFloat32,     "must be float32");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous());

    int N = (int)input.numel();
    if (N == 0) return output;
    if (N == 1) { output[0] = input[0]; return output; }

    // --- row dimensions ---
    int rows = (int)std::sqrt((double)N);
    int cols = (N + rows - 1) / rows;

    // Row offsets
    std::vector<int> row_offsets(rows + 1);
    for (int i = 0; i < rows; i++) row_offsets[i] = i * cols;
    row_offsets[rows] = N;

    // --- Phase 1: per-row BlockRadixSort ---
    {
        int* d_row_offsets;
        cudaMalloc(&d_row_offsets, (rows + 1) * sizeof(int));
        cudaMemcpy(d_row_offsets, row_offsets.data(), (rows + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);

        sort_rows_kernel<ROW_BLOCK_THREADS, ROW_ITEMS_PER_THREAD>
            <<<rows, ROW_BLOCK_THREADS>>>(
                input.data_ptr<float>(), d_row_offsets, rows, cols);

        cudaDeviceSynchronize();
        cudaFree(d_row_offsets);
    }

    // --- Phase 2: multi-level merge tree ---
    SegLayout cur = build_initial_layout(row_offsets.data(), rows);

    float* bufs[2] = { input.data_ptr<float>(), output.data_ptr<float>() };
    int which = 0;  // current source buffer

    int level = 0;
    while (true) {
        SegLayout next;
        std::vector<int> pair_info;
        std::vector<int> dst_offsets;
        int max_chunks = 0;

        bool more = next_merge_level(cur, next, pair_info, dst_offsets, max_chunks);
        if (!more) break;

        float* src_buf = bufs[which];
        float* dst_buf = bufs[1 - which];

        int num_pairs = (int)dst_offsets.size();
        if (num_pairs == 0) {
            // only one segment — just copy it
            int seg_len = cur.lengths[0];
            int seg_start = cur.starts[0];
            int blocks = (seg_len + 255) / 256;
            copy_segment_kernel<<<blocks, 256>>>(
                src_buf + seg_start, dst_buf, seg_len);
            cudaDeviceSynchronize();
            which = 1 - which;
            cur = next;
            continue;
        }

        // Upload pair info and dst offsets to GPU
        int* d_pair_info;
        int* d_dst_offsets;
        cudaMalloc(&d_pair_info, pair_info.size() * sizeof(int));
        cudaMalloc(&d_dst_offsets, dst_offsets.size() * sizeof(int));
        cudaMemcpy(d_pair_info, pair_info.data(),
                   pair_info.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dst_offsets, dst_offsets.data(),
                   dst_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

        dim3 grid(max_chunks, num_pairs);
        merge_level_kernel<<<grid, MERGE_BLOCK_THREADS>>>(
            src_buf, d_pair_info, dst_buf, d_dst_offsets, num_pairs, max_chunks);

        cudaDeviceSynchronize();
        cudaFree(d_pair_info);
        cudaFree(d_dst_offsets);

        // Copy odd leftover segment if present
        int n = (int)cur.lengths.size();
        if (n % 2 == 1 && n > 1) {
            int last_start = cur.starts[n - 1];
            int last_len   = cur.lengths[n - 1];
            int blocks = (last_len + 255) / 256;
            copy_segment_kernel<<<blocks, 256>>>(
                src_buf + last_start,
                dst_buf + next.starts[next.lengths.size() - 1],
                last_len);
            cudaDeviceSynchronize();
        }

        which = 1 - which;
        cur = next;
        level++;
    }

    cudaDeviceSynchronize();

    // If final result is in input buffer (odd number of flips), copy to output
    if (level % 2 == 1) {
        int blocks = (N + 255) / 256;
        copy_segment_kernel<<<blocks, 256>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), N);
        cudaDeviceSynchronize();
    }

    return output;
}
"""

segment_module = load_inline(
    name='segment_sort_module',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['segment_sort'],
    extra_cuda_cflags=['--expt-relaxed-constexpr', '-std=c++17'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Segmented sort: per-row BlockRadixSort (shared memory) then multi-level
    batched merge tree. Exploits the per-row-normal input: each row clusters
    around an increasing mean, so cross-row merges scan less overlap.
    """
    input_tensor, output_tensor = data
    segment_module.segment_sort(input_tensor, output_tensor)
    return output_tensor