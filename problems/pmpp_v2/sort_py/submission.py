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
// Phase 1: per-row segmented radix sort via cub::DeviceSegmentedRadixSort
// Each row is a segment; CUB sorts all rows independently in one batched call.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 2: batched merge kernel — 2D grid (chunks_per_pair, num_pairs)
// Each block handles one chunk (MERGE_ITEMS_PER_BLOCK items) of one pair.
// Threads binary-search independently on the merge path, then serially merge.
// ---------------------------------------------------------------------------
constexpr int MERGE_BLOCK_THREADS = 256;
constexpr int MERGE_ITEMS_PER_BLOCK = 2048;  // 8 items/thread

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
    int items_per_thread = MERGE_ITEMS_PER_BLOCK / MERGE_BLOCK_THREADS;  // 8

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
    for (int i = 0; i < t_len; i++) {
        if (posA < a_end_ptr && (posB >= b_end_ptr || src[posA] <= src[posB])) {
            out[i] = src[posA++];
        } else {
            out[i] = src[posB++];
        }
    }
}

// ---------------------------------------------------------------------------
// Copy kernel (for odd leftover segments)
// ---------------------------------------------------------------------------
__global__ void copy_segment_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Host-side segment layout
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

static bool next_merge_level(
    const SegLayout& cur,
    SegLayout&       next,
    std::vector<int>& pair_info,
    std::vector<int>& dst_offsets,
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

    // Odd leftover segment — pass through unsorted
    if (n % 2 == 1) {
        int last_start = cur.starts[n - 1];
        int last_len   = cur.lengths[n - 1];
        next.starts.push_back(pos);
        next.lengths.push_back(last_len);
        pos += last_len;
    }

    next.starts.push_back(pos);
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

    int rows = (int)std::sqrt((double)N);
    int cols = (N + rows - 1) / rows;

    // Row offsets
    std::vector<int> row_offsets(rows + 1);
    for (int i = 0; i < rows; i++) row_offsets[i] = i * cols;
    row_offsets[rows] = N;

    // --- Phase 1: per-row segmented radix sort ---
    // DeviceSegmentedRadixSort::SortKeys sorts each segment independently
    // d_begin_offsets[i] = start of segment i, d_end_offsets[i] = end of segment i
    {
        // Build begin/end offset arrays
        std::vector<int> h_begin_offsets(rows);
        std::vector<int> h_end_offsets(rows);
        for (int i = 0; i < rows; i++) {
            h_begin_offsets[i] = row_offsets[i];
            h_end_offsets[i]   = row_offsets[i + 1];
        }

        int* d_begin_offsets;
        int* d_end_offsets;
        cudaMalloc(&d_begin_offsets, rows * sizeof(int));
        cudaMalloc(&d_end_offsets,   rows * sizeof(int));
        cudaMemcpy(d_begin_offsets, h_begin_offsets.data(),
                   rows * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_end_offsets,   h_end_offsets.data(),
                   rows * sizeof(int), cudaMemcpyHostToDevice);

        // Query temp storage
        void*  d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, temp_bytes,
            input.data_ptr<float>(),  // keys in
            input.data_ptr<float>(),  // keys out (in-place sort)
            N, rows,
            d_begin_offsets, d_end_offsets,
            0, sizeof(float) * 8);

        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp, temp_bytes,
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N, rows,
            d_begin_offsets, d_end_offsets,
            0, sizeof(float) * 8);

        cudaDeviceSynchronize();
        cudaFree(d_temp);
        cudaFree(d_begin_offsets);
        cudaFree(d_end_offsets);
    }

    // --- Phase 2: multi-level merge tree ---
    SegLayout cur = build_initial_layout(row_offsets.data(), rows);

    float* bufs[2] = { input.data_ptr<float>(), output.data_ptr<float>() };
    int which = 0;
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
            // Only one segment — copy it to dst
            int seg_len = cur.lengths[0];
            int blocks = (seg_len + 255) / 256;
            copy_segment_kernel<<<blocks, 256>>>(
                src_buf + cur.starts[0], dst_buf, seg_len);
            cudaDeviceSynchronize();
            which = 1 - which;
            cur = next;
            continue;
        }

        // Upload pair info
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

        // Copy odd leftover segment
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

    // Final copy if result landed in input buffer
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
    Segmented sort: Phase 1 = cub::DeviceSegmentedRadixSort per row,
    Phase 2 = batched multi-level merge tree with merge-path parallel merge.
    Exploits the per-row-normal input structure: rows are centered at
    increasing means, so higher-level merges encounter minimal cross-row overlap.
    """
    input_tensor, output_tensor = data
    segment_module.segment_sort(input_tensor, output_tensor)
    return output_tensor