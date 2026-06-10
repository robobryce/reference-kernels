"""
Per-row Batcher's odd-even merge sort in shared memory per the brief's spec.
Each CUDA block sorts one row (~10K items) entirely in shared memory.
No CUB needed — just __syncthreads + smem swaps.

Multi-row merge: bucket by integer floor (10 bits, 1024 buckets),
then each bucket independently merges sorted row segments (~19 rows per bucket).
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

// --------------------------------------------------------------------------
// Per-row Batcher's odd-even merge sort kernel.
// One CUDA block per row. 10240 items (max row len for 100M shape).
// 256 threads, 40 items/thread.
// Items are sorted in shared memory.
// --------------------------------------------------------------------------

// Macro-based Batcher's sort: at each comparator stage, a pair (i, j) is compared.
// For each stage, threads cooperate: thread t handles its 40 items at positions
// t*40 .. t*40+39 and either their partners at (t XOR d)*40 .. for distance d.

#define N_ITEMS 10240
#define ITEMS_PER_THREAD 40
#define N_THREADS 256

__global__ void batcher_per_row_sort_kernel(
    const float* __restrict__ input,
    int N,
    int row_len,       // logical elements per row
    float* __restrict__ output)
{
    // One block = one row
    int row = blockIdx.x;
    int row_start = row * row_len;
    if (row_start >= N) return;
    int rlen = min(row_len, N - row_start);

    extern __shared__ float smem[];  // N_ITEMS floats

    int tid = threadIdx.x;

    // Load row into shared memory (pad with INF for partial rows)
    const float* row_ptr = input + row_start;
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        int idx = tid * ITEMS_PER_THREAD + k;
        smem[idx] = (idx < rlen) ? row_ptr[idx] : INFINITY;
    }
    __syncthreads();

    // Batcher's odd-even merge sort on N_ITEMS elements
    // Standard iteration: for (p = 1; p < N; p *= 2)
    //     for (k = p; k >= 1; k /= 2)
    //         compare adjacent groups of size k, distance k
    for (int p = 1; p < N_ITEMS; p *= 2) {
        for (int k = p; k >= 1; k /= 2) {
            // Distance between compared elements
            int d = k;

            // Each thread handles its ITEMS_PER_THREAD elements
            for (int item = 0; item < ITEMS_PER_THREAD; item++) {
                int i = tid * ITEMS_PER_THREAD + item;
                int j = i ^ d;  // bitwise XOR with distance

                // Only compare in one direction (i < j)
                if (i >= N_ITEMS || j >= N_ITEMS) continue;

                // Determining whether to sort ascending (i < j XOR i & p)
                // For odd positions relative to p: reversed sort
                int dir_asc = !(i & p);

                // i gets the min, j gets the max if ascending; swap if descending
                if (i < j) {
                    float a = smem[i];
                    float b = smem[j];
                    if (dir_asc) {
                        if (a > b) { smem[i] = b; smem[j] = a; }
                    } else {
                        if (a < b) { smem[i] = b; smem[j] = a; }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write sorted row to output
    float* out_ptr = output + row_start;
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        int idx = tid * ITEMS_PER_THREAD + k;
        if (idx < rlen) {
            out_ptr[idx] = smem[idx];
        }
    }
}

// --------------------------------------------------------------------------
// Compute floor boundaries per row (runs after per-row sort)
// --------------------------------------------------------------------------
__global__ void floor_boundaries_kernel(
    const float* __restrict__ sorted_data,
    int N,
    int row_len,
    int* __restrict__ row_floor_offsets,  // [num_rows * 12] floor start per floor per row
    int* __restrict__ row_floor_counts,   // [num_rows * 12]
    int* __restrict__ row_floor_vals,     // [num_rows * 12]
    int* __restrict__ row_nf,             // [num_rows]
    int max_floors)
{
    int row = blockIdx.x;
    int start = row * row_len;
    if (start >= N) { row_nf[row] = 0; return; }
    int rlen = min(row_len, N - start);
    if (rlen <= 0) { row_nf[row] = 0; return; }

    const float* rptr = sorted_data + start;
    int fidx = 0;
    int cur_floor = (int)__float2int_rz(rptr[0]);
    int seg_start = 0;

    for (int i = 1; i < rlen && fidx < max_floors; i++) {
        int f = (int)__float2int_rz(rptr[i]);
        if (f != cur_floor) {
            int off = row * max_floors + fidx;
            row_floor_vals[off] = cur_floor;
            row_floor_offsets[off] = seg_start;
            row_floor_counts[off] = i - seg_start;
            fidx++;
            cur_floor = f;
            seg_start = i;
        }
    }
    if (fidx < max_floors) {
        int off = row * max_floors + fidx;
        row_floor_vals[off] = cur_floor;
        row_floor_offsets[off] = seg_start;
        row_floor_counts[off] = rlen - seg_start;
        fidx++;
    }
    row_nf[row] = fidx;
}

// --------------------------------------------------------------------------
// Per-bucket merge kernel (1024 buckets, 10-bit)
// --------------------------------------------------------------------------
__global__ void bucket_merge_kernel(
    const float* __restrict__ sorted_data,
    int N,
    int row_len,
    int num_rows,
    const int* __restrict__ floor_vals,
    const int* __restrict__ floor_offsets,
    const int* __restrict__ floor_counts,
    const int* __restrict__ row_nf,
    int max_floors,
    int min_floor, int bucket_width, int num_buckets,
    const int* __restrict__ bucket_out_offsets,
    float* __restrict__ output)
{
    int bkt = blockIdx.x;
    if (bkt >= num_buckets) return;
    int f_start = min_floor + bkt * bucket_width;
    int f_end = min(min_floor + (bkt + 1) * bucket_width, min_floor + (1 << 14));
    int out_start = bucket_out_offsets[bkt];
    int out_end = bucket_out_offsets[bkt + 1];
    int total = out_end - out_start;
    if (total <= 0) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    __shared__ int seg_count;
    __shared__ int seg_rows[64];      // up to 64 segments per bucket
    __shared__ int seg_starts[64];   // absolute offset in sorted_data
    __shared__ int seg_lens[64];

    if (tid == 0) seg_count = 0;
    __syncthreads();

    // Each thread scans a range of rows for floors in [f_start, f_end)
    int rows_per_t = (num_rows + bdim - 1) / bdim;
    for (int r = tid * rows_per_t; r < min((tid+1)*rows_per_t, num_rows); r++) {
        int nf = row_nf[r];
        for (int j = 0; j < nf; j++) {
            int fv = floor_vals[r * max_floors + j];
            if (fv >= f_start && fv < f_end) {
                int slot = atomicAdd(&seg_count, 1);
                if (slot < 64) {
                    seg_rows[slot] = r;
                    seg_starts[slot] = r * row_len + floor_offsets[r * max_floors + j];
                    seg_lens[slot] = floor_counts[r * max_floors + j];
                }
            }
        }
    }
    __syncthreads();

    int ns = min(seg_count, 64);
    if (ns == 0) return;

    // Sequential k-way merge for each thread's output range
    int chunk = (total + bdim - 1) / bdim;
    int my_start = tid * chunk;
    int my_end = min(my_start + chunk, total);
    if (my_start >= my_end) return;
    int my_len = my_end - my_start;

    // Cursor positions in each segment
    int cur[64];
    for (int s = 0; s < ns; s++) cur[s] = 0;

    // Advance to my_start
    int gpos = 0;
    while (gpos < my_start) {
        float best = FLT_MAX;
        int best_s = -1;
        for (int s = 0; s < ns; s++) {
            if (cur[s] < seg_lens[s]) {
                float v = sorted_data[seg_starts[s] + cur[s]];
                if (v < best) { best = v; best_s = s; }
            }
        }
        if (best_s < 0) break;
        cur[best_s]++;
        gpos++;
    }

    // Emit
    int opos = 0;
    while (gpos < my_end) {
        float best = FLT_MAX;
        int best_s = -1;
        for (int s = 0; s < ns; s++) {
            if (cur[s] < seg_lens[s]) {
                float v = sorted_data[seg_starts[s] + cur[s]];
                if (v < best) { best = v; best_s = s; }
            }
        }
        if (best_s < 0) break;
        output[out_start + my_start + opos] = best;
        cur[best_s]++;
        opos++;
        gpos++;
    }
}

// --------------------------------------------------------------------------
// Host orchestration
// --------------------------------------------------------------------------
torch::Tensor batcher_bucket_sort(torch::Tensor input, torch::Tensor output) {
    int N = (int)input.numel();
    if (N <= 1) { if (N==1) output[0]=input[0].item<float>(); return output; }

    cudaStream_t stream = (cudaStream_t)0;

    int num_rows = (int)sqrt((double)N);
    int row_len = (N + num_rows - 1) / num_rows;

    // Phase 1: Per-row Batcher's sort
    auto sorted_data = torch::empty_like(input);

    int smem_size = N_ITEMS * sizeof(float);
    batcher_per_row_sort_kernel<<<num_rows, N_THREADS, smem_size, stream>>>(
        input.const_data_ptr<float>(), N, row_len,
        sorted_data.data_ptr<float>());
    cudaStreamSynchronize(stream);

    // Phase 2: Floor boundaries
    const int MAX_FLOORS = 12;
    int num_floor_slots = num_rows * MAX_FLOORS;

    auto d_fvals = torch::zeros({num_floor_slots},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto d_foffs = torch::zeros({num_floor_slots},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto d_fcounts = torch::zeros({num_floor_slots},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto d_rnf = torch::zeros({num_rows},
        torch::dtype(torch::kInt32).device(torch::kCUDA));

    floor_boundaries_kernel<<<num_rows, 64, 0, stream>>>(
        sorted_data.const_data_ptr<float>(), N, row_len,
        d_foffs.data_ptr<int>(),
        d_fcounts.data_ptr<int>(),
        d_fvals.data_ptr<int>(),
        d_rnf.data_ptr<int>(),
        MAX_FLOORS);
    cudaStreamSynchronize(stream);

    // Phase 3: Bucket histogram on CPU
    auto h_rnf = d_rnf.cpu();
    auto h_fvals = d_fvals.cpu();
    auto h_fcounts = d_fcounts.cpu();

    int min_floor = INT_MAX, max_floor = INT_MIN;
    for (int r = 0; r < num_rows; r++) {
        for (int j = 0; j < h_rnf.const_data_ptr<int>()[r]; j++) {
            int fv = h_fvals.const_data_ptr<int>()[r * MAX_FLOORS + j];
            if (fv < min_floor) min_floor = fv;
            if (fv > max_floor) max_floor = fv;
        }
    }

    // 10-bit bucketing: 1024 buckets
    int bucket_width = max(1, (max_floor - min_floor + 1024) / 1024);
    int num_buckets = (max_floor - min_floor + bucket_width) / bucket_width;

    std::vector<int> bucket_totals(num_buckets, 0);
    for (int r = 0; r < num_rows; r++) {
        int nf = h_rnf.const_data_ptr<int>()[r];
        for (int j = 0; j < nf; j++) {
            int fv = h_fvals.const_data_ptr<int>()[r * MAX_FLOORS + j];
            int bkt = (fv - min_floor) / bucket_width;
            if (bkt < num_buckets)
                bucket_totals[bkt] += h_fcounts.const_data_ptr<int>()[r * MAX_FLOORS + j];
        }
    }

    std::vector<int> out_offs(num_buckets + 1, 0);
    for (int b = 0; b < num_buckets; b++)
        out_offs[b+1] = out_offs[b] + bucket_totals[b];

    if (out_offs[num_buckets] != N) {
        printf("WARN: bucket total %d != N %d\n", out_offs[num_buckets], N);
        // fallback
        auto s = torch::sort(input);
        cudaMemcpy(output.data_ptr<float>(), std::get<0>(s).const_data_ptr<float>(),
                   N*sizeof(float), cudaMemcpyDeviceToDevice);
        return output;
    }

    auto d_out_offs = torch::from_blob(out_offs.data(), {num_buckets + 1},
        torch::dtype(torch::kInt32)).clone().to(torch::kCUDA);

    // Phase 4: Per-bucket merge
    bucket_merge_kernel<<<num_buckets, 256, 0, stream>>>(
        sorted_data.const_data_ptr<float>(), N, row_len, num_rows,
        d_fvals.const_data_ptr<int>(),
        d_foffs.const_data_ptr<int>(),
        d_fcounts.const_data_ptr<int>(),
        d_rnf.const_data_ptr<int>(),
        MAX_FLOORS,
        min_floor, bucket_width, num_buckets,
        d_out_offs.const_data_ptr<int>(),
        output.data_ptr<float>());
    cudaStreamSynchronize(stream);

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor batcher_bucket_sort(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='batcher_bucket_sort',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batcher_bucket_sort'],
    extra_cuda_cflags=['-O3'],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    inp = input_tensor.contiguous()
    sort_module.batcher_bucket_sort(inp, output_tensor)
    return output_tensor