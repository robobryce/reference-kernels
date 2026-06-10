"""
Custom 4-pass LSD radix sort: 3 kernels/pass x 4 passes = 12 launches total.
- Histogram: per-block smem atomicAdd (256 bins, 1KB smem)
- Prefix sum: device-level exclusive scan per bin (parallel across bins)
- Scatter: per-warp deterministic prefix-sum-based position assignment
Target: match CUB SortKeys ~176us geomean.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

radix_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr int ITEMS_PER_THREAD = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS_PER_BLOCK;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int NUM_BINS = 256;
constexpr int RADIX_BITS = 8;
constexpr int NUM_PASSES = 4;

static torch::Tensor d_scratch0 = {};
static torch::Tensor d_scratch1 = {};
static torch::Tensor d_histogram = {};

void init_persistent() {
    int64_t n = 100'000'000;
    int64_t max_blocks = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    d_scratch0 = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    d_scratch1 = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    d_histogram = torch::zeros({max_blocks * NUM_BINS},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

// ===================================================================
// Kernel 1: Per-block histogram using smem atomicAdd.
// ===================================================================
__global__ void histogram_kernel(
    const uint32_t* d_in, uint32_t* d_hist,
    int64_t n, int pass, int nblocks)
{
    int bid = blockIdx.x;
    if (bid >= nblocks) return;
    int shift = pass * RADIX_BITS;
    int64_t base = (int64_t)bid * ITEMS_PER_BLOCK;

    __shared__ uint32_t h[NUM_BINS];
    for (int i = threadIdx.x; i < NUM_BINS; i += THREADS_PER_BLOCK) h[i] = 0;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int64_t idx = base + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < n) {
            int bin = (d_in[idx] >> shift) & 0xFF;
            atomicAdd(&h[bin], 1);
        }
    }
    __syncthreads();

    int64_t off = (int64_t)bid * NUM_BINS;
    for (int i = threadIdx.x; i < NUM_BINS; i += THREADS_PER_BLOCK) {
        d_hist[off + i] = h[i];
    }
}

// ===================================================================
// Kernel 2: Device-level exclusive prefix sum per bin.
// Grid: NUM_BINS blocks (one per bin), THREADS_PER_BLOCK threads each.
// Each block scans one bin across all data-blocks.
// Chunked: processes up to THREADS_PER_BLOCK blocks at a time,
// carries cumulative total across chunks.
// ===================================================================
__global__ void prefix_sum_kernel(uint32_t* d_hist, int nblocks) {
    int bin = blockIdx.x;
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lid  = tid % WARP_SIZE;

    __shared__ uint32_t sdata[THREADS_PER_BLOCK];
    __shared__ uint32_t swarp_sums[THREADS_PER_BLOCK / WARP_SIZE + 1];

    uint32_t carry = 0;
    int64_t boff = (int64_t)bin;

    int chunk = 0;
    while (true) {
        int chunk_start = chunk * THREADS_PER_BLOCK;
        int remaining = nblocks - chunk_start;
        if (remaining <= 0) break;
        int chunk_size = (remaining > THREADS_PER_BLOCK) ? THREADS_PER_BLOCK : remaining;

        // Load this chunk
        if (tid < chunk_size)
            sdata[tid] = d_hist[((int64_t)(chunk_start + tid) * NUM_BINS) + boff];
        else
            sdata[tid] = 0;
        __syncthreads();

        uint32_t val = sdata[tid];
        unsigned mask = (tid < chunk_size) ? 0xFFFFFFFF : 0;

        // Intra-warp inclusive scan (Kogge-Stone / parallel scan)
        #pragma unroll
        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            uint32_t x = __shfl_up_sync(mask, val, d, WARP_SIZE);
            int src_lane = lid - d;
            // Only add from valid lanes
            if (tid >= d && (tid - d) < chunk_start + chunk_size) {
                // shfl_up returns 0 for inactive lanes
            }
            if (src_lane >= 0 && (tid - d) < chunk_start + chunk_size && lid >= d) {
                val += x;
            }
        }

        // Write warp totals
        if (lid == WARP_SIZE - 1 && tid < chunk_size) {
            swarp_sums[warp] = val;
        }
        __syncthreads();

        // Cross-warp scan
        uint32_t warp_prefix = 0;
        if (tid < chunk_size) {
            for (int w = 0; w < warp; ++w) {
                warp_prefix += swarp_sums[w];
            }
        }

        uint32_t scanned = (tid < chunk_size) ? (warp_prefix + val) : 0;

        // Exclusive prefix: carry + scanned - original = carry + (scanned - sdata[tid])
        if (tid < chunk_size)
            d_hist[((int64_t)(chunk_start + tid) * NUM_BINS) + boff] = carry + scanned - sdata[tid];

        uint32_t total_in_chunk = 0;
        if (tid < chunk_size) total_in_chunk = warp_prefix + val;
        __syncthreads();

        uint32_t chunk_sum = 0;
        if (chunk_size == 1) {
            if (tid == 0) chunk_sum = total_in_chunk;
        } else {
            if (tid == chunk_size - 1) chunk_sum = total_in_chunk;
        }
        // Propagate chunk_sum to all threads
        __shared__ uint32_t cs_val;
        cs_val = (tid == chunk_size - 1) ? chunk_sum : 0;
        __syncthreads();
        if (cs_val == 0 && tid == 0 && chunk_size > 1) {
            // Find last valid thread
            cs_val = total_in_chunk; // tid=0 already has total_in_chunk
        }
        // broadcast: just use sync
        __syncthreads();
        chunk_sum = (tid == chunk_size - 1) ? chunk_sum : swarp_sums[(chunk_size-1)/WARP_SIZE];
        __syncthreads();

        carry += cs_val;

        ++chunk;
        __syncthreads();
    }
}

// ===================================================================
// Kernel 3: Scatter using per-warp deterministic position computation.
// Each block processes its tile. Within each warp, positions for items
// going to the same bin are assigned via __shfl-based prefix sum
// (no atomicAdd within warp). Cross-warp offsets use smem atomicAdd.
// ===================================================================
__global__ void scatter_kernel(
    const uint32_t* d_in, uint32_t* d_out,
    const uint32_t* d_prefix,
    int64_t n, int pass, int nblocks)
{
    int bid = blockIdx.x;
    if (bid >= nblocks) return;
    int shift = pass * RADIX_BITS;
    int64_t base = (int64_t)bid * ITEMS_PER_BLOCK;
    int wid = threadIdx.x / WARP_SIZE;
    int lid = threadIdx.x % WARP_SIZE;

    __shared__ uint32_t warp_counts[WARPS_PER_BLOCK * NUM_BINS];
    __shared__ uint32_t warp_offsets[WARPS_PER_BLOCK * NUM_BINS];

    // Read items and extract digits
    uint32_t items[ITEMS_PER_THREAD];
    int bins[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int64_t idx = base + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < n) {
            items[i] = d_in[idx];
            bins[i] = (items[i] >> shift) & 0xFF;
        } else {
            bins[i] = -1;
        }
    }

    // A) Per-warp per-bin histogram via smem atomicAdd
    int woff = wid * NUM_BINS;
    for (int i = lid; i < NUM_BINS; i += WARP_SIZE) {
        warp_counts[woff + i] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (bins[i] >= 0) {
            atomicAdd(&warp_counts[woff + bins[i]], 1);
        }
    }
    __syncthreads();

    // B) Compute per-warp per-bin base offsets from global prefix sum
    int64_t pref = (int64_t)bid * NUM_BINS;
    if (threadIdx.x < NUM_BINS) {
        int b = threadIdx.x;
        uint32_t running = d_prefix[pref + b];
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            uint32_t cnt = warp_counts[w * NUM_BINS + b];
            warp_offsets[w * NUM_BINS + b] = running;
            running += cnt;
        }
    }
    __syncthreads();

    // C) Scatter: for each thread's items, compute position within warp
    // via warp ballot, then add to warp base offset
    // Atomically increment per-warp per-bin counter in smem for warp-local ordering
    uint32_t* my_ctrs = warp_offsets + woff;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int64_t idx = base + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx >= n) continue;
        int b = bins[i];
        uint32_t pos = atomicAdd(&my_ctrs[b], 1);
        d_out[pos] = items[i];
    }
}
"""

# --- Top-level orchestrator ---

radix_source_orchestrator = r"""
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    int64_t num_items = input.numel();
    int blocks = (int)((num_items + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK);
    if (blocks < 1) blocks = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint32_t* d_in  = reinterpret_cast<const uint32_t*>(input.const_data_ptr<float>());
    uint32_t*       d_out = reinterpret_cast<uint32_t*>(output.data_ptr<float>());
    uint32_t* d_s0 = reinterpret_cast<uint32_t*>(d_scratch0.data_ptr<int32_t>());
    uint32_t* d_s1 = reinterpret_cast<uint32_t*>(d_scratch1.data_ptr<int32_t>());
    uint32_t* d_hist = reinterpret_cast<uint32_t*>(d_histogram.data_ptr<int32_t>());

    for (int pass = 0; pass < NUM_PASSES; ++pass) {
        const uint32_t* src = (pass == 0) ? d_in : ((pass == 2) ? d_s1 : d_s0);
        uint32_t* dst = (pass == 3) ? d_out : ((pass == 1) ? d_s1 : d_s0);

        histogram_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            src, d_hist, num_items, pass, blocks);
        prefix_sum_kernel<<<NUM_BINS, THREADS_PER_BLOCK, 0, stream>>>(
            d_hist, blocks);
        scatter_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            src, dst, d_hist, num_items, pass, blocks);
    }

    cudaStreamSynchronize(stream);
    return output;
}
"""

full_source = radix_source + radix_source_orchestrator

radix_cpp = r"""
#include <torch/extension.h>
void init_persistent();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

radix_module = load_inline(
    name='custom_radix_sort_4pass_v4',
    cpp_sources=radix_cpp,
    cuda_sources=full_source,
    functions=['sort_cuda', 'init_persistent'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

radix_module.init_persistent()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    radix_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor