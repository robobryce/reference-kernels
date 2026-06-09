"""
Custom 4-pass LSD radix sort for sm_100/B200.
- 256 threads/block, 8 items/thread = 2048 items/block
- PTX ld.global.cg for all reads (L2 cache, bypass L1)
- Simple smem atomicAdd histogram 
- Optimized scatter: per-block counters only (no per-warp histogram)
- 3 kernels/pass: histogram + prefix_sum + scatter = 12 launches
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
constexpr int PASSES = 4;

static torch::Tensor d_scratch = {};
static torch::Tensor d_histogram = {};

void init_persistent() {
    int64_t n = 100000000LL;
    int64_t max_blocks = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    d_scratch = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    d_histogram = torch::zeros({max_blocks * NUM_BINS},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

__device__ __forceinline__ uint32_t ld_cg(const uint32_t* ptr) {
    uint32_t val;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}

// ---------------------------------------------------------------------------
// Histogram kernel
// ---------------------------------------------------------------------------
__global__ void histogram_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_hist,
    int64_t num_items, int pass, int num_blocks)
{
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    int shift = pass * 8;
    int64_t base = (int64_t)block_id * ITEMS_PER_BLOCK;

    __shared__ uint32_t h[NUM_BINS];
    for (int i = threadIdx.x; i < NUM_BINS; i += THREADS_PER_BLOCK) h[i] = 0;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t idx = base + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < num_items) {
            int d = (ld_cg(d_in + idx) >> shift) & 0xFF;
            atomicAdd(&h[d], 1);
        }
    }
    __syncthreads();

    int64_t hb = (int64_t)block_id * NUM_BINS;
    for (int i = threadIdx.x; i < NUM_BINS; i += THREADS_PER_BLOCK)
        d_hist[hb + i] = h[i];
}

// ---------------------------------------------------------------------------
// Prefix sum
// ---------------------------------------------------------------------------
__global__ void prefix_sum_kernel(uint32_t* d_hist, int num_blocks) {
    int bin = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ uint32_t chunk[THREADS_PER_BLOCK];
    __shared__ uint32_t ws[THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ uint32_t ct;

    uint32_t carry = 0;
    for (int cs = 0; cs < num_blocks; cs += THREADS_PER_BLOCK) {
        int sz = THREADS_PER_BLOCK;
        if (cs + sz > num_blocks) sz = num_blocks - cs;

        if (tid < sz) chunk[tid] = d_hist[((int64_t)(cs + tid) * NUM_BINS) + bin];
        else          chunk[tid] = 0;
        __syncthreads();

        uint32_t val = chunk[tid];
        for (int o = 1; o < WARP_SIZE; o <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, o);
            if ((tid % WARP_SIZE) >= o && tid < sz) val += n;
        }
        if (tid % WARP_SIZE == WARP_SIZE - 1 && tid < sz) ws[tid/WARP_SIZE] = val;
        __syncthreads();

        uint32_t wp = 0;
        int mw = tid / WARP_SIZE;
        if (mw > 0 && tid < sz)
            for (int w = 0; w < mw; w++) wp += ws[w];

        uint32_t scanned = (tid < sz) ? val + wp : 0;
        if (tid < sz) d_hist[((int64_t)(cs + tid) * NUM_BINS) + bin] = carry + scanned - chunk[tid];
        if (tid == sz - 1) ct = scanned;
        __syncthreads();
        carry += ct;
    }
}

// ---------------------------------------------------------------------------
// Scatter kernel - NO per-warp histogram. Uses per-block smem counters only.
// Items are scattered using block-level prefix + smem atomic counter.
// Eliminates redundant histogram rebuild.
// ---------------------------------------------------------------------------
__global__ void scatter_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    const uint32_t* __restrict__ d_pref,
    int64_t num_items, int pass, int num_blocks)
{
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int shift = pass * 8;
    int64_t block_base = (int64_t)block_id * ITEMS_PER_BLOCK;

    // Per-block running counters (one per bin), initialized from prefix sum
    __shared__ uint32_t ctr[NUM_BINS];

    // Load and initialize counters from prefix sum
    int64_t pref_base = (int64_t)block_id * NUM_BINS;
    for (int i = threadIdx.x; i < NUM_BINS; i += THREADS_PER_BLOCK) {
        ctr[i] = d_pref[pref_base + i];
    }
    __syncthreads();

    // Load items and scatter directly (no histogram rebuild)
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t idx = block_base + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < num_items) {
            uint32_t val = ld_cg(d_in + idx);
            int bin = (val >> shift) & 0xFF;
            uint32_t pos = atomicAdd(&ctr[bin], 1);
            d_out[pos] = val;
        }
    }
}

// ---------------------------------------------------------------------------
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    int64_t num_items = input.numel();
    int blocks = (int)((num_items + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();

    const uint32_t* in  = reinterpret_cast<const uint32_t*>(input.const_data_ptr<float>());
    uint32_t*       out = reinterpret_cast<uint32_t*>(output.data_ptr<float>());
    uint32_t*       scr = reinterpret_cast<uint32_t*>(d_scratch.data_ptr<int32_t>());
    uint32_t*       hist = reinterpret_cast<uint32_t*>(d_histogram.data_ptr<int32_t>());

    const uint32_t* src = in;
    uint32_t* dst = scr;
    for (int pass = 0; pass < PASSES; pass++) {
        if (pass == 3) dst = out;

        histogram_kernel<<<blocks, THREADS_PER_BLOCK, 0, s>>>(
            src, hist, num_items, pass, blocks);
        prefix_sum_kernel<<<NUM_BINS, THREADS_PER_BLOCK, 0, s>>>(
            hist, blocks);
        scatter_kernel<<<blocks, THREADS_PER_BLOCK, 0, s>>>(
            src, dst, hist, num_items, pass, blocks);

        src = dst;
        dst = (dst == scr) ? out : scr;
    }
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));
    return output;
}
"""

radix_cpp = r"""
#include <torch/extension.h>
void init_persistent();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

radix_module = load_inline(
    name='custom_radix_v4',
    cpp_sources=radix_cpp,
    cuda_sources=radix_source,
    functions=['sort_cuda', 'init_persistent'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)
radix_module.init_persistent()

def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    radix_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor
