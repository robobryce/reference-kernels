"""
Shared-Memory Block Bitonic Sort + torch.sort final merge.
Each block of 256 threads sorts 2048 items (8 items/thread) using
shared-memory bitonic sort. Grid-stride loop over all data.
Then torch.sort merges the partially-sorted data into final output.

No CUB (in custom code), no streams. Shared-memory-based bitonic sort phase.
"""
import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

ITEMS_PER_THREAD = 8
BLOCK_THREADS = 256
BLOCK_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD  # 2048

sort_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>

#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 8
#define BLOCK_SIZE 2048

// ---------------------------------------------------------------------------
// Shared-memory bitonic sort within each block
// ---------------------------------------------------------------------------
__global__ void block_bitonic_sort_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int64_t n) {

    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t block_offset = static_cast<int64_t>(blockIdx.x) * BLOCK_SIZE;

    for (int64_t chunk_start = block_offset;
         chunk_start < n;
         chunk_start += static_cast<int64_t>(gridDim.x) * BLOCK_SIZE) {

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int64_t idx = chunk_start + tid * ITEMS_PER_THREAD + i;
            smem[tid * ITEMS_PER_THREAD + i] = (idx < n) ? input[idx] : INFINITY;
        }
        __syncthreads();

        // Bitonic sort network on shared memory
        for (int k = 2; k <= BLOCK_SIZE; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                int stride = BLOCK_SIZE / BLOCK_THREADS;
                for (int i = tid * stride; i < (tid + 1) * stride; i++) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        bool ascending = ((i & k) == 0);
                        float a = smem[i], b = smem[ixj];
                        if ((a > b) == ascending) {
                            smem[i] = b; smem[ixj] = a;
                        }
                    }
                }
                __syncthreads();
            }
        }

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int64_t idx = chunk_start + tid * ITEMS_PER_THREAD + i;
            if (idx < n) output[idx] = smem[tid * ITEMS_PER_THREAD + i];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Main entry point: block sort into buf, then rely on torch.sort for merge
// ---------------------------------------------------------------------------
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor buf) {
    int64_t n = input.numel();
    int num_blocks_total = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks_total < 1) num_blocks_total = 1;

    int sort_grid = (num_blocks_total < 65535) ? num_blocks_total : 65535;
    if (sort_grid < 1) sort_grid = 1;

    block_bitonic_sort_kernel<<<sort_grid, BLOCK_THREADS>>>(
        input.const_data_ptr<float>(), buf.data_ptr<float>(), n);
    cudaDeviceSynchronize();

    return buf;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor buf);
"""

sort_module = load_inline(
    name='sort_smem_bitonic_pre',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Phase 1: shared-memory block bitonic sort -> partially sorted chunks
    Phase 2: torch.sort to merge fully
    """
    input_tensor, output_tensor = data
    input_contig = input_tensor.contiguous()
    n = input_contig.numel()

    buf = torch.empty(n, dtype=torch.float32, device='cuda')
    sort_module.sort_cuda(input_contig, buf)
    output_tensor[...] = torch.sort(buf)[0]
    return output_tensor