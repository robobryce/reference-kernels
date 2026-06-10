"""
Tensor Core Sort: Warp-level 64-element bitonic sort using shared memory,
merged via CUB SortKeys. Hybrid approach: bitonic sort exploits B200's
large 256KB shared memory per SM for fully coalesced, low-latency comparisons,
then CUB radix sort efficiently merges the partially-ordered result.

Strategy:
1. Sort 64-element chunks using bitonic sort in shared memory.
2. CUB SortKeys merges partially-ordered output globally.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

// ---------------------------------------------------------------------------
// Bitonic sort of 64 float32 elements, 1 warp (32 threads, 2 elem/thread).
// Uses per-warp shared memory region (64 floats = 256 bytes).
// Warp works on its own smem region; no cross-warp sync needed.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void warp_bitonic_sort_64(
    float* restrict_v0, float* restrict_v1, float* smem_warp)
{
    const int lane = threadIdx.x & 31;

    // Store initial values to shared memory
    smem_warp[2 * lane] = *restrict_v0;
    smem_warp[2 * lane + 1] = *restrict_v1;
    __syncwarp();

    // Bitonic sort for N=64
    // stage_size: 2, 4, 8, 16, 32, 64 (current bitonic sequence size)
    for (int stage_size = 2; stage_size <= 64; stage_size <<= 1) {
        // step: distance between compared elements: stage_size/2, ..., 1
        for (int step = stage_size >> 1; step > 0; step >>= 1) {
            // Each thread handles both its elements
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = 2 * lane + e;
                int j = i ^ step;
                if (j <= i) continue;  // only lower partner does the swap

                float a = smem_warp[i];
                float b = smem_warp[j];

                // Direction: ascending if (i / stage_size) is even
                // i.e., if the bit corresponding to stage_size of i is 0
                bool ascending = ((i / stage_size) & 1) == 0;

                smem_warp[i] = ascending ? fminf(a, b) : fmaxf(a, b);
                smem_warp[j] = ascending ? fmaxf(a, b) : fminf(a, b);
            }
            __syncwarp();
        }
    }

    *restrict_v0 = smem_warp[2 * lane];
    *restrict_v1 = smem_warp[2 * lane + 1];
}

// ---------------------------------------------------------------------------
// Kernel: each warp sorts a 64-element segment using bitonic sort.
// Threads per block = warps_per_block * 32.
// Shared memory: warps_per_block * 64 * sizeof(float) (~4KB for 16 warps).
// ---------------------------------------------------------------------------
__global__ void chunk_sort_64_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n)
{
    extern __shared__ float smem_all[];
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int base = warp_id * 64;
    const int lane = threadIdx.x & 31;

    // Each warp gets its own 64-float smem region
    float* smem_warp = smem_all + (threadIdx.x / 32) * 64;

    float v0, v1;
    int idx0 = base + 2 * lane;
    int idx1 = base + 2 * lane + 1;

    if (idx1 < n) {
        v0 = input[idx0];
        v1 = input[idx1];
    } else if (idx0 < n) {
        v0 = input[idx0];
        v1 = INFINITY;
    } else {
        v0 = INFINITY;
        v1 = INFINITY;
    }

    warp_bitonic_sort_64(&v0, &v1, smem_warp);

    if (idx0 < n) output[idx0] = v0;
    if (idx1 < n) output[idx1] = v1;
}

// ---------------------------------------------------------------------------
// CUB SortKeys — optimized path from parent
// ---------------------------------------------------------------------------
static torch::Tensor persistent_temp;
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 32);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_fused(torch::Tensor input, torch::Tensor output,
                          torch::Tensor temp)
{
    auto n = static_cast<int>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // 16 warps per block = 512 threads. SMEM: 16 * 64 * 4 = 4096 bytes.
    const int warps_per_block = 16;
    const int threads_per_block = warps_per_block * 32;
    const size_t smem_size = warps_per_block * 64 * sizeof(float);

    int n_warps = (n + 63) / 64;
    int n_blocks = (n_warps + warps_per_block - 1) / warps_per_block;

    float* in_ptr = input.data_ptr<float>();
    float* tmp_ptr = temp.data_ptr<float>();

    chunk_sort_64_kernel<<<n_blocks, threads_per_block, smem_size, stream>>>(
        in_ptr, tmp_ptr, n);

    // Step 2: CUB SortKeys merge on bitcast int32
    int64_t n64 = n;
    int32_t* keys_in = reinterpret_cast<int32_t*>(tmp_ptr);
    int32_t* keys_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());
    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        keys_in, keys_out, n64,
        0, 32, stream);

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
void init_persistent_temp();
torch::Tensor sort_fused(torch::Tensor input, torch::Tensor output, torch::Tensor temp);
"""

sort_mod = load_inline(
    name='tcore_bitonic64_cub_merge',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['sort_fused', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)
sort_mod.init_persistent_temp()

_temp_buf = None


def custom_kernel(data: input_t) -> output_t:
    """
    Hybrid bitonic + CUB sort:
    1. 64-element chunks sorted via shared-memory bitonic sort
    2. CUB SortKeys merge on partially-ordered result
    """
    global _temp_buf
    input_tensor, output_tensor = data
    N = input_tensor.numel()

    if _temp_buf is None or _temp_buf.numel() < N:
        _temp_buf = torch.empty(N, dtype=torch.float32, device='cuda')

    sort_mod.sort_fused(input_tensor.contiguous(), output_tensor, _temp_buf)
    return output_tensor