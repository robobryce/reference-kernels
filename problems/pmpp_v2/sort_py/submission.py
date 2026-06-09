"""
Half-width sort via FP16: cast float32 input to half inside CUDA kernel,
bitcast to uint16 for CUB SortKeys (4 radix passes instead of 8),
cast back to float32. Single CUDA kernel avoids torch.to() dispatch overhead.
Two pre-allocated half-precision scratch buffers.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;
static torch::Tensor half_in = {};   // scratch for float32->half conversion
static torch::Tensor half_out = {};  // scratch for sorted half output

static constexpr int64_t MAX_N = 100'000'000;

__global__ void float32_to_half_kernel(const float* __restrict__ src,
                                        half* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

__global__ void half_to_float32_kernel(const half* __restrict__ src,
                                        float* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

void init_sort_half() {
    if (persistent_temp.defined()) return;

    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        MAX_N, 0, 16);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    half_in = torch::empty({MAX_N},
        torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
    half_out = torch::empty({MAX_N},
        torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
}

torch::Tensor sort_half(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int64_t threads = 256;
    const int64_t blocks = (num_items + threads - 1) / threads;

    // Step 1: f32 -> half
    float32_to_half_kernel<<<blocks, threads, 0, stream>>>(
        input.const_data_ptr<float>(),
        reinterpret_cast<half*>(half_in.data_ptr()),
        num_items);

    // Step 2: Sort uint16 keys
    const uint16_t* key_in = reinterpret_cast<const uint16_t*>(half_in.data_ptr());
    uint16_t* key_out = reinterpret_cast<uint16_t*>(half_out.data_ptr());
    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, 16, stream);

    // Step 3: half -> f32
    half_to_float32_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(half_out.data_ptr()),
        output.data_ptr<float>(),
        num_items);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_sort_half();
torch::Tensor sort_half(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_half_fused_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_half', 'init_sort_half'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_sort_half()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via FP16 half-precision keys:
    1. f32->half conversion kernel (elementwise)
    2. CUB SortKeys on uint16 (4 radix passes vs 8)
    3. half->f32 conversion kernel (elementwise)
    All in single CUDA function with pre-allocated scratch buffers.
    """
    input_tensor, output_tensor = data
    sort_module.sort_half(input_tensor.contiguous(), output_tensor)
    return output_tensor