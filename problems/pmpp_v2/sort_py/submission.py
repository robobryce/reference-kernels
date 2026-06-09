"""CUB DeviceRadixSort with in-place encode/sort/decode on output.

Uses output tensor directly: copy+encode into output (viewed as int32),
CUB DeviceRadixSort in-place, decode. Only CUB temp workspace is extra.
"""

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void copy_encode_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ keys,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        int32_t v = __float_as_int(input[idx]);
        keys[idx] = v ^ ((v >> 31) | (int32_t)(1u << 31));
    }
}

__global__ void decode_kernel(
    int32_t* __restrict__ keys,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        int32_t v = keys[idx];
        keys[idx] = v ^ (~(v >> 31) | (int32_t)(1u << 31));
    }
}

torch::Tensor custom_kernel_fn(torch::Tensor data, torch::Tensor output) {
    int n = data.numel();
    if (n == 0) return output;

    // Use output buffer directly as int32 work buffer.
    auto keys = output.view(torch::kInt32);

    int threads = 256;
    int blocks  = std::min((n + threads - 1) / threads, 65535);

    // 1. Copy+encode: float in data -> sortable int32 in keys (= output).
    copy_encode_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        keys.data_ptr<int>(),
        n);

    // 2. CUB DeviceRadixSort in-place (sort all 32 bits, unsigned).
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        n, 0, 32);
    auto cub_temp = torch::empty(
        {static_cast<int64_t>(temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(data.device()));
    cub::DeviceRadixSort::SortKeys(
        cub_temp.data_ptr(), temp_bytes,
        keys.data_ptr<int>(),
        keys.data_ptr<int>(),
        n, 0, 32);

    // 3. Decode: sortable int32 -> float32 in-place.
    decode_kernel<<<blocks, threads>>>(keys.data_ptr<int>(), n);

    return output;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor custom_kernel_fn(torch::Tensor data, torch::Tensor output);
"""

module = load_inline(
    name="cub_sort_outbuf",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["custom_kernel_fn"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    module.custom_kernel_fn(data, output)
    return output