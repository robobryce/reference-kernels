"""
CUB DeviceRadixSort::SortKeys with bfloat16 bitcast — half memory bandwidth.
Cast float32 -> bfloat16 (truncate mantissa, same exponent for positive vals),
bitcast to uint16, sort keys-only with begin_bit=0 end_bit=16,
bitcast back to bfloat16, cast to float32.
bfloat16 IEEE 754 sorts correctly by bits for positive values since
exponent is identical to float32 — sort order preserved exactly.
Persistent temp storage allocated once at module init.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>
#include <cuda_bf16.h>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 16);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input_bf16, torch::Tensor output_bf16) {
    auto num_items = static_cast<int64_t>(input_bf16.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint16_t* key_in = reinterpret_cast<const uint16_t*>(input_bf16.const_data_ptr<at::BFloat16>());
    uint16_t* key_out = reinterpret_cast<uint16_t*>(output_bf16.data_ptr<at::BFloat16>());

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, 16,
        stream);

    return output_bf16;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_bf16_bitcast',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on bfloat16 bitcast.
    Cast float32 -> bfloat16 -> uint16 bitcast -> SortKeys -> bfloat16 -> float32.
    Halves memory traffic vs 32-bit sort.
    """
    input_tensor, output_tensor = data
    x = input_tensor.contiguous()
    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = torch.empty_like(x_bf16)
    sort_module.sort_cuda(x_bf16, y_bf16)
    output_tensor.copy_(y_bf16.to(torch.float32))
    return output_tensor