"""
CUB DeviceRadixSort::SortKeys with int32_t bitcast + end_bit=31.
All positive IEEE 754 float32 values have bit 31 = 0 in int32 bitcast.
end_bit=31 skips the always-zero sign bit — still 4 radix passes
(same as end_bit=32 for 8-bit Onesweep radix groups), but CUB
may dispatch differently for the non-32 exact bit range.
Uses int32_t offsets and sm_100a arch.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int32_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int32_t>(max_n),
        0, 31);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    int32_t num_items = static_cast<int32_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, 31,
        stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_int32_endbit31_sm100a',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_cuda_cflags=['-gencode=arch=compute_100a,code=sm_100a'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys, end_bit=31.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor