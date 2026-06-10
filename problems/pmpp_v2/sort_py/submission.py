"""
CUB DeviceRadixSort::SortKeys with int32 bitcast (no float conversion).
Since all data is positive IEEE 754, raw bits are in correct sort order.
Interpret float* as int*, sort keys-only, re-interpret back as float.
SUBTRACTIVE TEST (b): removed persistent temp — per-call tensor allocation measures contribution.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes,
        key_in, key_out, num_items,
        0, 32,
        stream);

    auto temp_tensor = torch::empty(
        {static_cast<int64_t>(temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cub::DeviceRadixSort::SortKeys(
        temp_tensor.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, 32,
        stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_int32_bitcast_percall',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    Per-call temp allocation (no persistent storage).
    SUBTRACTIVE TEST (b): no persistent temp.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor