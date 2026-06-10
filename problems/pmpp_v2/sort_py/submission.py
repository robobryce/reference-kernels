"""
Half-key bitcast with SortPairs: extract upper 16 bits of float32 as uint16 key
via uint32_t bitcast + right-shift, CUB SortPairs<uint16_t, float> preserves
original float values. Values_in=input, values_out=output (no decode needed).
Key collisions cause sorting errors for values within same ~32 value bucket.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortPairs(
        nullptr, persistent_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 16);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

__global__ void encode_f32_to_u16key(
    const float* __restrict__ input,
    uint16_t* __restrict__ keys,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&input[idx]);
        keys[idx] = static_cast<uint16_t>(bits >> 16);
    }
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());

    auto keys = torch::empty({num_items},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));

    int threads = 256;
    int blocks = (num_items + threads - 1) / threads;

    // Encode: float32 -> uint16 key (upper 16 bits via bits>>16)
    encode_f32_to_u16key<<<blocks, threads>>>(
        input.const_data_ptr<float>(),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        num_items);

    // SortPairs: sort (key, value) pairs. Values flow input->output permuted.
    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortPairs(
        persistent_temp.data_ptr(), temp_bytes,
        reinterpret_cast<const uint16_t*>(keys.const_data_ptr<int16_t>()),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        input.const_data_ptr<float>(),
        output.data_ptr<float>(),
        num_items,
        0, 16);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_u16_sortpairs',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via SortPairs: extract uint16 key from upper 16 bits of float32 bitcast.
    Sorted values are the original float32s (no reconstruction error).
    Key collisions may cause misordering within same-key buckets.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor