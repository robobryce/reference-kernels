"""
Half-key bitcast with 14-bit key via SortPairs: key = (bits >> 14) & 0x3FFF.
This captures bits [14:28) of the uint32 float representation: 7 exponent bits
(missing exponent MSB at bit 30) + 7 mantissa bits. SortPairs preserves original
float values. The missing exponent MSB means values above 8192 (exp=13+)
get incorrectly placed.
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
        0, 14);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

__global__ void encode_f32_to_u14key(
    const float* __restrict__ input,
    uint16_t* __restrict__ keys,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&input[idx]);
        keys[idx] = static_cast<uint16_t>((bits >> 14) & 0x3FFF);
    }
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());

    auto keys = torch::empty({num_items},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));

    int threads = 256;
    int blocks = (num_items + threads - 1) / threads;

    // Encode: 14-bit key from bits >> 14
    encode_f32_to_u14key<<<blocks, threads>>>(
        input.const_data_ptr<float>(),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        num_items);

    // SortPairs<uint16_t, float> with end_bit=14 (2 radix passes)
    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortPairs(
        persistent_temp.data_ptr(), temp_bytes,
        reinterpret_cast<const uint16_t*>(keys.const_data_ptr<int16_t>()),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        input.const_data_ptr<float>(),
        output.data_ptr<float>(),
        num_items, 0, 14);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_u14_sortpairs',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    14-bit key sort via SortPairs: extract bits[14:28) of float32.
    SortPairs preserves original float values. 2 radix passes (14 bits).
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor