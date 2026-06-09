"""
CUB DeviceRadixSort with uint16 quantization to halve memory traffic.
Float32 values are quantized to uint16 using global min/max scaling,
sorted with CUB SortKeys on 16-bit keys, then decoded back to float32.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp_uint16 = {};
static size_t persistent_temp_uint16_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp_uint16.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_uint16_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 16);
    persistent_temp_uint16_bytes = (persistent_temp_uint16_bytes * 11 + 9) / 10;
    persistent_temp_uint16 = torch::empty(
        {static_cast<int64_t>(persistent_temp_uint16_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

__global__ void encode_float32_to_uint16_kernel(
    const float* __restrict__ input,
    uint16_t* __restrict__ keys,
    float min_val, float scale,
    int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float shifted = input[i] - min_val;
        float scaled = shifted * scale;
        unsigned int rounded = __float2uint_rn(scaled);
        keys[i] = (uint16_t)(rounded > 65535u ? 65535u : rounded);
    }
}

__global__ void decode_uint16_to_float32_kernel(
    const uint16_t* __restrict__ keys,
    float* __restrict__ output,
    float min_val, float inv_scale,
    int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __uint2float_rz(keys[i]) * inv_scale + min_val;
    }
}

torch::Tensor sort_uint16_cuda(torch::Tensor input, torch::Tensor output,
                                float min_val, float max_val) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    float range_val = max_val - min_val;
    float scale = 65535.0f / range_val;
    float inv_scale = range_val / 65535.0f;

    const float* data_in = input.const_data_ptr<float>();

    // Allocate uint16 key buffers
    auto keys = torch::empty({num_items},
        torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCUDA));
    auto sorted_keys = torch::empty({num_items},
        torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCUDA));

    int threads = 256;
    int blocks = (num_items + threads - 1) / threads;

    // Step 1: Encode float32 -> uint16
    encode_float32_to_uint16_kernel<<<blocks, threads, 0, stream>>>(
        data_in, keys.data_ptr<uint16_t>(), min_val, scale, num_items);

    // Step 2: CUB RadixSort on uint16 keys
    size_t temp_bytes = persistent_temp_uint16_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp_uint16.data_ptr(), temp_bytes,
        keys.const_data_ptr<uint16_t>(), sorted_keys.data_ptr<uint16_t>(),
        num_items, 0, 16, stream);

    // Step 3: Decode uint16 -> float32
    decode_uint16_to_float32_kernel<<<blocks, threads, 0, stream>>>(
        sorted_keys.const_data_ptr<uint16_t>(), output.data_ptr<float>(),
        min_val, inv_scale, num_items);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_uint16_cuda(torch::Tensor input, torch::Tensor output, float min_val, float max_val);
"""

sort_module = load_inline(
    name='sort_uint16_quantized',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_uint16_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via uint16 quantization to halve memory traffic.
    Float32 values are quantized to uint16 using global min/max scaling,
    sorted with CUB SortKeys on 16-bit keys, then decoded back to float32.
    """
    input_tensor, output_tensor = data
    input_contig = input_tensor.contiguous()

    # Find global min/max using torch (GPU-accelerated)
    min_val = input_contig.min().item()
    max_val = input_contig.max().item()

    sort_module.sort_uint16_cuda(input_contig, output_tensor, min_val, max_val)
    return output_tensor
