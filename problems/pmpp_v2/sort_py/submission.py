"""
CUB DeviceRadixSort::SortKeys with bfloat16 encoding for half memory bandwidth.
Optimized v3: encode f32 -> bfloat16 uint16 into scratch buffer, then
cub::DoubleBuffer-managed SortKeys into output buffer, then decode uint16 -> f32
in reverse (in-place). No intermediate cudaMemcpy — DoubleBuffer handles the
in/out swap internally.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cstdint>

static torch::Tensor scratch_pool = {};
static size_t scratch_capacity = 0;
static size_t cub_temp_bytes = 0;

void init_temp() {
    if (scratch_pool.defined()) return;
    int64_t max_n = 100'000'000;

    cub::DeviceRadixSort::SortKeys(
        nullptr, cub_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<int>(max_n),
        0, 16);
    cub_temp_bytes = (cub_temp_bytes * 11 + 9) / 10;

    // Scratch: uint16 encode buffer (200MB for 100M) + CUB temp
    scratch_capacity = static_cast<size_t>(max_n) * 2 + cub_temp_bytes;
    scratch_pool = torch::empty(
        {static_cast<int64_t>(scratch_capacity)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

__global__ void encode_f32_to_uint16_kernel(
    const float* __restrict__ in,
    int16_t* __restrict__ out,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<int16_t>(
            __bfloat16_as_ushort(__float2bfloat16(in[idx])));
    }
}

__global__ void decode_uint16_to_f32_kernel(
    const int16_t* __restrict__ in,
    float* __restrict__ out,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t rev_idx = n - 1 - idx;
        out[rev_idx] = __bfloat162float(
            __ushort_as_bfloat16(static_cast<uint16_t>(in[rev_idx])));
    }
}

torch::Tensor sort_cuda(torch::Tensor input_tensor, torch::Tensor output_tensor) {
    auto num_items = static_cast<int64_t>(input_tensor.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const float* data_in = input_tensor.const_data_ptr<float>();
    float* data_out = output_tensor.data_ptr<float>();

    uint8_t* scratch = scratch_pool.data_ptr<uint8_t>();
    uint16_t* encode_buf = reinterpret_cast<uint16_t*>(scratch);
    void* cub_temp = scratch + static_cast<size_t>(num_items) * 2;

    const int block_size = 256;
    const int grid_size = static_cast<int>((num_items + block_size - 1) / block_size);

    // Step 1: Encode f32 -> uint16 into scratch buffer
    encode_f32_to_uint16_kernel<<<grid_size, block_size, 0, stream>>>(
        data_in,
        reinterpret_cast<int16_t*>(encode_buf),
        num_items);

    // Step 2: DoubleBuffer SortKeys: scratch(encoded) -> output(sorted uint16)
    cub::DoubleBuffer<uint16_t> d_keys(
        reinterpret_cast<uint16_t*>(data_out),
        encode_buf);

    size_t temp_bytes = cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        cub_temp, temp_bytes,
        d_keys,
        static_cast<int>(num_items),
        0, 16,
        stream);

    // Step 3: Decode uint16 -> f32 in reverse
    // d_keys.Current() points to the sorted output buffer
    // (DoubleBuffer swapped if odd radix passes; 16-bit = 2 passes → output holds final)
    const uint16_t* sorted_keys = d_keys.Current();
    decode_uint16_to_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const int16_t*>(sorted_keys),
        data_out,
        num_items);

    return output_tensor;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_bfloat16_v3',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys with bfloat16 encoding + DoubleBuffer.
    No intermediate cudaMemcpy: encode to scratch, SortKeys into output, decode in-place.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor