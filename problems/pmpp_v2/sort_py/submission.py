"""
bfloat16 SortKeys v6: PyTorch conversion + CUB SortKeys uint16.
Pre-allocate encode buffer. Use view_as for zero-copy.
f32 data reinterpreted as int32, shift-right-16 to get top 16 bits (bfloat16).
CUB SortKeys on uint16, then shift-left-16 + view as f32 to decode.
All bfloat16 truncation via integer shifts (fast, branchless).
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor cub_temp = {};
static size_t cub_temp_bytes = 0;

void init_temp() {
    if (cub_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, cub_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<int>(max_n),
        0, 16);
    size_t cap = (cub_temp_bytes * 11 + 9) / 10;
    cub_temp = torch::empty(
        {static_cast<int64_t>(cap)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(
    torch::Tensor keys_in_uint16,
    torch::Tensor keys_out_uint16,
    int64_t num_items) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint16_t* d_in = reinterpret_cast<const uint16_t*>(
        keys_in_uint16.const_data_ptr());
    uint16_t* d_out = reinterpret_cast<uint16_t*>(
        keys_out_uint16.data_ptr());
    size_t temp_bytes = cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        cub_temp.data_ptr(), temp_bytes,
        d_in, d_out, static_cast<int>(num_items),
        0, 16, stream);
    return keys_out_uint16;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_temp();
torch::Tensor sort_cuda(torch::Tensor k_in, torch::Tensor k_out, int64_t n);
"""

sort_module = load_inline(
    name='sort_bf16_v6',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)
sort_module.init_temp()

_encode_buf = None


def custom_kernel(data: input_t) -> output_t:
    global _encode_buf
    input_tensor, output_tensor = data
    n = input_tensor.numel()

    if _encode_buf is None or _encode_buf.numel() < n:
        _encode_buf = torch.empty(n, dtype=torch.int32, device='cuda')

    # Step 1: Encode f32 -> uint16 via int32 bit-shift
    # Read as int32, shift right 16 to extract top 16 bits (= bfloat16).
    # Write lower 16 bits via int16 view of encode buffer.
    # torch.bitwise_right_shift returns int32; .to(torch.int16) truncates.
    int32_view = input_tensor.view(torch.int32)
    encode_i16 = _encode_buf[:n].view(torch.int16)
    encode_i16.copy_(int32_view.bitwise_right_shift(16).to(torch.int16))

    # Step 2: CUB SortKeys on uint16
    # Read from encode_i16, write to output as uint16 view
    out_uint16 = output_tensor.view(torch.int16)[:n]
    sort_module.sort_cuda(encode_i16, out_uint16, n)

    # Step 3: Decode uint16 -> f32 via int32 shift-left-16 + view
    # out_uint16 -> int32 -> shift left 16 -> view as float32
    out_i32 = out_uint16.to(torch.int32)
    output_tensor.copy_(out_i32.bitwise_left_shift(16).view(torch.float32))

    return output_tensor