"""
In-place sort via cub::DoubleBuffer — wraps output tensor's data_ptr
and a temp buffer to manage input/output swap during SortKeys.
Copy input to DoubleBuffer's current slot, sort swaps to alternate,
then copy back if needed. Persistent temp storage + DoubleBuffer.
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
static torch::Tensor persistent_dbl_buffer = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 32);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    persistent_dbl_buffer = torch::empty(
        {max_n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(input.sizes() == output.sizes(), "Input and output must have same size");

    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Copy input to output first (we sort the output buffer in-place)
    cudaMemcpyAsync(
        output.data_ptr<float>(), input.const_data_ptr<float>(),
        num_items * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // DoubleBuffer: current=output buffer, alternate=temp dbl buffer
    int32_t* dbl_buf = reinterpret_cast<int32_t*>(persistent_dbl_buffer.data_ptr());
    int32_t* out_keys = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    cub::DoubleBuffer<int32_t> d_keys(out_keys, dbl_buf);

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        d_keys, num_items,
        0, 32,
        stream);

    // If result is in dbl_buf (alternate), copy back to output
    if (d_keys.Current() != out_keys) {
        cudaMemcpyAsync(
            output.data_ptr<float>(), dbl_buf,
            num_items * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_doublebuffer',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor