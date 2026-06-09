"""
CUB DeviceRadixSort::SortPairs with DoubleBuffer API (is_overwrite_okay=true).
Uses ~N temp storage instead of ~2N. Input/output tensors are used directly
as DoubleBuffer slots to avoid extra copies.
Keys: int32 bitcast of float32. Values: uint8 dummy (minimal O(N) overhead).
Persistent buffers allocated once at module init.
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
static torch::Tensor persistent_val_buf1 = {};
static torch::Tensor persistent_val_buf2 = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;

    persistent_val_buf1 = torch::empty(
        {max_n}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    persistent_val_buf2 = torch::empty(
        {max_n}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    cub::DoubleBuffer<uint8_t> d_values(
        persistent_val_buf1.data_ptr<uint8_t>(),
        persistent_val_buf2.data_ptr<uint8_t>());

    // Use nullptr for keys (will query with real buffers at call time)
    cub::DeviceRadixSort::SortPairs(
        nullptr, persistent_temp_bytes,
        static_cast<int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<uint8_t*>(nullptr),
        static_cast<uint8_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 32);

    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Use input and output tensors as DoubleBuffer slots for keys
    // input = current (has data), output = alternate (temp/result destination)
    cub::DoubleBuffer<int32_t> d_keys(
        const_cast<int32_t*>(reinterpret_cast<const int32_t*>(input.const_data_ptr<float>())),
        reinterpret_cast<int32_t*>(output.data_ptr<float>()));

    cub::DoubleBuffer<uint8_t> d_values(
        persistent_val_buf1.data_ptr<uint8_t>(),
        persistent_val_buf2.data_ptr<uint8_t>());

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortPairs(
        persistent_temp.data_ptr(), temp_bytes,
        d_keys, d_values,
        num_items,
        0, 32,
        stream);

    // d_keys.Current() now points to the sorted result — copy if needed
    if (d_keys.Current() == reinterpret_cast<int32_t*>(output.data_ptr<float>())) {
        // Result already in output buffer — nothing to do
    } else {
        // Result still in input buffer — copy to output
        cudaMemcpyAsync(
            output.data_ptr<float>(),
            input.const_data_ptr<float>(),
            num_items * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream);
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
    name='sort_cuda_sortpairs_doublebuf_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortPairs DoubleBuffer API.
    Input/output tensors used directly as DoubleBuffer slots.
    Persistent temp + value buffers at module init.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor