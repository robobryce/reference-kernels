import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

// Persistent temp storage: allocated once, reused across all sort_cuda calls.
// Sorted output is written to the user-supplied output tensor, not the temp
// buffer, so the temp storage is safe to reuse without per-call allocation.
static void* g_temp_storage = nullptr;
static size_t g_temp_storage_bytes = 0;

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(input.sizes() == output.sizes(), "Input and output must have same size");

    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Step 1: query temp storage size
    size_t required_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, required_bytes,
        static_cast<const float*>(input.const_data_ptr<float>()),
        static_cast<float*>(output.data_ptr<float>()),
        num_items,
        0, sizeof(float) * 8,
        stream);

    // Step 2: allocate or grow persistent temp storage (first call or larger input)
    if (required_bytes > g_temp_storage_bytes) {
        if (g_temp_storage) {
            cudaFree(g_temp_storage);
        }
        cudaMalloc(&g_temp_storage, required_bytes);
        g_temp_storage_bytes = required_bytes;
    }

    // Step 3: run the sort using persistent temp storage
    cub::DeviceRadixSort::SortKeys(
        g_temp_storage,
        required_bytes,
        static_cast<const float*>(input.const_data_ptr<float>()),
        static_cast<float*>(output.data_ptr<float>()),
        num_items,
        0, sizeof(float) * 8,
        stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_onesweep',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Sort using direct CUB DeviceRadixSort::SortKeys (keys-only, no values payload).
    """
    input_tensor, output_tensor = data
    output_tensor[...] = sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor