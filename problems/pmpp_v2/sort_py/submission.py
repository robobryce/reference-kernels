"""
CUB DeviceRadixSort::SortKeys via standalone sort.cu built with cpp_extension.load.
Sorts int32 bitcast of positive float32 using CUB SortKeys with stream=0.
No CUDAContext.h, no load_inline — only torch/extension.h + cub headers.
"""
import os
import torch
from torch.utils.cpp_extension import load
from task import input_t, output_t

_sort_dir = os.path.dirname(os.path.abspath(__file__))
_sort_src = os.path.join(_sort_dir, "sort.cu")

sort_module = load(
    name='sort_cuda_stream0',
    sources=[_sort_src],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    Stream=0 literal — no cudaStream_t API, no CUDAContext.h dependency.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor