"""
CUB DeviceRadixSort::SortKeys via standalone .so + ctypes.CDLL.
ZERO PyTorch build system — no load_inline, no cpp_extension, no CUDAContext.h.
The pre-compiled libsort.so exposes sort_float32(in_ptr, out_ptr, n).
All data is positive IEEE 754 float32 — raw int32 bits sort identically.
"""
import ctypes
import os
import torch

from task import input_t, output_t

# Load the pre-built shared library
_so_dir = os.path.dirname(os.path.abspath(__file__))
_libsort = ctypes.CDLL(os.path.join(_so_dir, "libsort.so"))

# Function signatures
# int sort_float32_init() -> returns 0 on success
_libsort.sort_float32_init.argtypes = []
_libsort.sort_float32_init.restype = ctypes.c_int

# int sort_float32(float* d_in, float* d_out, int n) -> returns 0 on success
_libsort.sort_float32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_libsort.sort_float32.restype = ctypes.c_int

# Allocate persistent temp storage at module import
_ret = _libsort.sort_float32_init()
if _ret != 0:
    raise RuntimeError("sort_float32_init failed: unable to allocate GPU temp storage")


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys called through ctypes.CDLL.
    No PyTorch build system — pure ctypes + pre-compiled .so.
    """
    input_tensor, output_tensor = data
    input_tensor = input_tensor.contiguous()
    n = input_tensor.numel()

    ret = _libsort.sort_float32(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_tensor.data_ptr()),
        ctypes.c_int(n),
    )
    if ret != 0:
        raise RuntimeError(f"sort_float32 failed with code {ret}")

    return output_tensor