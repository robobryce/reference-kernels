"""
GPU sort via pre-compiled shared library + ctypes.
Sorts float32 data using a pre-built sort kernel.
All positive IEEE 754 float32 values have int32 bits in sort order.
"""
import ctypes
import os
import torch

from task import input_t, output_t

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsort.so"))

_lib.sort_float32_init.argtypes = []
_lib.sort_float32_init.restype = ctypes.c_int

_lib.sort_float32.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]
_lib.sort_float32.restype = ctypes.c_int

_r = _lib.sort_float32_init()
if _r != 0:
    raise RuntimeError("GPU sort library init failed")


def custom_kernel(data: input_t) -> output_t:
    inp, out = data
    inp = inp.contiguous()
    n = inp.numel()
    s = torch.cuda.current_stream().cuda_stream
    r = _lib.sort_float32(
        ctypes.c_void_p(inp.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(n),
        ctypes.c_void_p(s),
    )
    if r != 0:
        raise RuntimeError("GPU sort failed")
    return out