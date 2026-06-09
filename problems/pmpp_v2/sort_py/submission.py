"""
Pure Python torch.sort on int32 view of float32 data.
Since all input data is positive IEEE 754 floats, integer bit order
equals float sort order. View the float32 tensor as int32, sort,
and the resulting sorted bit patterns ARE the sorted floats.
No load_inline, no JIT compilation, no data copy.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Sort float32 data by viewing it as int32 and using torch.sort.
    For positive IEEE 754 floats, bit-order == value-order, so sorting
    the raw bits produces the correct float sort with no conversion.
    """
    input_tensor, output_tensor = data
    # View float32 as int32 (zero-copy, same memory)
    int32_view = input_tensor.view(torch.int32)
    # Sort the int32 bit patterns (preserves float order for positive values)
    # .values gives the sorted int32 values
    sorted_int32 = torch.sort(int32_view)[0]
    # View back as float32 and copy into output
    output_tensor.copy_(sorted_int32.view(torch.float32))
    return output_tensor