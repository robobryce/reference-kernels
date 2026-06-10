"""
SUBTRACTIVE TEST (c): Original baseline — torch.sort with torch.compile reduce-overhead.
This measures the full cost of torch.sort+compile vs CUB SortKeys.
"""
import torch
from task import input_t, output_t


def _custom_kernel(data: input_t) -> output_t:
    """
    Implements sort using PyTorch.
    Args:
        data: Input tensor to be sorted
    Returns:
        Sorted tensor
    """
    data_tensor, output = data
    output[...] = torch.sort(data_tensor)[0]
    return output


custom_kernel = torch.compile(_custom_kernel, mode="reduce-overhead")