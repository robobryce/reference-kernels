"""
Pure PyTorch torch.sort on int32 bitcast — NO CUDAGraph, baseline measurement.
Used to quantify CUDAGraph benefit.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    in_contig = input_tensor.contiguous()
    out_contig = output_tensor.contiguous()
    n = in_contig.numel()

    input_int = in_contig.view(torch.int32)
    output_int = out_contig.view(torch.int32)
    idx = torch.empty(n, dtype=torch.int64, device=in_contig.device)

    torch.sort(input_int, out=(output_int, idx))
    return output_tensor