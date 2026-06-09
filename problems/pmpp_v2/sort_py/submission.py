import triton
import triton.language as tl
import torch
from task import input_t, output_t


@triton.jit
def bitonic_merge_stride(
    data_ptr,
    n_elements,
    stage_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-stride bitonic merge. Controller writes both positions.
    Used for inter-block strides.
    """
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + tid
    partner = idx ^ stride
    c = idx < partner
    io = idx < n_elements
    po = partner < n_elements
    asc = (idx & stage_size) == 0
    v = tl.load(data_ptr + idx, mask=io, other=float('inf'))
    pv = tl.load(data_ptr + partner, mask=po, other=float('inf'))
    sw = tl.where(asc, v > pv, v < pv) & c & io & po
    tl.store(data_ptr + idx, tl.where(sw, pv, v), mask=io & c)
    tl.store(data_ptr + partner, tl.where(sw, v, pv), mask=po & c)


@triton.jit
def bitonic_intrablock_kernel(
    data_ptr,
    n_elements,
    stage_size,
    first_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Intra-block merge. Runs all strides from first_stride down to 1
    in a single launch, with tl.debug_barrier between strides.
    """
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)
    st = first_stride
    while st > 0:
        idx = pid * BLOCK_SIZE + tid
        partner = idx ^ st
        c = idx < partner
        io = idx < n_elements
        po = partner < n_elements
        asc = (idx & stage_size) == 0
        v = tl.load(data_ptr + idx, mask=io, other=float('inf'))
        pv = tl.load(data_ptr + partner, mask=po, other=float('inf'))
        sw = tl.where(asc, v > pv, v < pv) & c & io & po
        tl.store(data_ptr + idx, tl.where(sw, pv, v), mask=io & c)
        tl.store(data_ptr + partner, tl.where(sw, v, pv), mask=po & c)
        tl.debug_barrier()
        st //= 2


def custom_kernel(data: input_t) -> output_t:
    """
    Sort a 1D float32 array using Triton bitonic sort.

    Inter-block strides (>= BLOCK_SIZE): one launch each.
    Intra-block strides (< BLOCK_SIZE): one launch per stage with barriers.
    """
    data_tensor, output_tensor = data
    n = data_tensor.numel()

    padded_n = 1
    while padded_n < n:
        padded_n *= 2

    padded = torch.empty(padded_n, dtype=torch.float32, device=data_tensor.device)
    padded[:n] = data_tensor
    padded[n:] = float('inf')

    BLOCK_SIZE: int = 1024
    grid = (padded_n + BLOCK_SIZE - 1) // BLOCK_SIZE

    stage_size = 2
    while stage_size <= padded_n:
        stride = stage_size // 2
        while stride >= BLOCK_SIZE:
            bitonic_merge_stride[(grid,)](
                padded, padded_n, stage_size, stride,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            stride //= 2
        if stride > 0:
            bitonic_intrablock_kernel[(grid,)](
                padded, padded_n, stage_size, stride,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        stage_size *= 2

    output_tensor[:] = padded[:n]
    return output_tensor