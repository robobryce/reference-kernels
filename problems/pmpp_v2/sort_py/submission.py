import triton
import triton.language as tl
import torch
from task import input_t, output_t


@triton.jit
def bitonic_sort_kernel(
    data_ptr,
    n_elements,       # padded_n — inf sentinels must participate
    stage_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Bitonic merge kernel. Thread 'idx' compares element at 'idx' with
    'idx ^ stride'. Controller thread (idx < partner) performs both
    writes; non-controller threads do nothing.
    """
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)

    idx = pid * BLOCK_SIZE + tid
    partner = idx ^ stride

    is_controller = idx < partner
    in_bounds = idx < n_elements
    partner_in_bounds = partner < n_elements

    # Bitonic direction: ascending when the stage-size bit is 0
    ascending = (idx & stage_size) == 0

    # Load both elements
    val = tl.load(data_ptr + idx, mask=in_bounds)
    partner_val = tl.load(data_ptr + partner, mask=partner_in_bounds)

    # Swap condition — inf participates for correct non-power-of-2 sort
    swap_needed = tl.where(ascending, val > partner_val, val < partner_val)
    do_swap = swap_needed & is_controller & in_bounds & partner_in_bounds

    # Controller writes to both positions
    tl.store(
        data_ptr + idx,
        tl.where(do_swap, partner_val, val),
        mask=in_bounds & is_controller,
    )
    tl.store(
        data_ptr + partner,
        tl.where(do_swap, val, partner_val),
        mask=partner_in_bounds & is_controller,
    )


def custom_kernel(data: input_t) -> output_t:
    """
    Sort a 1D float32 array using a Triton bitonic sort kernel.
    """
    data_tensor, output_tensor = data
    n = data_tensor.numel()

    # Pad to next power of 2 for the bitonic network structure.
    padded_n = 1
    while padded_n < n:
        padded_n *= 2

    # Allocate a padded buffer for the sort — the output tensor has
    # exactly n elements, so inf sentinels need their own space.
    padded = torch.empty(padded_n, dtype=torch.float32, device=data_tensor.device)
    padded[:n] = data_tensor
    padded[n:] = float('inf')

    BLOCK_SIZE: int = 1024
    grid_size = (padded_n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Full bitonic sort: stages k = 2, 4, 8, ..., padded_n
    stage_size = 2
    while stage_size <= padded_n:
        stride = stage_size // 2
        while stride > 0:
            bitonic_sort_kernel[(grid_size,)](
                padded, padded_n, stage_size, stride,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            stride //= 2
        stage_size *= 2

    # Copy sorted result back to output (excluding inf padding)
    output_tensor[:] = padded[:n]
    return output_tensor