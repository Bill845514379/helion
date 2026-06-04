"""
Element-wise Addition Example
=============================

This example demonstrates how to implement an element-wise addition kernel using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch
import torch_npu

# Match matmul NPU setup: avoid internal-format tensors that can disagree with
# Helion/Triton pointer loads and fault the Ascend vector core.
torch_npu.npu.config.allow_internal_format = True

import helion
# from helion._testing import DEVICE
DEVICE = "npu"
from helion._testing import run_example
import helion.language as hl

# %%
# Addition Kernel
# ---------------


# %%
@helion.kernel(
    static_shapes=True,
    autotune_ignore_errors=True,
    autotune_effort="full",
  )
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors element-wise with broadcasting support.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        A new tensor containing the element-wise sum of x and y
    """
    # Assumes inputs are already broadcastable (same shape for common case)
    out = torch.empty_like(x)
    # Flatten to 1D for NPU: avoids 2D PID decomposition overhead,
    # double masking, and strided pointer arithmetic on Ascend.
    total = out.numel()
    out_flat = out.reshape(total)
    x_flat = x.reshape(total)
    y_flat = y.reshape(total)
    # Let Helion infer block_size from tile
    block_size = hl.register_block_size(1024, min(x.numel(), 8192))
    for tile in hl.tile(total, block_size=block_size):
        out_flat[tile] = x_flat[tile] + y_flat[tile]
    return out


# %%
# Verification Function
# ---------------------


# %%
def check(m: int, n: int) -> None:
    """
    Verify the add kernel implementation against PyTorch's native add function.

    Args:
        m: First dimension of the test tensors
        n: Second dimension of the test tensors
    """
    x = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn([m, n], device=DEVICE, dtype=torch.bfloat16)
    x, y = torch.broadcast_tensors(x, y)
    run_example(add, torch.add, (x, y))

# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the add kernel verification with 1024x1024 tensors.
    """
    check(8192, 8192)


if __name__ == "__main__":
    import time
    time0 = time.time()
    main()
    print(f"time cost: {time.time()-time0}")