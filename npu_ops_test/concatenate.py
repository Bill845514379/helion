"""
Tensor Concatenation Example
============================

This example demonstrates how to implement a tensor concatenation operation using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch
import torch_npu

# Match other NPU ops tests: internal-format tensors can disagree with Helion/Triton
# pointer loads and fault the Ascend vector core.
torch_npu.npu.config.allow_internal_format = True

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Concatenation Kernel
# --------------------


# %%
@helion.kernel(
    config=helion.Config(
        block_sizes=[32, 1024, 512],
        pid_type="flat",
    ),
    static_shapes=True,
)
def concat2d_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Concatenates two 2D tensors along dimension 1 (columns).

    Args:
        x: First input tensor of shape [M, N1]
        y: Second input tensor of shape [M, N2] with same first dimension as x

    Returns:
        Output tensor of shape [M, N1+N2] containing the concatenation of x and y along dimension 1
    """
    assert x.size(0) == y.size(0)
    m = x.size(0)
    n1 = x.size(1)
    n2 = y.size(1)
    out = torch.empty(
        [m, n1 + n2], dtype=x.dtype, device=x.device
    )
    for tile0 in hl.tile(m):
        for tile1 in hl.tile(n1):
            out[tile0, tile1] = x[tile0, tile1]
        for tile1 in hl.tile(n2):
            out[tile0, tile1 + n1] = y[tile0, tile1]
    return out


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the concatenation kernel verification.
    Tests with two tensors of shapes [1500, 400] and [1500, 600].
    """
    x = torch.randn([1500, 400], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([1500, 600], device=DEVICE, dtype=HALF_DTYPE)
    run_example(concat2d_dim1, lambda x, y: torch.cat([x, y], dim=1), (x, y))


if __name__ == "__main__":
    main()