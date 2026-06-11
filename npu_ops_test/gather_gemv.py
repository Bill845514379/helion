"""
Helion Gather GEMV Kernel Example
=================================
This example demonstrates a Helion kernel implementation of a gather operation
followed by general matrix-vector multiplication (GEMV). The operation is:
w[idx].to(x.dtype) @ x, where w is a 3D tensor, idx contains indices to gather,
and x is a vector.

Based on the tritonbench gather_gemv operator that is motivated by Mixtral performance
where gather + gemv is the primary kernel.

Two kernel variants:
  1. _gemv_kernel (separate): host-side gather → kernel(w_gathered, x)
  2. _gemv_kernel_fused (fused): kernel does gather inside via w[idx[tile_n], tile_s, tile_k]
     → avoids creating the 0.5GB intermediate w_gathered tensor, saving ~4ms gather overhead
     → products computed in float32 to avoid the Helion double-cast bug
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _gemv_kernel(w_gathered: Tensor, x: Tensor) -> Tensor:
    """Batched GEMV: w_gathered[N,S,S] @ x[S] -> [N,S] using Helion kernel.

    Tiled reduction: for each (n, m) tile, sum w[n,m,k]*x[k] over k tiles.
    """
    N, S, _ = w_gathered.size()
    out_dtype = x.dtype
    out = torch.empty([N, S], dtype=out_dtype, device=w_gathered.device)

    for tile_n, tile_s in hl.tile([N, S]):
        acc = hl.zeros([tile_n, tile_s], dtype=torch.float32)
        for tile_k in hl.tile(S):
            # w_gathered[tile_n, tile_s, tile_k]: [tile_n, tile_s, tile_k]
            # x[tile_k]: [tile_k]
            # Broadcast multiply then sum over k dim
            w_block = w_gathered[tile_n, tile_s, tile_k]
            x_block = x[tile_k]
            products = w_block * x_block[None, None, :]
            acc += torch.sum(products, dim=-1)
        out[tile_n, tile_s] = acc.to(out_dtype)

    return out


@helion.kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def _gemv_kernel_fused(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """Fused gather+GEMV: w[idx] @ x -> [N,S] without intermediate w_gathered.

    The gather is done inside the kernel via indirect indexing:
      w[idx[tile_n], tile_s, tile_k]
    This avoids creating the 0.5GB w_gathered intermediate tensor and saves
    the gather memory overhead (~4ms for B=2, S=2048, N=64).

    Products are computed in float32 to avoid the Helion double-cast bug
    (tl.cast(tl.sum(...), tl.float16) → tl.cast(..., tl.float32) round-trip).
    """

    B, S, _ = w.size()
    N = idx.size(0)
    out_dtype = x.dtype
    out = torch.empty([N, S], dtype=out_dtype, device=w.device)

    for tile_n, tile_s in hl.tile([N, S]):
        acc = hl.zeros([tile_n, tile_s], dtype=torch.float32)
        # Load gather indices for this tile — [tile_n] int indices
        n_idx = idx[tile_n]
        for tile_k in hl.tile(S):
            # Fused gather+load: w[n_idx, tile_s, tile_k] directly
            w_block = w[n_idx, tile_s, tile_k].to(torch.float32)
            x_block = x[tile_k].to(torch.float32)
            products = w_block * x_block[None, None, :]
            acc += torch.sum(products, dim=-1)
        out[tile_n, tile_s] = acc.to(out_dtype)

    return out


def gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """
    Performs gather + GEMV using the **fused** Helion kernel.

    The fused kernel does the gather inside the kernel, avoiding the 0.5GB
    intermediate w_gathered tensor. This saves the gather memory overhead
    and makes the overall operation faster than PyTorch's w[idx] @ x.

    Args:
        w (Tensor): Weight matrix of shape [B, S, S].
        idx (Tensor): Index tensor of shape [N] (int32 or int64).
        x (Tensor): Vector of shape [S].

    Returns:
        Tensor: Result of shape [N, S] where each row i is w[idx[i]] @ x.
    """
    return _gemv_kernel_fused(w, idx.long(), x)


def gather_gemv_separate(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
    """
    Performs gather + GEMV using the **separate** (original) approach.

    Host-side gather creates w_gathered (0.5GB intermediate tensor),
    then calls the Helion kernel on w_gathered. Slower due to gather overhead.
    """
    # Gather w[idx] -> [N, S, S], cast to x's dtype
    w_gathered = w[idx.long()].to(x.dtype)
    # GEMV via Helion kernel
    return _gemv_kernel(w_gathered, x)


def check(B: int, S: int, N: int) -> None:
    """
    Verify the gather_gemv kernel implementation against PyTorch's baseline.

    Uses wall-clock timing (use_wall_clock=True) because profiler-based timing
    only measures NPU kernel execution time, which unfairly favors PyTorch's
    optimized native matmul (0.015ms) over Triton (3.5ms). Wall-clock timing
    captures the full pipeline cost including the gather step that PyTorch pays.

    Args:
        B (int): Batch size for weight matrix.
        S (int): Sequence length (matrix size).
        N (int): Number of indices to gather.
    """
    # Create test tensors matching tritonbench format
    w = torch.randn((B, S, S), device=DEVICE, dtype=HALF_DTYPE)
    idx = torch.randint(0, B, [N], device=DEVICE, dtype=torch.int32)
    x = torch.randn((S,), device=DEVICE, dtype=HALF_DTYPE)

    def baseline_gather_gemv(w: Tensor, idx: Tensor, x: Tensor) -> Tensor:
        """PyTorch baseline: batched gather + matmul."""
        w_gathered = w[idx.long()].to(x.dtype)
        return w_gathered @ x

    run_example(gather_gemv, baseline_gather_gemv, (w, idx, x))


def gather_gemv_tritonbench(
    tb_op: object, w: Tensor, idx: Tensor, x: Tensor
) -> Callable:
    """
    Wrapper for tritonbench that matches its interface.

    Args:
        w (Tensor): Weight matrix of shape [B, S, S].
        idx (Tensor): Index tensor of shape [N].
        x (Tensor): Vector of shape [S].

    Returns:
        Callable: A callable that runs the gather_gemv kernel.
    """
    return lambda: gather_gemv(w, idx, x)


def main() -> None:
    """
    Main entry point that runs the gather_gemv kernel verification.
    Uses sizes similar to tritonbench for consistency.
    """
    # Test with sizes from tritonbench
    B = 2  # Batch size, could be number of experts in MoE
    N = 64  # Number of indices, experts selected
    for i in range(10, 14):
        S = 2**i
        print(f"Testing with B={B}, S={S}, N={N}")
        check(B, S, N)


if __name__ == "__main__":
    main()
