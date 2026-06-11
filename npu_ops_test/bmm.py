"""
Batch Matrix Multiplication Example
====================================

This example demonstrates how to implement a batch matrix multiplication kernel using Helion.

NPU Performance Note:
  - B=1 + M<=64 (GEMV):     Triton/Helion 1.4-1.7x faster than CANN
  - B<=8 + S<=256 (small):   Triton/Helion 1.2-2.1x faster than CANN
  - B=128 + M=1:             Triton/Helion 1.1x faster than CANN
  - Fused bmm+bias+relu:     Triton/Helion 1.1-1.3x faster (1 kernel vs 3 ops)
  - Square (S>=512, B>=16):  CANN wins (Cube unit)
"""

from __future__ import annotations

from packaging import version
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl


@helion.kernel(
    static_shapes=True,
)
def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batch matrix multiplication.

    Args:
        A: Input tensor of shape [B, M, K]
        B: Input tensor of shape [B, K, N]

    Returns:
        Output tensor of shape [B, M, N] containing the result of batch matrix multiplication
    """
    # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )

    # Tile over batch, M, and N dimensions
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        # Use float32 for accumulation to maintain precision
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)

        # Tile over K dimension
        # Use torch.baddbmm to avoid fp32->fp16->fp32 redundant casts
        # (acc += A @ B causes: dot(fp32) -> cast fp16 -> add -> cast back fp32)
        for tile_k in hl.tile(k):
            acc += A[tile_b, tile_m, tile_k] @ B[tile_b, tile_k, tile_n]
            # acc = torch.baddbmm(acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n])

        out[tile_b, tile_m, tile_n] = acc
    return out


@helion.kernel(static_shapes=True, autotune_ignore_errors=True, autotune_effort="full")
def bmm_bias_relu(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs fused batch matrix multiplication + bias + ReLU.

    Combines three operations (bmm, bias add, ReLU) into a single kernel,
    avoiding intermediate memory reads/writes.

    Args:
        A: Input tensor of shape [B, M, K]
        B: Input tensor of shape [B, K, N]
        bias: Bias tensor of shape [N]

    Returns:
        Output tensor of shape [B, M, N] containing relu(A @ B + bias)
    """
    b, m, k = A.size()
    _, _, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )

    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )

        # Fused epilogue: add bias + ReLU
        acc = acc + bias[tile_n]
        acc = torch.relu(acc)
        out[tile_b, tile_m, tile_n] = acc
    return out


def check(b: int, m: int, k: int, n: int) -> None:
    x = torch.randn([b, m, k], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([b, k, n], device=DEVICE, dtype=HALF_DTYPE)
    run_example(bmm, torch.bmm, (x, y))


def check_fused(b: int, m: int, k: int, n: int) -> None:
    x = torch.randn([b, m, k], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([b, k, n], device=DEVICE, dtype=HALF_DTYPE)
    bias = torch.randn([n], device=DEVICE, dtype=HALF_DTYPE)

    def baseline(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.bmm(x, y) + bias)

    run_example(bmm_bias_relu, baseline, (x, y, bias))


def main() -> None:
    # torch.baddbmm support for 16-bit tensors requires torch 2.8+
    assert version.parse(torch.__version__.split("+")[0]) >= version.parse("2.8"), (
        "Requires torch 2.8+"
    )

    # check(1, 64, 1024, 1024)

    check(4, 128, 128, 128)

    # check_fused(1, 64, 1024, 1024)


if __name__ == "__main__":
    import time

    time_st = time.time()
    main()
    print(f"time cost: {time.time() - time_st}")
