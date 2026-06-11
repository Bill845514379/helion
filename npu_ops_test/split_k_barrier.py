"""
Helion Split-K Matmul - NPU Adapted Version
============================================
Two-stage split-K matrix multiplication using host-side synchronization
instead of hl.barrier() which is not supported on NPU (Ascend).

Stage 1: Compute partial products using atomic_add
Host Barrier: torch.npu.synchronize() ensures all partials are written
Stage 2: Reduce partials across the split dimension
"""

from __future__ import annotations

import torch
import torch_npu

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

torch_npu.npu.config.allow_internal_format = True


@helion.kernel(static_shapes=True, autotune_ignore_errors=True, autotune_effort="full")
def split_k_matmul_stage1(a: torch.Tensor, b: torch.Tensor, tmp: torch.Tensor) -> None:
    """
    Stage 1: Compute partial products and accumulate into tmp tensor.

    Uses atomic_add to handle parallel accumulation from multiple tiles.

    Args:
        a: Input matrix [M, K]
        b: Input matrix [K, N]
        tmp: Partial results tensor [M, N, split_k]
    """
    m, k = a.shape
    _, n = b.shape
    split_k = tmp.shape[2]
    k_block = helion.next_power_of_2(helion.cdiv(k, split_k))

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(k, block_size=k_block):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            split_idx = tile_k.begin // k_block
            hl.atomic_add(tmp, [tile_m, tile_n, split_idx], acc)
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)


@helion.kernel(static_shapes=True, autotune_ignore_errors=True, autotune_effort="full")
def split_k_reduce(tmp: torch.Tensor, out: torch.Tensor) -> None:
    """
    Stage 2: Reduce partial results across split dimension.

    Args:
        tmp: Partial results tensor [M, N, split_k]
        out: Output tensor [M, N]
    """
    m, n, split_k = tmp.shape

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)


def split_k_matmul(a: torch.Tensor, b: torch.Tensor, split_k: int = 16) -> torch.Tensor:
    """
    Two-stage split-K matmul using host-side synchronization.

    Args:
        a: Input matrix [M, K]
        b: Input matrix [K, N]
        split_k: Number of K dimension splits

    Returns:
        Output matrix [M, N]
    """
    m, k = a.shape
    _, n = b.shape

    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=torch.float32)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    split_k_matmul_stage1(a, b, tmp)

    torch.npu.synchronize()

    split_k_reduce(tmp, out)

    return out


def check(m: int, k: int, n: int) -> None:
    """Check correctness against PyTorch matmul."""
    a = torch.randn(m, k, device=DEVICE, dtype=torch.float32)
    b = torch.randn(k, n, device=DEVICE, dtype=torch.float32)

    run_example(
        lambda a, b: split_k_matmul(a, b, split_k=16),
        torch.matmul,
        args=(a, b),
        atol=5e-1,  # Tolerance for split-K accumulation errors
    )


def main() -> None:
    """Run tests."""
    torch.manual_seed(0)

    print("Testing split_k_matmul...")
    check(16, 4096, 16)


if __name__ == "__main__":
    import time

    time_st = time.time()
    main()
    print(f"time cost: {time.time() - time_st}")
