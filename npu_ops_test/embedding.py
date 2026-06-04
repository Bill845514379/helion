from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
# Embedding Kernel
# ----------------

@helion.kernel(
    static_shapes=True,
)
def embedding(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs embedding lookup for input indices.
    """
    b, seq = x.size()
    vocab, embedding_dim = weight.size()
    out = torch.empty([b, seq, embedding_dim], dtype=weight.dtype, device=weight.device)

    for tile_b, tile_s, tile_e in hl.tile([b, seq, embedding_dim]):
        out[tile_b, tile_s, tile_e] = weight[x[tile_b, tile_s], tile_e]

    return out


# %%
# Benchmark Wrapper
# -----------------

def embedding_tritonbench(
        tb_op: object, V: int, D: int, inp: torch.Tensor, shared_weight: torch.Tensor
) -> Callable[[], torch.Tensor]:
    return lambda: embedding(inp, shared_weight)


# %%
# Main Function
# -------------

def main() -> None:
    num_embeddings, embedding_dim = 128, 128
    x = torch.randint(0, num_embeddings, [128, 8], device=DEVICE, dtype=torch.int32)

    weight = torch.randn([num_embeddings, embedding_dim], device=DEVICE, dtype=torch.float16)

    run_example(
        embedding, torch.nn.functional.embedding, (x, weight), atol=1e-3, rtol=1e-3
    )


if __name__ == "__main__":
    main()