from __future__ import annotations

import contextlib
import functools
import math
from typing import Any
from typing import Callable
from typing import Generator

from helion._compiler.compile_environment import CompileEnvironment

import torch
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import make_pointwise
from torch._inductor.lowering import to_dtype
from torch._inductor.virtualized import ops as vops
inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}


# pyrefly: ignore [implicit-import]
register_inductor_lowering = torch._inductor.lowering.register_lowering
npu_only_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}


try:
    if hasattr(torch.ops, "npu") and hasattr(torch.ops.npu, "_npu_dtype_cast"):
        _npu_dtype_cast_op = torch.ops.npu._npu_dtype_cast.default
    else:
        _npu_dtype_cast_op = None
except (AttributeError, RuntimeError):
    _npu_dtype_cast_op = None

def is_npu_backend():
    from helion._compiler.compile_environment import CompileEnvironment
    try:
        env = CompileEnvironment.current()
        return env.device.type == "npu"
    except:
        return False

def create_fp16_to_fp32_unary_fallback_lowering(
    original_op: Callable[..., object],
) -> Callable[..., object]:
    """Create a lowering that converts fp16/bfloat16 inputs to fp32 before calling the operation."""

    @functools.wraps(original_op)
    def fp32_fallback_lowering(x: object) -> object:
        if isinstance(x, TensorBox) and (original_dtype := x.get_dtype()) in (
            torch.float16,
            torch.bfloat16,
        ):
            x_fp32 = to_dtype(x, torch.float32)
            result_fp32 = original_op(x_fp32)
            assert isinstance(result_fp32, TensorBox)
            return to_dtype(result_fp32, original_dtype)
        return original_op(x)

    return fp32_fallback_lowering


# Operations that need fp32 fallbacks due to libdevice/tl_math limitations
FP32_FALLBACK_OPS_UNARY = [
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.sin.default,
    torch.ops.aten.cos.default,
    torch.ops.aten.log.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.log1p.default,
    torch.ops.aten.expm1.default,
    torch.ops.aten.exp.default,
]

# Register fp32 fallback lowerings for ops that don't support fp16/bfloat16
for op in FP32_FALLBACK_OPS_UNARY:
    inductor_lowering_dispatch[op] = create_fp16_to_fp32_unary_fallback_lowering(
        original_lowerings[op]
    )


# Handle NPU dtype cast operation by delegating to standard to_dtype
if _npu_dtype_cast_op is not None:
    @register_inductor_lowering(
        [_npu_dtype_cast_op],
        lowering_dict=inductor_lowering_dispatch,
    )
    def npu_dtype_cast(
        x: TensorBox,
        dtype: torch.dtype,
    ) -> TensorBox:
        return to_dtype(x, dtype)


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, Any, Any]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    # pyrefly: ignore [implicit-import]
    original_lowerings = torch._inductor.lowering.lowerings.copy()
    try:
        # pyrefly: ignore [implicit-import]
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)

        if is_npu_backend():
            torch._inductor.lowering.lowerings.update(npu_only_lowering_dispatch)

        yield
    finally:
        # pyrefly: ignore [implicit-import]
        torch._inductor.lowering.lowerings = original_lowerings


def var_mean_helper_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    from torch._inductor.lowering import var_mean_sum_
    from torch._prims_common import get_computation_dtype

    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)

    x = to_dtype(x, compute_dtype, copy=False)

    kwargs = {
        "x": x,
        "axis": axis,
        "correction": correction,
        "keepdim": keepdim,
        "return_mean": return_mean,
    }
    # TODO(yf225): support Welford reduction in Helion, then switch back to use Inductor `var_mean_helper_()`.
    output = var_mean_sum_(**kwargs)
    output = tuple(to_dtype(o, out_dtype, copy=False) for o in output)
    # pyrefly: ignore [bad-return]
    return output[0] if not return_mean else output


@register_inductor_lowering(
    [torch.ops.aten.var.correction],
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(
    torch.ops.aten.var_mean.correction,
    lowering_dict=inductor_lowering_dispatch,
)
def var_mean(
    # pyrefly: ignore [implicit-import]
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
    # pyrefly: ignore [implicit-import]
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )

aten = torch.ops.aten

@register_inductor_lowering(aten.exp2.default, lowering_dict=npu_only_lowering_dispatch)
def exp2_lowering(x):
    """
    Custom lowering implementation for aten.exp2 operation.
    Computes 2^x for each element in the input tensor.

    Implementation: exp2(x) = exp(x * ln(2))

    Args:
        x: Input tensor

    Returns:
        ComputedBuffer representing 2^x
    """
    log2_val = math.log(2)  # Natural logarithm of 2
    dtype = x.get_dtype()  # Get data type outside the inner function

    def exp2_fn(x):
        # Compute exp2(x) as exp(x * ln(2))
        return vops.exp(vops.mul(x, vops.constant(log2_val, dtype)))

    return make_pointwise(exp2_fn)(x)


@register_inductor_lowering(aten._log_softmax.default, lowering_dict=npu_only_lowering_dispatch)
def log_softmax_lowering(x, dim, half_to_float=False):
    """
    Numerically stable implementation of log-softmax with fp32 upcast.

    Computes: log_softmax(x) = x - log(sum(exp(x - max(x))))

    For fp16/bf16 inputs, upcasts to fp32 once at the start and computes
    all 6 decomposition steps in fp32, then casts back once. This avoids
    the 12 nested fp16↔fp32 round-trips that occur when each step
    independently triggers FP32_FALLBACK_OPS_UNARY for exp/log.
    """

    dtype = x.get_dtype()
    ndim = len(x.get_size())
    if dim < 0:
        dim = ndim + dim

    # Upcast once at the start if input is sub-fp32
    needs_upcast = dtype in (torch.float16, torch.bfloat16)
    if needs_upcast:
        x = to_dtype(x, torch.float32)

    # All 6 steps in fp32
    x_max = original_lowerings[aten.amax.default](x, axis=[dim], keepdims=True)
    shifted = original_lowerings[aten.sub.Tensor](x, x_max)
    exp_shifted = original_lowerings[aten.exp.default](shifted)
    sum_exp = original_lowerings[aten.sum.dim_IntList](
        exp_shifted, axis=[dim], keepdims=True
    )
    log_sum_exp = original_lowerings[aten.log.default](sum_exp)
    result = original_lowerings[aten.sub.Tensor](shifted, log_sum_exp)

    # Cast back once at the end
    if needs_upcast and not half_to_float:
        result = to_dtype(result, dtype)
    elif half_to_float and dtype in (torch.float16, torch.bfloat16):
        result = to_dtype(result, torch.float32)

    return result

@register_inductor_lowering(aten.log2.default, lowering_dict=npu_only_lowering_dispatch)
def log2_scalar_lowering(x):
    """
    Custom lowering implementation for aten.log2 operation with scalar input.

    Args:
        x: Scalar input

    Returns:
        ComputedBuffer representing log2(x)
    """

    def log2_fn(x):
        return vops.log2(x)

    return make_pointwise(log2_fn)(x)


@register_inductor_lowering(aten.remainder.Scalar_Tensor, lowering_dict=npu_only_lowering_dispatch)
@register_inductor_lowering(aten.remainder.Scalar, lowering_dict=npu_only_lowering_dispatch)
def remainder_scalar_lowering(x, divisor):
    """
    Custom lowering for aten.remainder.Scalar and aten.remainder.Scalar_Tensor.
    """
    if hasattr(divisor, 'get_dtype'):
        x_size = x.get_size()

        if hasattr(divisor, 'get_size'):
            d_size = divisor.get_size()
            if len(d_size) == 0:
                divisor = original_lowerings[aten.expand.default](divisor, x_size)

        def remainder_fn(x, d):
            return vops.mod(x, d)

        return make_pointwise(remainder_fn)(x, divisor)
    else:
        def remainder_fn(x):
            return vops.mod(x, divisor)

        return make_pointwise(remainder_fn)(x)

@register_inductor_lowering(aten.bitwise_or.Tensor, lowering_dict=npu_only_lowering_dispatch)
def bitwise_or_tensor_lowering(x, y):
    """
    Custom lowering implementation for aten.bitwise_or.Tensor operation.
    Computes element-wise bitwise OR of two tensors.

    Args:
        x: First input tensor
        y: Second input tensor

    Returns:
        ComputedBuffer representing x | y (element-wise bitwise OR)
    """
    def bitwise_or_fn(x, y):
        return vops.bitwise_or(x, y)

    return make_pointwise(bitwise_or_fn)(x, y)


@register_inductor_lowering(aten.__lshift__.Scalar, lowering_dict=npu_only_lowering_dispatch)
def lshift_scalar_lowering(x, shift_amount):
    """
    Custom lowering implementation for aten.__lshift__.Scalar (left bit shift).

    Computes: x << shift_amount

    Args:
        x: Input tensor (typically int8)
        shift_amount: Number of bits to shift left (scalar integer)

    Returns:
        ComputedBuffer representing x << shift_amount
    """

    def lshift_fn(x):
        return vops.lshift(x, shift_amount)

    return make_pointwise(lshift_fn)(x)


@register_inductor_lowering(aten.__rshift__.Scalar, lowering_dict=npu_only_lowering_dispatch)
def rshift_scalar_lowering(x, shift_amount):
    """
    Custom lowering implementation for aten.__rshift__.Scalar (right bit shift).

    Computes: x >> shift_amount (arithmetic right shift for signed integers)

    Args:
        x: Input tensor (typically int8)
        shift_amount: Number of bits to shift right (scalar integer)

    Returns:
        ComputedBuffer representing x >> shift_amount
    """

    def rshift_fn(x):
        return vops.rshift(x, shift_amount)

    return make_pointwise(rshift_fn)(x)