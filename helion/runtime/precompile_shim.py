from __future__ import annotations

import inspect
import os
import sys
from typing import TYPE_CHECKING

from ..autotuner.logger import classify_triton_exception
from ..autotuner.logger import format_triton_compile_failure


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton.runtime.jit import JITFunction

    from .config import Config
    from .kernel import BoundKernel

from triton.compiler import make_backend
from triton.runtime.driver import driver

try:
    from triton.runtime.jit import compute_cache_key as _triton_compute_cache_key
except ImportError:
    _triton_compute_cache_key = None


def _cache_key_modern(
    kernel_key_cache: object, specialization: object, options: object
) -> str:
    """Match Triton JITFunction.run cache keying when compute_cache_key is available."""
    if _triton_compute_cache_key is not None:
        return _triton_compute_cache_key(kernel_key_cache, specialization, options)
    key_pair = (tuple(specialization), str(options))  # type: ignore[arg-type]
    cached = kernel_key_cache.get(key_pair)  # type: ignore[union-attr]
    if cached is not None:
        return cached
    ck = str(specialization) + str(options)
    kernel_key_cache[key_pair] = ck  # type: ignore[index]
    return ck


def _ensure_jit_binder(fn: object, backend: object) -> None:
    """Triton JITFunction.create_binder is (self) on some builds and (self, backend) on others."""
    if getattr(fn, "binder", None) is not None:
        return
    create = fn.create_binder
    if len(inspect.signature(create).parameters) == 0:
        create()
    else:
        create(backend)


def make_precompiler(
    fn: JITFunction[object],
    config: Config,
    bound_kernel: BoundKernel,
) -> Callable[..., Callable[[], None]]:
    def _make_precompiler(*args: object, **kwargs: object) -> Callable[[], None]:
        """
        This is based on the Triton JITFunction.run, but breaks compile into two
        parts so we can wrap it in a subprocess to handle configs that hang in
        Triton compile and never return.
        """
        # pyrefly: ignore [bad-argument-type]
        # device = _find_device([*args, *kwargs.values()])
        kwargs["debug"] = (
            kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"
        )
        device = driver.active.get_current_device()

        # Modern Triton: create_binder is a defaultdict factory; binder lives in
        # device_caches[device][-1], not on fn.binder. Legacy: fn.binder + fn.cache[device].
        kernel_key_cache: object = {}
        if hasattr(fn, "device_caches"):
            cache_tuple = fn.device_caches[device]
            kernel_cache = cache_tuple[0]
            if len(cache_tuple) >= 5:
                kernel_key_cache = cache_tuple[1]
            target, backend, binder = (
                cache_tuple[-3],
                cache_tuple[-2],
                cache_tuple[-1],
            )
        else:
            target = driver.active.get_current_target()
            backend = make_backend(target)
            _ensure_jit_binder(fn, backend)
            binder = fn.binder  # type: ignore[attr-defined]
            kernel_cache = fn.cache[device]  # type: ignore[attr-defined]

        bound = binder(*args, **kwargs)
        if len(bound) == 3:
            bound_args, specialization, opt_for_key = bound
            key = _cache_key_modern(kernel_key_cache, specialization, opt_for_key)
            if kernel_cache.get(key, None) is not None:
                return already_compiled

            pack = getattr(fn, "_pack_args", None)
            if pack is None:
                raise TypeError(
                    "Triton JITFunction missing _pack_args; cannot precompile with "
                    "3-value binder result"
                )
            options, signature, constexprs, attrs = pack(
                backend, kwargs, bound_args, specialization, opt_for_key
            )

            def finish_it() -> None:
                src = fn.ASTSource(fn, signature, constexprs, attrs)
                try:
                    kernel_cache[key] = fn.compile(
                        src, target=target, options=options.__dict__
                    )
                except Exception as e:
                    action = classify_triton_exception(e)
                    if action != "debug":
                        print(
                            format_triton_compile_failure(config, e, bound_kernel),
                            file=sys.stderr,
                        )
                    sys.exit(1)

            return finish_it

        if len(bound) != 5:
            raise TypeError(
                f"Unexpected Triton binder return arity {len(bound)} (expected 3 or 5)"
            )

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = (
            bound
        )
        key = "".join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        if kernel_cache.get(key, None) is not None:
            return already_compiled

        options = backend.parse_options(kwargs)
        sigkeys = [x.name for x in fn.params]
        sigvals = [x[0] for x in sig_and_spec]
        signature = dict(zip(sigkeys, sigvals, strict=False))
        bound_vals = tuple(bound_args.values())
        configs = (backend.get_attrs_descriptor(fn.params, bound_vals),)
        constant_params = configs[0].get_constants()
        constexprs = {
            p.name: v
            for (v, p) in zip(bound_vals, fn.params)
            if p.is_constexpr or (p.num in constant_params) or v is None
        }

        def finish_it() -> None:
            src = fn.ASTSource(fn, signature, constexprs, configs[0])
            try:
                kernel_cache[key] = fn.compile(
                    src, target=target, options=options.__dict__
                )
            except Exception as e:
                action = classify_triton_exception(e)
                if action != "debug":
                    print(
                        format_triton_compile_failure(config, e, bound_kernel),
                        file=sys.stderr,
                    )
                sys.exit(1)

        return finish_it

    return _make_precompiler


def already_compiled() -> None:
    return None
