from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING
from typing import cast

from .._compat import get_triton_find_paths_if
from .._compat import get_triton_iterable_path
from ..autotuner.logger import classify_triton_exception
from ..autotuner.logger import format_triton_compile_failure


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton.runtime.jit import JITFunction

    from .config import Config
    from .kernel import BoundKernel

from triton.runtime.driver import driver
from triton.compiler import make_backend


def make_precompiler(
    fn: JITFunction[object],
    config: Config,
    bound_kernel: BoundKernel,
) -> Callable[..., Callable[[], None]]:
    from .kernel import _find_device

    def _make_precompiler(*args: object, **kwargs: object) -> Callable[[], None]:
        """
        This is based on the Triton JITFunction.run, but breaks compile into two
        parts so we can wrap it in a subprocess to handle configs that hang in
        Triton compile and never return.
        """
        # pyrefly: ignore [bad-argument-type]
        # device = _find_device([*args, *kwargs.values()])
        kwargs["debug"] = (
            # kwargs.get("debug", fn.debug) or os.environ.get("TRITON_DEBUG", "0") == "1"
            kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"
        )
        # kernel_cache, *_, target, backend, binder = fn.device_caches[device]
        # bound_args, specialization, options = fn.binder(*args, **kwargs)
        device = driver.active.get_current_device()
        target = driver.active.get_current_target()
        backend = make_backend(target)
        if fn.binder is None:
            fn.create_binder(backend)
        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = fn.binder(*args, **kwargs)
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        # key = str(specialization) + str(options)
        # kernel = kernel_cache.get(key, None)


        kernel = fn.cache[device].get(key, None)
        if kernel is not None:
            return already_compiled  # cache hit

        options = backend.parse_options(kwargs)
        sigkeys = [x.name for x in fn.params]
        sigvals = [x[0] for x in sig_and_spec]
        signature = dict(zip(sigkeys, sigvals, strict=False))
        # find_paths_if = get_triton_find_paths_if()
        # get_iterable_path = get_triton_iterable_path()
        # constexpr_paths = cast(
        #     "list[tuple[int, ...]]",
        #     find_paths_if(sigvals, lambda _, val: val == "constexpr"),
        # )
        # constexprs = {
        #     path: get_iterable_path(list(bound_args.values()), path)
        #     for path in constexpr_paths
        # }
        bound_vals = tuple(bound_args.values())
        configs = (backend.get_attrs_descriptor(fn.params, bound_vals), )
        constant_params = configs[0].get_constants()
        constexprs = {
                p.name: v
                for (v, p) in zip(bound_vals, fn.params)
                if p.is_constexpr or (p.num in constant_params) or v is None
        }
        # attrvals = [x[1] for x in sig_and_spec]
        # attr_paths = cast(
        #     "list[tuple[int, ...]]",
        #     find_paths_if(attrvals, lambda _, x: isinstance(x, str)),
        # )
        # attrs = {
        #     k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attr_paths
        # }

        def finish_it() -> None:
            src = fn.ASTSource(fn, signature, constexprs, configs[0])
            # here we update the cache so if this is called in the parent we skip a extra compile

            try:
                fn.cache[device][key] = fn.compile(
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
