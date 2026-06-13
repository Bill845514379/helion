from __future__ import annotations

import abc
import collections
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import contextlib
import dataclasses
import datetime
import functools
import inspect
from itertools import count
import logging
import math
from math import inf
import multiprocessing as mp
from multiprocessing import connection
import os
from pathlib import Path
import pickle
import pprint
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import NamedTuple
from typing import NoReturn
from typing import Protocol
from typing import cast
from unittest.mock import patch
import uuid

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map
from torch.utils._pytree import tree_map_only
from torch.utils._pytree import tree_unflatten

from .. import exc
from .._compat import extract_device
from .._compat import get_device_name
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import make_precompiler
from .benchmarking import _bench_device_synchronize
from .benchmarking import default_do_bench
from .benchmarking import default_interleaved_bench
from .benchmarking import do_bench_generic
from .benchmarking import sync_object
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import AutotuneLogEntry
from .logger import AutotuningLogger
from .logger import _get_failure_dump_dir
from .logger import capture_output
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import match_unrecoverable_runtime_error
from .logger import maybe_dump_triton_failure
from .metrics import AutotuneMetrics
from .metrics import _run_post_autotune_hooks
from .progress_bar import iter_with_progress

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from . import ConfigSpec
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig
    from .local_cache import SavedBestConfig


class _HasDevice(Protocol):
    device: torch.device


class _AutotunableKernel(Protocol):
    @property
    def config_spec(self) -> ConfigSpec: ...

    @property
    def settings(self) -> Settings: ...

    @property  # pyrefly: ignore[bad-return]
    def env(self) -> _HasDevice: ...

    @property
    def configs(self) -> Sequence[Config]: ...

    def compile_config(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        allow_print: bool = True,
    ) -> Callable[..., object]: ...

    def format_kernel_decorator(self, config: Config, settings: Settings) -> str: ...

    def get_cached_path(self, config: Config | None = None) -> str | None: ...

    def to_triton_code(
        self,
        config: Config | dict[str, object] | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str | None: ...

    def maybe_log_repro(
        self,
        log_func: Callable[[str], None],
        args: Sequence[object],
        config: Config | None = None,
    ) -> None: ...


def _kernel_autotune_device_type(kernel: object) -> str | None:
    """Return ``kernel.env.device.type`` when *kernel* is a BoundKernel-like object."""
    env = getattr(kernel, "env", None)
    if env is None:
        return None
    device = getattr(env, "device", None)
    if device is None:
        return None
    return getattr(device, "type", None)


_NPU_AUTOTUNE_LAST_CONFIG_PATH = os.path.join(
    tempfile.gettempdir(), "helion_last_autotune_config.txt"
)


def _autotune_stderr_debug_enabled(kernel: object) -> bool:
    """True when :attr:`~helion.runtime.settings.Settings.autotune_debug_stderr` is set."""
    st = getattr(kernel, "settings", None)
    return bool(getattr(st, "autotune_debug_stderr", False))


def _npu_trace_autotune_candidate(
    kernel: object, config: object, phase: str, *, persist: bool = False
) -> None:
    """On NPU, optionally log which autotune *config* is active.

    Stderr lines are emitted only when ``Settings.autotune_debug_stderr`` is true
    (``HELION_AUTOTUNE_DEBUG_STDERR=1`` or ``@helion.kernel(autotune_debug_stderr=True)``).

    ``parallel_benchmark`` compiles a batch of configs before timing them; writing
    the same file during that compile loop would leave the **last in the batch**,
    not the config currently executing.  Therefore we only ``persist`` (overwrite
    ``helion_last_autotune_config.txt`` with ``fsync``) for ``pre_device_launch``,
    immediately before the kernel is run—so after a hard crash that file names the
    candidate that was actually in flight.
    """
    if _kernel_autotune_device_type(kernel) != "npu":
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    config_repr: str
    env = getattr(kernel, "env", None)
    spec = getattr(env, "config_spec", None) if env is not None else None
    if spec is not None:
        from ..runtime.config import Config as HelionConfig

        if isinstance(config, HelionConfig):
            dbg = dict(config.config)
            spec.coerce_npu_tl_range_tunables(dbg)
            config_repr = repr(HelionConfig(**dbg))
        else:
            config_repr = repr(config)
    else:
        config_repr = repr(config)
    line = f"[helion autotune] {phase} {ts} config={config_repr}\n"
    if _autotune_stderr_debug_enabled(kernel):
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except OSError:
            pass
    if not persist:
        return
    try:
        with open(_NPU_AUTOTUNE_LAST_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


_CODE_OBJECT_RE = re.compile(r"<code object .+?, line \d+>")


class _CodeSentinel:
    """Stable stand-in for types.CodeType so spec key comparison is repr-independent."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<code>"


_CODE_SENTINEL = _CodeSentinel()


def _normalize_spec_key(key: object) -> object:
    """Replace types.CodeType with a stable sentinel in a spec key tree."""
    return tree_map_only(types.CodeType, lambda _: _CODE_SENTINEL, key)


def _normalize_spec_key_str(s: str) -> str:
    """Normalize a specialization_key string for cache comparison.

    Replaces code object repr strings with a stable '<code>' sentinel,
    allowing FROM_BEST_AVAILABLE to match function arguments based
    on their closure values only, ignoring code object identity.
    """
    return _CODE_OBJECT_RE.sub("<code>", s)


class BaseAutotuner(abc.ABC):
    """
    Abstract base class for all autotuners and classes that wrap autotuners, like caching.
    """

    @abc.abstractmethod
    def autotune(self, *, skip_cache: bool = False) -> Config:
        raise NotImplementedError


class BenchmarkResult(NamedTuple):
    """Result tuple returned by parallel_benchmark."""

    config: Config
    fn: Callable[..., object]
    perf: float
    status: Literal["ok", "error", "timeout"]
    compile_time: float | None


_FP8_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


def _assert_close(actual: object, expected: object, atol: float, rtol: float) -> None:
    """Like torch.testing.assert_close but handles fp8 and uses chunked comparison for large tensors."""

    def convert(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.uint8) if t.dtype in _FP8_DTYPES else t

    actual_flat, actual_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, actual)
    )
    expected_flat, expected_spec = tree_flatten(
        tree_map_only(torch.Tensor, convert, expected)
    )

    if actual_spec != expected_spec:
        raise AssertionError(
            f"Output tree structure mismatch during autotuner accuracy check:\n"
            f"  actual:   {actual_spec} ({len(actual_flat)} leaves)\n"
            f"  expected: {expected_spec} ({len(expected_flat)} leaves)"
        )

    for a, e in zip(actual_flat, expected_flat, strict=True):
        if isinstance(a, torch.Tensor):
            _chunked_assert_close(a, e, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol)


def _chunked_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_size: int = 2**22,  # ~4M elements per chunk
) -> None:
    """Memory-efficient assert_close for large tensors.

    Processes the comparison in chunks to avoid allocating multiple
    full-size temporary tensors.  Uses torch.testing.assert_close on
    each chunk so error messages retain full detail.
    """
    if actual.numel() <= chunk_size:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    a_flat = actual.reshape(-1)
    e_flat = expected.reshape(-1)
    for i in range(0, a_flat.numel(), chunk_size):
        a_chunk = a_flat[i : i + chunk_size]
        e_chunk = e_flat[i : i + chunk_size]
        torch.testing.assert_close(a_chunk, e_chunk, atol=atol, rtol=rtol)


def _autotune_outputs_match_baseline(
    output: object,
    args: Sequence[object],
    *,
    baseline_output: object,
    baseline_post_args: Sequence[object] | None,
    mutated_arg_indices: Sequence[int],
    atol: float,
    rtol: float,
) -> None:
    """Raise ``AssertionError`` if outputs or mutated args disagree with the baseline."""
    _assert_close(output, baseline_output, atol=atol, rtol=rtol)
    if len(mutated_arg_indices) > 0 and baseline_post_args is not None:
        _assert_close(args, baseline_post_args, atol=atol, rtol=rtol)


def _autotune_parallel_benchmark_workers_allowed() -> bool:
    """Whether explore-phase benchmark subprocesses are safe to use in this process.

    ``ProcessPoolExecutor`` + ``spawn`` under pytest (or Meta unit tests) often deadlocks
    or hangs with CUDA/NPU runtimes. Parallel benchmarking is disabled there unless
    ``HELION_AUTOTUNE_BENCHMARK_SUBPROCESS_IN_TEST=1`` is set.
    """
    v = (
        os.environ.get("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS_IN_TEST", "")
        .strip()
        .lower()
    )
    if v in ("1", "true", "yes", "on"):
        return True
    if torch._utils_internal.is_fb_unit_test():
        return False
    return not os.environ.get("PYTEST_CURRENT_TEST")


def _clone_args(
    args: Sequence[object],
    idx_to_clone: Sequence[int] | None = None,
) -> Sequence[object]:
    """
    Clone the given arguments, but cloning only the tensors specified by
      idx_to_clone. If idx_to_clone is None, clone all tensors.
    """

    args_flat, tree_spec = tree_flatten(args)
    tensor_idx = 0
    for i, arg in enumerate(args_flat):
        if not isinstance(arg, torch.Tensor):
            continue
        if idx_to_clone is None or tensor_idx in idx_to_clone:
            clone = arg.detach().clone()
            clone.requires_grad_(arg.requires_grad)
            args_flat[i] = clone
        tensor_idx += 1

    return tree_unflatten(args_flat, tree_spec)


class BaseSearch(BaseAutotuner):
    """
    Base class for search algorithms. This class defines the interface and utilities for all
    search algorithms.

    Attributes:
        kernel: The kernel to be tuned (any ``_AutotunableKernel``).
        settings: The settings associated with the kernel.
        config_spec: The configuration specification for the kernel.
        args: The arguments to be passed to the kernel.
        counters: A counter to track various metrics during the search.
    """

    _baseline_output: object
    _mutated_arg_indices: Sequence[int] = []
    _baseline_post_args: Sequence[object] | None
    _jobs: int
    _precompile_result_counter: count[int]
    _effective_atol: float
    _effective_rtol: float

    def __init__(self, kernel: _AutotunableKernel, args: Sequence[object]) -> None:
        """
        Initialize the BaseSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__()
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.config_spec: ConfigSpec = kernel.config_spec
        self.args: Sequence[object] = args
        self.log = AutotuningLogger(self.settings)
        self.best_perf_so_far = inf
        self._prepared = False
        self._precompile_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._precompile_args_path: str | None = None
        self._precompile_result_counter = count()

    def _prepare(self) -> None:
        """Some initialization deferred until autotuning actually runs.

        This is called at the start of autotune() so that cache hits skip it.
        """
        if self._prepared:
            return
        self._prepared = True
        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"Autotune random seed: {seed}")
        self._autotune_metrics: AutotuneMetrics = AutotuneMetrics(
            kernel_name=getattr(getattr(self.kernel, "kernel", None), "name", ""),
            input_shapes=str(
                [tuple(arg.shape) for arg in self.args if isinstance(arg, torch.Tensor)]
            ),
            hardware=get_device_name(extract_device(self.args)) or "",
            random_seed=self.settings.autotune_random_seed,
            search_algorithm=type(self).__name__,
        )
        (
            self._baseline_output,
            self._mutated_arg_indices,
            self._baseline_post_args,
        ) = self._compute_baseline()
        self._effective_atol, self._effective_rtol = (
            self._compute_effective_tolerances()
        )
        self._jobs = self._decide_num_jobs()

    def _next_precompile_result_path(self) -> str:
        assert self._precompile_tmpdir is not None
        return os.path.join(
            self._precompile_tmpdir.name,
            f"result_{next(self._precompile_result_counter)}.pkl",
        )

    def cleanup(self) -> None:
        if self._precompile_tmpdir is not None:
            self._precompile_tmpdir.cleanup()
            self._precompile_tmpdir = None
        self._precompile_args_path = None
        self._precompile_result_counter = count()

    def _compute_baseline(
        self,
    ) -> tuple[object, Sequence[int], Sequence[object] | None]:
        """
        Compute baseline output for accuracy validation during autotuning.
        Also detect if the kernel mutates any of its input arguments.

        The baseline is computed in one of two ways:
        - If settings.autotune_baseline_fn is provided, use that custom function
        - Otherwise, run the kernel with the default config
        """
        new_args = _clone_args(self.args)

        # Use custom baseline function if provided
        if self.settings.autotune_baseline_fn is not None:
            try:
                baseline_output = self.settings.autotune_baseline_fn(*new_args)
                _bench_device_synchronize()
            except Exception as e:
                raise exc.AutotuneError(
                    "Custom baseline function failed while computing baseline.\n"
                    f"Baseline function: {self.settings.autotune_baseline_fn}\n"
                ) from e
        else:
            # Use default config
            baseline_config = self.config_spec.default_config()
            try:
                baseline_output = self.kernel.compile_config(
                    baseline_config, allow_print=False
                )(*new_args)
                _bench_device_synchronize()
            except Exception as e:
                decorator = self.kernel.format_kernel_decorator(
                    baseline_config, self.settings
                )
                log_generated_triton_code_debug(
                    self.log,
                    self.kernel,
                    baseline_config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.kernel.maybe_log_repro(self.log.error, new_args, baseline_config)
                raise exc.InvalidConfig(
                    "Default config failed while computing baseline.\n"
                    f"Default config: {decorator}\n"
                    f"{SUPPRESSED_TRITON_CODE_MSG}\n"
                    "To work around this error, you could set `@helion.kernel(autotune_baseline_fn=...)` "
                    "to provide a custom baseline function (e.g. PyTorch eager implementation of your kernel)."
                ) from e

        original_args_flat, _ = tree_flatten(self.args)
        new_args_flat, _ = tree_flatten(new_args)
        mutated_tensor_idxs = []
        # we should only count tensors, since they won't be bound or removed
        tensor_idx = 0
        for old, new in zip(original_args_flat, new_args_flat, strict=False):
            if not (isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor)):
                continue
            try:
                equal = torch.equal(new, old)
            except RuntimeError:
                # torch.equal and device-to-host copies can fail on some
                # devices (e.g., TPU for large tensors).  Conservatively
                # assume the argument was not mutated.
                equal = True
            if not equal:
                mutated_tensor_idxs.append(tensor_idx)
            tensor_idx += 1
        baseline_post_args = _clone_args(new_args, idx_to_clone=mutated_tensor_idxs)
        return baseline_output, mutated_tensor_idxs, baseline_post_args

    def _compute_effective_tolerances(self) -> tuple[float, float]:
        """
        Compute effective tolerances based on the dtypes in the baseline output.

        For low-precision dtypes (fp8), we need stricter tolerances to ensure
        bitwise comparison works correctly. This method automatically detects
        such dtypes and adjusts tolerances accordingly.

        Returns:
            A tuple of (atol, rtol) to use for accuracy validation.
        """
        # Default tolerance when not user-specified
        DEFAULT_TOL = 1e-2

        # Get user-specified or default tolerances
        atol = self.settings.autotune_baseline_atol
        rtol = self.settings.autotune_baseline_rtol

        # Collect all dtypes from baseline output and mutated args
        dtypes = set()

        def collect_dtypes(obj: object) -> object:
            if isinstance(obj, torch.Tensor):
                dtypes.add(obj.dtype)
            return obj

        tree_map_only(torch.Tensor, collect_dtypes, self._baseline_output)
        if len(self._mutated_arg_indices) > 0 and self._baseline_post_args is not None:
            tree_map_only(torch.Tensor, collect_dtypes, self._baseline_post_args)

        # Only apply strict tolerances if ALL dtypes are fp8
        # Mixed dtypes (fp8 + fp32) would be too strict with atol=0.0, rtol=0.0
        all_dtypes_are_fp8 = dtypes and all(dtype in _FP8_DTYPES for dtype in dtypes)

        if all_dtypes_are_fp8:
            # All dtypes are fp8 - use bitwise comparison
            # unless the user explicitly set either tolerance value (i.e., not None)
            if atol is None and rtol is None:
                self.log(
                    f"Detected fp8 dtype(s) in output: {dtypes}. "
                    "Using bitwise comparison (atol=0.0, rtol=0.0) for autotuning accuracy check."
                )
                return 0.0, 0.0

        # Use user-specified values or defaults
        return (
            atol if atol is not None else DEFAULT_TOL,
            rtol if rtol is not None else DEFAULT_TOL,
        )

    def _decide_num_jobs(self) -> int:
        if not self.settings.autotune_precompile:
            return 1

        jobs = self.settings.autotune_precompile_jobs
        if not jobs:
            jobs = os.cpu_count() or 1

        if self.settings.autotune_precompile != "spawn":
            return jobs

        memory_per_job = _estimate_tree_bytes(self.args) + _estimate_tree_bytes(
            self._baseline_output
        )
        memory_per_job *= 2  # safety factor
        if memory_per_job <= 0:
            return jobs

        device = self.kernel.env.device
        if device.type != "cuda":
            # TODO(jansel): support non-cuda devices
            return jobs

        available_memory, _ = torch.cuda.mem_get_info(device)
        jobs_by_memory = available_memory // memory_per_job
        if jobs_by_memory < jobs:
            gib_per_job = memory_per_job / (1024**3)
            available_gib = available_memory / (1024**3)
            if jobs_by_memory > 0:
                self.log.warning(
                    f"Reducing autotune precompile spawn jobs from {jobs} to {jobs_by_memory} "
                    f"due to limited GPU memory (estimated {gib_per_job:.2f} GiB per job, "
                    f"{available_gib:.2f} GiB free). "
                    f"Set HELION_AUTOTUNE_PRECOMPILE_JOBS={jobs_by_memory} "
                    "to make this lower cap persistent, "
                    'set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            else:
                raise exc.AutotuneError(
                    "Autotune precompile spawn mode requires at least one job, but estimated "
                    "memory usage exceeds available GPU memory."
                    f"Estimated {gib_per_job:.2f} GiB per job, but only "
                    f"{available_gib:.2f} GiB free. "
                    'Set HELION_AUTOTUNE_PRECOMPILE="fork" to disable spawning, or reduce GPU memory usage.'
                )
            jobs = jobs_by_memory

        return jobs

    def _validate_against_baseline(
        self, config: Config, output: object, args: Sequence[object]
    ) -> bool:
        try:
            _autotune_outputs_match_baseline(
                output,
                args,
                baseline_output=self._baseline_output,
                baseline_post_args=self._baseline_post_args,
                mutated_arg_indices=self._mutated_arg_indices,
                atol=self._effective_atol,
                rtol=self._effective_rtol,
            )
        except AssertionError as e:
            if not self.settings.autotune_ignore_errors:
                self.log.warning(
                    f"Skipping config with accuracy mismatch: {config!r}\n{e!s}\nUse HELION_AUTOTUNE_ACCURACY_CHECK=0 to disable this check.\n"
                )
            return False
        return True

    def benchmark(self, config: Config) -> tuple[Callable[..., object], float]:
        """
        Benchmark a specific configuration.

        This method compiles the kernel with the given configuration and measures its performance.

        Args:
            config: The configuration to benchmark.

        Returns:
            The function and performance of the configuration in ms.
        """
        _npu_trace_autotune_candidate(self.kernel, config, "pre_compile(slow_path)")
        try:
            fn = self.kernel.compile_config(config, allow_print=False)
        except BaseException as compile_err:
            if _autotune_stderr_debug_enabled(self.kernel):
                print(
                    f"[helion autotune] COMPILE FAILED for config={config!r}: "
                    f"{type(compile_err).__name__}: {compile_err}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        if self.create_precompile_future(config, fn)():
            return fn, self.benchmark_function(config, fn)
        return fn, inf

    def _handle_autotune_benchmark_runtime_error(
        self,
        config: Config,
        fn: CompiledConfig,
        e: BaseException,
        *,
        captured_output: str | None = None,
        classification_override: str | None = None,
        force_unrecoverable: bool = False,
    ) -> None:
        """Log and classify a failed benchmark; may raise for fatal error policies."""
        e.__traceback__ = None
        if _autotune_stderr_debug_enabled(self.kernel):
            print(
                f"[helion autotune] BENCHMARK FAILED for config={config!r}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
        maybe_dump_triton_failure(
            self.kernel,
            config,
            e,
            captured_output=captured_output,
        )
        if force_unrecoverable or match_unrecoverable_runtime_error(e):
            self.kernel.maybe_log_repro(self.log.error, self.args, config)
            raise exc.TritonUnrecoverableRuntimeError(
                reason=str(e),
                decorator=self.kernel.format_kernel_decorator(config, self.settings),
                error=f"{type(e).__qualname__}: {e}",
            ) from e
        _backend = getattr(getattr(self, "config_spec", None), "backend", None)
        action = classification_override or (
            (_backend.classify_autotune_exception(e) if _backend is not None else None)
            or classify_triton_exception(e)
        )
        if self.settings.autotune_ignore_errors:
            pass
        elif action == "raise":
            decorator = self.kernel.format_kernel_decorator(config, self.settings)
            log_generated_triton_code_debug(
                self.log,
                self.kernel,
                config,
                prefix=f"Generated Triton code for {decorator}:",
            )
            self.kernel.maybe_log_repro(self.log.error, self.args, config)
            raise exc.TritonError(
                error=f"{type(e).__qualname__}: {e}",
                decorator=decorator,
                code=SUPPRESSED_TRITON_CODE_MSG,
            ) from e
        elif action == "warn":
            decorator = self.kernel.format_kernel_decorator(config, self.settings)
            log_generated_triton_code_debug(
                self.log,
                self.kernel,
                config,
                prefix=f"Generated Triton code for {decorator}:",
            )
            self.log.warning(format_triton_compile_failure(config, e, self.kernel))
            self.kernel.maybe_log_repro(self.log.warning, self.args, config)
        else:
            decorator = self.kernel.format_kernel_decorator(config, self.settings)
            log_generated_triton_code_debug(
                self.log,
                self.kernel,
                config,
                prefix=f"Generated Triton code for {decorator}:",
            )
            self.log.debug(f"Benchmarking failed: {type(e).__name__}: {e}")
            self.kernel.maybe_log_repro(self.log.debug, self.args, config)

        self._autotune_metrics.num_compile_failures += 1

    def benchmark_function(self, config: Config, fn: CompiledConfig) -> float:
        """
        Benchmark a compiled function.  This function is called by the autotuner to measure the
        performance of a specific configuration.

        Args:
            config: The configuration to benchmark.
            fn: A precompiled version of config.

        Returns:
            The performance of the configuration in ms.
        """
        self._autotune_metrics.num_configs_tested += 1
        self.log.debug(lambda: f"Running benchmark for {config!r}")
        _captured_output: list[str] = [""]
        _capture_ctx = (
            capture_output()
            if _get_failure_dump_dir()
            else contextlib.nullcontext(_captured_output)
        )
        try:
            # TODO(jansel): early exit with fewer trials if early runs are slow
            self.log.debug(lambda: f"Running {config} at {datetime.datetime.now()}")
            t0 = time.perf_counter()
            if len(self._mutated_arg_indices) > 0:
                working_args = _clone_args(
                    self.args, idx_to_clone=self._mutated_arg_indices
                )
            else:
                working_args = self.args
            _npu_trace_autotune_candidate(
                self.kernel, config, "pre_device_launch", persist=True
            )
            _bench_device_synchronize()
            with _capture_ctx as _captured_output:
                output = fn(*working_args)  # make sure the kernel is compiled
            _bench_device_synchronize()
            if (
                self.settings.autotune_accuracy_check
                and not self._validate_against_baseline(config, output, working_args)
            ):
                self._autotune_metrics.num_accuracy_failures += 1
                return inf

            t1 = time.perf_counter()
            _backend = getattr(getattr(self, "config_spec", None), "backend", None)
            _bench_fn = (
                _backend.get_do_bench() if _backend is not None else None
            ) or default_do_bench()
            res = _bench_fn(
                functools.partial(fn, *working_args),
                return_mode="median",
                warmup=1,  # we are already warmed up above
                rep=50,
            )
            res = sync_object(res)
            t2 = time.perf_counter()
            assert isinstance(res, float)
            self.log.debug(
                lambda: f"result: {res:.4f}ms (took {t1 - t0:.1f}s + {t2 - t1:.1f}s)",
            )
            if res < self.best_perf_so_far:
                self.best_perf_so_far = res
            return res
        except Exception as e:
            # e.__traceback__ holds references to all local variables in the call stack frames.
            # When a Triton kernel fails, the output tensors allocated by the Helion kernel function
            # were being held by the traceback, preventing them from being freed.
            self._handle_autotune_benchmark_runtime_error(
                config,
                fn,
                e,
                captured_output=_captured_output[0] or None,
            )
            return inf

    def set_adaptive_compile_timeout(
        self,
        members: list[PopulationMember],
        min_seconds: float,
        quantile: float,
    ) -> None:
        """
        Compute and set an adaptive compile timeout based on observed compile times.

        Uses the specified quantile of compile times from the population:
            adaptive_timeout = min(max(quantile_value, min_seconds), original_timeout)

        This feature must be enabled via the setting autotune_adaptive_timeout=True
        or the environment variable HELION_AUTOTUNE_ADAPTIVE_TIMEOUT=1.

        Args:
            members: List of population members with compile_time information.
            min_seconds: Lower bound for the adaptive timeout in seconds.
            quantile: The quantile of compile times to use (e.g., 0.9 for 90th percentile).
        """
        if not self.settings.autotune_adaptive_timeout:
            return

        # Collect valid compile times (non-None and positive)
        compile_times = [
            m.compile_time
            for m in members
            if m.compile_time is not None and m.compile_time > 0
        ]

        if not compile_times:
            self.log("No valid compile times found, keeping default timeout")
            return

        original_timeout = self.settings.autotune_compile_timeout

        # Compute the quantile
        compile_times_sorted = sorted(compile_times)
        quantile_index = min(
            int(len(compile_times_sorted) * quantile),
            len(compile_times_sorted) - 1,
        )
        quantile_value = compile_times_sorted[quantile_index]

        # adaptive_timeout = min(max(quantile_value, min_seconds), original_timeout)
        adaptive_timeout = int(min(max(quantile_value, min_seconds), original_timeout))

        self.settings.autotune_compile_timeout = adaptive_timeout

        self.log(
            f"Adaptive compile timeout: {adaptive_timeout}s "
            f"({quantile:.0%} percentile={quantile_value:.1f}s, "
            f"bounds=[{min_seconds}s, {original_timeout}s])"
        )

    def create_precompile_future(
        self,
        config: Config,
        fn: CompiledConfig,
        *,
        fork_triton_cache_dir: str | None = None,
    ) -> PrecompileFuture:
        """
        Run the kernel in a spawned subprocess to detect hangs during compilation or execution.
        We use the subprocess timeout to guard against Triton kernels that never finish.
        We also do this in parallel (when called from parallel_benchmark) to do faster autotuning.
        Note that we compile in parallel, but we benchmark one-by-one to avoid noisy results.

        Args:
            config: The config that generated fn.
            fn: The function to be precompiled.
            fork_triton_cache_dir: Optional ``TRITON_CACHE_DIR`` for the precompile child.

        Returns:
            True if the compilation was successful, False if it hung.
        """
        if not self.settings.autotune_precompile:
            return PrecompileFuture.skip(self, config, True)
        mode = self.settings.autotune_precompile
        if mode not in {"fork", "spawn"}:
            raise exc.InvalidAPIUsage("autotune_precompile must be 'fork' or 'spawn'")
        if len(self._mutated_arg_indices) > 0:
            device_args = _clone_args(self.args, idx_to_clone=self._mutated_arg_indices)
        else:
            device_args = self.args

        decorator = self.kernel.format_kernel_decorator(config, self.settings)

        if mode == "spawn":
            ctx = mp.get_context("spawn")
            assert self._precompile_args_path is not None
            try:
                fn_spec = _serialize_compiled_fn(fn)
            except RuntimeError as err:
                raise exc.AutotuneError(
                    "Failed to serialize compiled kernel for spawn precompile."
                    ' Set HELION_AUTOTUNE_PRECOMPILE="fork" to fall back to fork mode.'
                ) from err
            result_path = self._next_precompile_result_path()
            process = cast(
                "mp.Process",
                ctx.Process(
                    target=_run_kernel_in_subprocess_spawn,
                    args=(fn_spec, self._precompile_args_path, result_path, decorator),
                ),
            )
            process.daemon = True
        else:
            precompiler = _prepare_precompiler_for_fork(
                fn, device_args, config, self.kernel, decorator, self.log
            )
            if precompiler is None:
                return PrecompileFuture.skip(self, config, True)
            ctx = mp.get_context("fork")
            result_path = self._next_precompile_result_path()
            process = cast(
                "mp.Process",
                ctx.Process(
                    target=_run_kernel_in_subprocess_fork,
                    args=(precompiler, config, self.kernel, result_path, decorator),
                ),
            )
            process.daemon = True
        return PrecompileFuture(
            search=self,
            config=config,
            process=process,
            timeout=self.settings.autotune_compile_timeout,
            result_path=result_path,
            fork_triton_cache_dir=fork_triton_cache_dir,
        )

    def _autotune_parallel_bench_kind(self) -> Literal["default", "generic"]:
        _backend = getattr(getattr(self, "config_spec", None), "backend", None)
        _custom = _backend.get_do_bench() if _backend is not None else None
        if _custom is do_bench_generic:
            return "generic"
        return "default"

    def _finish_subprocess_explore_benchmark(
        self,
        config: Config,
        fn: CompiledConfig,
        message_data: dict[str, object],
    ) -> tuple[float, Literal["ok", "error"]]:
        self._autotune_metrics.num_configs_tested += 1
        status_raw = message_data.get("status")
        if status_raw == "ok":
            perf_ms = message_data.get("perf_ms")
            if not isinstance(perf_ms, float):
                raise TypeError(
                    f"Unexpected perf_ms in subprocess benchmark result: {perf_ms!r}"
                )
            if perf_ms < self.best_perf_so_far:
                self.best_perf_so_far = perf_ms
            st: Literal["ok", "error"] = "ok" if math.isfinite(perf_ms) else "error"
            return perf_ms, st
        if status_raw == "accuracy_fail":
            self._autotune_metrics.num_accuracy_failures += 1
            return inf, "error"
        exc_args_raw = message_data.get("exc_args", ())
        if not isinstance(exc_args_raw, tuple):
            exc_args_raw = (str(exc_args_raw),)
        err = RemoteError(
            exc_type=str(message_data.get("exc_type", "Exception")),
            exc_module=(
                str(m) if (m := message_data.get("exc_module")) is not None else None
            ),
            exc_args=cast("tuple[object, ...]", exc_args_raw),
            traceback=cast("str | None", message_data.get("traceback")),
            classification=cast("str | None", message_data.get("classification")),
            captured_output=cast("str | None", message_data.get("captured_output")),
        )
        e = err.to_exception()
        self._handle_autotune_benchmark_runtime_error(
            config,
            fn,
            e,
            captured_output=err.captured_output,
            classification_override=err.classification,
            force_unrecoverable=bool(message_data.get("unrecoverable")),
        )
        return inf, "error"

    def parallel_benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """
        Benchmark multiple configurations in parallel.

        Compilation (and optional precompile subprocesses) runs concurrently; timing
        of successful builds is sequential by default. Set
        ``HELION_AUTOTUNE_BENCHMARK_JOBS`` to an integer greater than ``1`` to run
        the explore-phase ``benchmark_function`` work in up to that many spawned
        worker processes (requires serializable compiled kernels). Under pytest or
        Meta unit tests this pool is skipped (jobs treated as ``1``) to avoid hangs;
        set ``HELION_AUTOTUNE_BENCHMARK_SUBPROCESS_IN_TEST=1`` to force it on.

        Args:
            configs: A list of configurations to benchmark.
            desc: Description for the progress bar.

        Returns:
            A list of BenchmarkResult entries containing the configuration, compiled
            callable, measured performance, status, and compilation time.
        """
        from .local_cache import autotune_fresh_triton_subdir_per_benchmark
        from .local_cache import autotune_stable_triton_subdir_per_config
        from .local_cache import per_config_triton_cache_for_autotune
        from .local_cache import triton_cache_dir_for_autotune_candidate

        fresh_subdir = autotune_fresh_triton_subdir_per_benchmark()
        stable_subdir = fresh_subdir and autotune_stable_triton_subdir_per_config()
        fns: list[Callable[..., object]] = []
        if fresh_subdir:
            triton_bench_dirs = [
                triton_cache_dir_for_autotune_candidate(c, stable=stable_subdir)
                for c in configs
            ]
            for config, bench_dir in zip(configs, triton_bench_dirs, strict=True):
                _npu_trace_autotune_candidate(
                    self.kernel, config, "pre_compile(parallel_batch)"
                )
                with per_config_triton_cache_for_autotune(bench_dir):
                    fn = self.kernel.compile_config(config, allow_print=False)
                fns.append(fn)
        else:
            triton_bench_dirs = [None] * len(configs)
            for config in configs:
                _npu_trace_autotune_candidate(
                    self.kernel, config, "pre_compile(parallel_batch)"
                )
                fn = self.kernel.compile_config(config, allow_print=False)
                fns.append(fn)

        futures: list[PrecompileFuture] | None = None
        if self.settings.autotune_precompile:
            futures = [
                self.create_precompile_future(
                    cfg,
                    fn,
                    fork_triton_cache_dir=bd,
                )
                for cfg, fn, bd in zip(configs, fns, triton_bench_dirs, strict=True)
            ]
            precompile_desc = (
                f"{desc} precompiling" if self.settings.autotune_progress_bar else None
            )
            is_workings = PrecompileFuture.wait_for_all(futures, desc=precompile_desc)
            precompile_status: list[Literal["ok", "error", "timeout"]] = []
            for future, ok in zip(futures, is_workings, strict=True):
                reason = future.failure_reason
                if ok:
                    precompile_status.append("ok")
                elif reason == "timeout":
                    precompile_status.append("timeout")
                else:
                    precompile_status.append("error")
        else:
            is_workings = [True] * len(configs)
            precompile_status = ["ok"] * len(configs)

        n = len(configs)
        slot_results: list[BenchmarkResult | None] = [None] * n

        def compile_time_at(index: int) -> float | None:
            if futures is None:
                return None
            future = futures[index]
            return (
                future.elapsed
                if future.process is not None and future.started
                else None
            )

        def finalize_triton_cache_dir(index: int) -> None:
            bench_dir = triton_bench_dirs[index]
            if bench_dir is not None:
                self.kernel.invalidate_compile_cache_entry(configs[index])
                if not stable_subdir:
                    shutil.rmtree(bench_dir, ignore_errors=True)

        def explore_sequential(index: int) -> None:
            config = configs[index]
            fn = fns[index]
            bench_dir = triton_bench_dirs[index]
            compile_time = compile_time_at(index)
            self.log.record_autotune_entry(
                AutotuneLogEntry(
                    generation=self._autotune_metrics.num_generations,
                    status="started",
                    perf_ms=None,
                    compile_time=compile_time,
                    config=config,
                )
            )
            if bench_dir is not None:
                os.environ["TRITON_CACHE_DIR"] = bench_dir
            perf = self.benchmark_function(config, fn)
            status: Literal["ok", "error"] = "ok" if math.isfinite(perf) else "error"
            self.log.record_autotune_entry(
                AutotuneLogEntry(
                    generation=self._autotune_metrics.num_generations,
                    status=status,
                    perf_ms=perf if math.isfinite(perf) else None,
                    compile_time=compile_time,
                    config=config,
                )
            )
            slot_results[index] = BenchmarkResult(
                config=config,
                fn=fn,
                perf=perf,
                status=status,
                compile_time=compile_time,
            )
            finalize_triton_cache_dir(index)

        working_indices: list[int] = []
        for index in range(n):
            fn_i = fns[index]
            is_working = is_workings[index]
            reason = precompile_status[index]
            compile_time = compile_time_at(index)
            if not is_working:
                status: Literal["ok", "error", "timeout"] = (
                    "timeout" if reason == "timeout" else "error"
                )
                slot_results[index] = BenchmarkResult(
                    config=configs[index],
                    fn=fn_i,
                    perf=inf,
                    status=status,
                    compile_time=compile_time,
                )
                finalize_triton_cache_dir(index)
            else:
                working_indices.append(index)

        benchmark_jobs = self.settings.autotune_benchmark_jobs
        effective_benchmark_jobs = benchmark_jobs
        if (
            effective_benchmark_jobs > 1
            and not _autotune_parallel_benchmark_workers_allowed()
        ):
            effective_benchmark_jobs = 1
        spawnable: list[tuple[int, SerializedCompiledFunction]] = []
        if effective_benchmark_jobs > 1:
            from contextlib import suppress

            for index in working_indices:
                with suppress(RuntimeError):
                    spawnable.append(
                        (
                            index,
                            _serialize_compiled_fn(
                                cast("CompiledConfig", fns[index]),
                            ),
                        )
                    )

        use_pool = effective_benchmark_jobs > 1 and len(spawnable) > 0
        if not use_pool:
            iterator = iter_with_progress(
                working_indices,
                total=len(working_indices),
                description=f"{desc} exploring neighbors",
                enabled=self.settings.autotune_progress_bar,
            )
            for index in iterator:
                explore_sequential(index)
        else:
            from rich.console import Console
            from rich.progress import BarColumn
            from rich.progress import MofNCompleteColumn
            from rich.progress import Progress
            from rich.progress import TextColumn

            from .progress_bar import SpeedColumn

            tmpdir = self._precompile_tmpdir
            assert tmpdir is not None
            bench_kind = self._autotune_parallel_bench_kind()
            baseline_bundle_path: str | None = None
            if self.settings.autotune_accuracy_check:
                baseline_bundle_path = os.path.join(
                    tmpdir.name, f"baseline_bench_{uuid.uuid4().hex}.pt"
                )
                torch.save(
                    (self._baseline_output, self._baseline_post_args),
                    baseline_bundle_path,
                )

            spawn_by_index = dict(spawnable)
            not_spawnable = [i for i in working_indices if i not in spawn_by_index]
            total_tasks = len(working_indices)
            progress_console = Console(stderr=True)
            use_rich_pb = (
                self.settings.autotune_progress_bar
                and not torch._utils_internal.is_fb_unit_test()
                and (progress_console.is_interactive or progress_console.is_jupyter)
            )

            def run_parallel_spawn_phase(
                advance: Callable[[], None] | None,
            ) -> None:
                max_workers = min(effective_benchmark_jobs, len(spawnable))
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=max_workers, mp_context=ctx
                ) as executor:
                    future_meta: dict[
                        object,
                        tuple[int, str, str, str],
                    ] = {}
                    for index, fn_spec in spawnable:
                        config = configs[index]
                        compile_time = compile_time_at(index)
                        self.log.record_autotune_entry(
                            AutotuneLogEntry(
                                generation=self._autotune_metrics.num_generations,
                                status="started",
                                perf_ms=None,
                                compile_time=compile_time,
                                config=config,
                            )
                        )
                        payload_path = os.path.join(
                            tmpdir.name, f"bpay_{uuid.uuid4().hex}.pkl"
                        )
                        result_path = self._next_precompile_result_path()
                        if len(self._mutated_arg_indices) > 0:
                            working_args = _clone_args(
                                self.args,
                                idx_to_clone=self._mutated_arg_indices,
                            )
                        else:
                            working_args = self.args
                        args_path = os.path.join(
                            tmpdir.name, f"bargs_{uuid.uuid4().hex}.pt"
                        )
                        torch.save(working_args, args_path)
                        decorator = self.kernel.format_kernel_decorator(
                            config, self.settings
                        )
                        payload = _AutotuneBenchSubprocessPayload(
                            fn_spec=fn_spec,
                            working_args_path=args_path,
                            bench_dir=triton_bench_dirs[index],
                            accuracy_check=self.settings.autotune_accuracy_check,
                            baseline_bundle_path=baseline_bundle_path,
                            atol=self._effective_atol,
                            rtol=self._effective_rtol,
                            mutated_arg_indices=tuple(self._mutated_arg_indices),
                            bench_kind=bench_kind,
                            decorator=decorator,
                        )
                        with open(payload_path, "wb") as pf:
                            pickle.dump(payload, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        fut = executor.submit(
                            _autotune_bench_subprocess_worker,
                            payload_path,
                            result_path,
                        )
                        future_meta[fut] = (index, payload_path, result_path, args_path)

                    for fut in as_completed(future_meta):
                        index, payload_path, result_path, args_path = future_meta[fut]
                        fut.result()
                        with open(result_path, "rb") as rf:
                            msg = pickle.load(rf)
                        assert isinstance(msg, dict)
                        for path in (payload_path, result_path, args_path):
                            with contextlib.suppress(Exception):
                                if os.path.exists(path):
                                    os.remove(path)
                        config = configs[index]
                        fn = fns[index]
                        compile_time = compile_time_at(index)
                        perf, st = self._finish_subprocess_explore_benchmark(
                            config, cast("CompiledConfig", fn), msg
                        )
                        self.log.record_autotune_entry(
                            AutotuneLogEntry(
                                generation=self._autotune_metrics.num_generations,
                                status=st,
                                perf_ms=perf if math.isfinite(perf) else None,
                                compile_time=compile_time,
                                config=config,
                            )
                        )
                        slot_results[index] = BenchmarkResult(
                            config=config,
                            fn=fn,
                            perf=perf,
                            status=st,
                            compile_time=compile_time,
                        )
                        finalize_triton_cache_dir(index)
                        if advance is not None:
                            advance()

            if use_rich_pb:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    BarColumn(
                        bar_width=None,
                        complete_style="yellow",
                        finished_style="green",
                    ),
                    MofNCompleteColumn(),
                    SpeedColumn(),
                    console=progress_console,
                ) as progress:
                    task_id = progress.add_task(
                        f"{desc} exploring neighbors", total=total_tasks
                    )
                    for index in sorted(not_spawnable):
                        explore_sequential(index)
                        progress.update(task_id, advance=1)
                    run_parallel_spawn_phase(
                        lambda: progress.update(task_id, advance=1),
                    )
            else:
                for index in sorted(not_spawnable):
                    explore_sequential(index)
                run_parallel_spawn_phase(None)

        assert all(x is not None for x in slot_results)
        return cast("list[BenchmarkResult]", slot_results)

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """
        Perform autotuning to find the best configuration.

        This method searches for the optimal configuration by benchmarking multiple configurations.

        Returns:
            The best configuration found during autotuning.
        """
        self._prepare()
        start = time.perf_counter()
        exit_stack = contextlib.ExitStack()
        with exit_stack:
            if self.settings.autotune_log:
                exit_stack.enter_context(self.log.autotune_logging())
            self.log.reset()
            # Autotuner triggers bugs in remote triton compile service.
            # Skip storing Triton intermediate IRs (.ttir, .ttgir, .llir, etc.)
            # during autotuning to reduce cache size by ~40%. Only binaries and
            # metadata are needed for execution.
            env_overrides = {"TRITON_LOCAL_BUILD": "1"}
            if "TRITON_STORE_BINARY_ONLY" not in os.environ:
                env_overrides["TRITON_STORE_BINARY_ONLY"] = "1"
            exit_stack.enter_context(patch.dict(os.environ, env_overrides, clear=False))
            assert self._precompile_tmpdir is None
            tempdir = tempfile.TemporaryDirectory()
            self._precompile_tmpdir = tempdir
            if self.settings.autotune_precompile == "spawn":
                args_path = os.path.join(tempdir.name, "args.pt")
                torch.save(self.args, args_path)
                self._precompile_args_path = args_path
            exit_stack.callback(self.cleanup)
            try:
                best = self._autotune()
            finally:
                self._finalize_autotune_metrics()
        end = time.perf_counter()
        kernel_decorator = self.kernel.format_kernel_decorator(best, self.settings)
        self.log(
            f"Autotuning complete in {end - start:.1f}s after searching {self._autotune_metrics.num_configs_tested} configs.\n"
            "One can hardcode the best config and skip autotuning with:\n"
            f"    {kernel_decorator}\n",
            level=logging.INFO + 5,
        )
        cached_path = self.kernel.get_cached_path(best)
        if cached_path is not None:
            self.log(f"Code of selected kernel: {cached_path}")
        self.kernel.maybe_log_repro(self.log.warning, self.args, best)
        if self.settings.print_output_code:
            triton_code = self.kernel.to_triton_code(best)
            if triton_code is not None:
                print(triton_code, file=sys.stderr)
        return best

    def _autotune(self) -> Config:
        """
        Abstract method to perform the actual autotuning.

        This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def set_generation(self, generation: int) -> None:
        self._autotune_metrics.num_generations = generation

    def _finalize_autotune_metrics(self) -> None:
        self._autotune_metrics.best_perf_ms = (
            self.best_perf_so_far if math.isfinite(self.best_perf_so_far) else 0.0
        )
        self._autotune_metrics.finalize()
        _run_post_autotune_hooks(self._autotune_metrics)


@dataclasses.dataclass
class PopulationMember:
    """
    Represents a member of the population in population-based search algorithms.

    Attributes:
        perfs (list[float]): The performance of the configuration, accumulated over multiple benchmarks.
        flat_values (FlatConfig): The flat representation of the configuration values.
        config (Config): The full configuration object.
        compile_time (float | None): The compilation time for this configuration.
    """

    fn: Callable[..., object]
    perfs: list[float]
    flat_values: FlatConfig
    config: Config
    status: Literal["ok", "error", "timeout", "unknown"] = "unknown"
    compile_time: float | None = None

    @property
    def perf(self) -> float:
        return self.perfs[-1]


def performance(member: PopulationMember) -> float:
    """
    Retrieve the performance of a population member.  Used as a sort key.

    Args:
        member: The population member.

    Returns:
        The performance of the member.
    """
    return member.perf


def _estimate_tree_bytes(obj: object) -> int:
    """Estimate the memory usage of a pytree of objects, counting shared storage only once."""
    total = 0
    seen_ptrs: set[int] = set()

    def _accumulate(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        size = tensor.element_size() * tensor.numel()
        try:
            storage = tensor.untyped_storage()
        except RuntimeError:
            pass
        else:
            ptr = storage.data_ptr()
            if ptr in seen_ptrs:
                return tensor
            seen_ptrs.add(ptr)
            size = storage.nbytes()
        total += size
        return tensor

    tree_map_only(torch.Tensor, _accumulate, obj)
    return total


class PopulationBasedSearch(BaseSearch):
    """
    Base class for search algorithms that use a population of configurations.

    Attributes:
        population (list[PopulationMember]): The current population of configurations.
        flat_spec (list[ConfigSpecFragment]): The flattened configuration specification.
    """

    finishing_rounds: int = 0

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
    ) -> None:
        """
        Initialize the PopulationBasedSearch object.

        Args:
            kernel: The kernel to be tuned.
            args: The arguments to be passed to the kernel.
        """
        super().__init__(kernel, args)
        self.population: list[PopulationMember] = []
        self.config_gen: ConfigGeneration = self.config_spec.create_config_generation(
            overrides=self.settings.autotune_config_overrides or None,
            advanced_controls_files=self.settings.autotune_search_acf or None,
        )

    @property
    def best(self) -> PopulationMember:
        """
        Retrieve the best configuration in the population.

        Returns:
            The best population member.
        """
        return min(self.population, key=performance)

    @best.setter
    def best(self, value: PopulationMember) -> None:
        """Replace the current best member in the population."""
        idx = min(range(len(self.population)), key=lambda i: self.population[i].perf)
        self.population[idx] = value

    def benchmark_flat(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Benchmark a flat configuration.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with the benchmark results.
        """
        config = self.config_gen.unflatten(flat_values)
        member = PopulationMember(_unset_fn, [], flat_values, config)
        self.parallel_benchmark_population([member], desc="Benchmarking")
        return member

    def parallel_benchmark_flat(
        self, to_check: list[FlatConfig]
    ) -> list[PopulationMember]:
        """
        Benchmark multiple flat configurations in parallel.

        Args:
            to_check: A list of flat configurations to benchmark.

        Returns:
            A list of population members with the benchmark results.
        """
        result = [*map(self.make_unbenchmarked, to_check)]
        return self.parallel_benchmark_population(result)

    def make_unbenchmarked(self, flat_values: FlatConfig) -> PopulationMember:
        """
        Create a population member with unbenchmarked configuration.  You
        should pass the result of this to parallel_benchmark_population.

        Args:
            flat_values: The flat configuration values.

        Returns:
            A population member with undefined performance.
        """
        config = self.config_gen.unflatten(flat_values)
        return PopulationMember(_unset_fn, [], flat_values, config)

    def _get_current_hardware_and_specialization(
        self,
    ) -> tuple[str | None, str | None]:
        """
        Get the current hardware and specialization_key for matching cached configs.

        Returns:
            A tuple of (hardware, specialization_key) strings.
        """
        hardware = get_device_name(extract_device(self.args))

        inner_kernel = getattr(self.kernel, "kernel", None)
        if inner_kernel is None or not hasattr(inner_kernel, "specialization_key"):
            return hardware, None
        spec_key = inner_kernel.specialization_key(self.args)
        specialization_key = str(_normalize_spec_key(spec_key))

        return hardware, specialization_key

    def _find_similar_cached_configs(self, max_configs: int) -> list[SavedBestConfig]:
        """
        Find cached configs that match hardware, specialization_key, and
        structural fingerprint (config_spec_hash).

        Args:
            max_configs: Maximum number of configs to return.

        Returns:
            List of matching SavedBestConfig objects, sorted by file modification time (most recent first).
        """
        from .local_cache import get_helion_cache_dir
        from .local_cache import iter_cache_entries

        current_hardware, current_spec_key = (
            self._get_current_hardware_and_specialization()
        )
        if current_hardware is None or current_spec_key is None:
            return []

        current_fingerprint_hash = self.config_spec.structural_fingerprint_hash()

        matching: list[SavedBestConfig] = []
        for entry in iter_cache_entries(
            get_helion_cache_dir(),
            max_scan=self.settings.autotune_best_available_max_cache_scan,
        ):
            if entry.hardware != current_hardware:
                continue
            if _normalize_spec_key_str(entry.specialization_key) != current_spec_key:
                continue
            # Skip entries without a matching structural fingerprint or flat_config.
            if entry.config_spec_hash != current_fingerprint_hash:
                continue
            if entry.flat_config is None:
                continue
            matching.append(entry)
            if len(matching) >= max_configs:
                break

        return matching

    def _generate_best_available_population_flat(self) -> list[FlatConfig]:
        """
        Generate initial population using default config plus cached configs.

        Always starts with the default configuration, then adds up to
        MAX_BEST_AVAILABLE_CONFIGS matching cached configs from previous runs.
        No random configs are added.  Duplicate configs are discarded.

        Returns:
            A list of unique FlatConfig values for the initial population.
            Minimum size is 1 (just default), maximum is 1 + autotune_best_available_max_configs setting.
        """
        # Always start with the default config as FROM_DEFAULT
        default_flat = self.config_gen.default_flat()
        default_config = self.config_gen.unflatten(default_flat)
        seen: set[Config] = {default_config}
        result: list[FlatConfig] = [default_flat]
        self.log("Starting with default config")

        max_configs = self.settings.autotune_best_available_max_configs
        cached_entries = self._find_similar_cached_configs(max_configs)

        if cached_entries:
            self.log.debug(
                f"Found {len(cached_entries)} cached config(s) from previous runs"
            )

        duplicates = 0
        for i, entry in enumerate(cached_entries):
            try:
                self.log.debug(f"Cached config {i + 1}: {entry.config}")
                flat = entry.to_mutable_flat_config()
                transferred_config = self.config_gen.unflatten(flat)
                if transferred_config in seen:
                    duplicates += 1
                    self.log.debug(
                        f"Cached config {i + 1} is a duplicate, skipping: {transferred_config}"
                    )
                    continue
                seen.add(transferred_config)
                result.append(flat)
                self.log.debug(
                    f"Cached config {i + 1} (transferred): {transferred_config}"
                )
            except (ValueError, TypeError, KeyError, AssertionError) as e:
                self.log(f"Failed to transfer cached config {i + 1}: {e}")
                continue

        if duplicates > 0:
            self.log.debug(f"Discarded {duplicates} duplicate config(s)")

        self.log(
            f"Initial population: 1 default + {len(result) - 1} unique cached = {len(result)} total"
        )

        return result

    def parallel_benchmark_population(
        self, members: list[PopulationMember], *, desc: str = "Benchmarking"
    ) -> list[PopulationMember]:
        """
        Benchmark multiple population members in parallel.  Members should be created with make_unbenchmarked.

        Args:
            members: The list of population members to benchmark.
            desc: Description for the progress bar.
        """
        results = self.parallel_benchmark([m.config for m in members], desc=desc)
        for member, result in zip(members, results, strict=True):
            assert result.config is member.config
            member.perfs.append(result.perf)
            member.fn = result.fn
            member.status = result.status
            member.compile_time = result.compile_time
        return members

    def compare(self, a: PopulationMember, b: PopulationMember) -> int:
        """
        Compare two population members based on their performance, possibly with re-benchmarking.

        Args:
            a: The first population member.
            b: The second population member.

        Returns:
            -1 if a is better than b, 1 if b is better than a, 0 if they are equal.
        """
        if self.should_rebenchmark(a) and self.should_rebenchmark(b):
            self.rebenchmark([a, b])
        return (a.perf > b.perf) - (a.perf < b.perf)

    def should_rebenchmark(self, member: PopulationMember) -> bool:
        """
        Determine if a population member should be re-benchmarked to avoid outliers.

        Args:
            member: The population member to check.

        Returns:
            True if the member should be re-benchmarked, False otherwise.
        """
        threshold = self.settings.get_rebenchmark_threshold()
        return member.perf < threshold * self.best_perf_so_far and math.isfinite(
            member.perf
        )

    def rebenchmark(
        self, members: list[PopulationMember], *, desc: str = "Rebenchmarking"
    ) -> None:
        """
        Re-benchmark a list of population members to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if len(members) < 2:
            return

        # Calculate repeat count based on best performance
        base_repeat = (
            int(200 / self.best_perf_so_far)
            if math.isfinite(self.best_perf_so_far) and self.best_perf_so_far > 0
            else 1000
        )
        repeat = min(1000, max(3, base_repeat))
        if len(self._mutated_arg_indices) > 0:
            bench_args = _clone_args(self.args, idx_to_clone=self._mutated_arg_indices)
        else:
            bench_args = self.args
        iterator = [functools.partial(m.fn, *bench_args) for m in members]
        _backend = getattr(getattr(self, "config_spec", None), "backend", None)
        _ib = (
            _backend.get_interleaved_bench() if _backend is not None else None
        ) or default_interleaved_bench()
        bench_fn: Callable[..., list[float]] = (
            self.settings.autotune_benchmark_fn or _ib
        )
        if self.settings.autotune_progress_bar:
            new_timings = bench_fn(iterator, repeat=repeat, desc=desc)
        else:
            new_timings = bench_fn(iterator, repeat=repeat)
        new_timings = sync_object(new_timings)
        for m, t in zip(members, new_timings, strict=True):
            m.perfs.append(t)
            if t < self.best_perf_so_far:
                self.best_perf_so_far = t

    def rebenchmark_population(
        self,
        members: list[PopulationMember] | None = None,
        *,
        desc: str = "Rebenchmarking",
    ) -> None:
        """
        Re-benchmark the entire population to avoid outliers.

        Args:
            members: The list of population members to rebenchmark.
            desc: Description for the progress bar.
        """
        if members is None:
            members = self.population
        self.rebenchmark([p for p in members if self.should_rebenchmark(p)], desc=desc)

    def statistics(self) -> str:
        """
        Generate statistics for the current population.

        Returns:
            A string summarizing the population performance.
        """
        return population_statistics(self.population)

    def run_finishing_phase(
        self, best: PopulationMember, rounds: int
    ) -> PopulationMember:
        """
        Run finishing rounds to minimize the configuration by resetting attributes to defaults.

        This phase attempts to simplify the found configuration by resetting as many
        attributes as possible to their default values, while ensuring performance
        does not get worse. It's similar to pattern search but mutations only move
        towards the default configuration.

        Args:
            best: The best configuration found during the main search.
            rounds: Number of finishing rounds to run. If 0, returns best unchanged.

        Returns:
            The minimized configuration (may be the same as input if no simplifications helped).
        """
        if rounds <= 0:
            return best

        self.log(f"Starting finishing phase with {rounds} rounds")
        default_flat = self.config_gen.default_flat()
        current = best

        for round_num in range(1, rounds + 1):
            simplified = False
            candidates: list[PopulationMember] = [current]

            # Generate candidates by resetting each parameter to its default
            for i in range(len(current.flat_values)):
                if current.flat_values[i] != default_flat[i]:
                    # Create a new config with this parameter reset to default
                    new_flat = [*current.flat_values]
                    new_flat[i] = default_flat[i]
                    candidate = self.make_unbenchmarked(new_flat)
                    # Only add if this produces a different config
                    if candidate.config != current.config:
                        candidates.append(candidate)

            if len(candidates) <= 1:
                self.log(f"Finishing round {round_num}: no more parameters to simplify")
                break

            # Benchmark the candidates
            unbenchmarked = [m for m in candidates if len(m.perfs) == 0]
            if unbenchmarked:
                self.set_generation(self._autotune_metrics.num_generations + 1)
                self.parallel_benchmark_population(
                    unbenchmarked, desc=f"Finishing round {round_num}"
                )

            # Rebenchmark all candidates (including current) for fair comparison
            self.rebenchmark(candidates, desc=f"Finishing round {round_num}: verifying")

            # Log performance of each candidate at debug level
            current_perf = current.perf
            for candidate in candidates[1:]:
                delta = candidate.perf - current_perf
                delta_pct = (delta / current_perf * 100) if current_perf != 0 else 0
                status = "ok" if candidate.perf <= current_perf else "worse"
                self.log.debug(
                    f"  reset to {candidate.config}: {candidate.perf:.4f}ms "
                    f"(delta={delta:+.4f}ms, {delta_pct:+.1f}%) [{status}]"
                )

            # Collect all single-attribute resets that maintained performance
            good_candidates = [
                c
                for c in candidates[1:]
                if math.isfinite(c.perf) and c.perf <= current.perf
            ]

            if len(good_candidates) > 1:
                # Try combining all good single-attribute resets at once
                combined_flat = [*current.flat_values]
                for c in good_candidates:
                    for i in range(len(combined_flat)):
                        if c.flat_values[i] != current.flat_values[i]:
                            combined_flat[i] = c.flat_values[i]
                combined = self.make_unbenchmarked(combined_flat)
                if combined.config != current.config:
                    self.parallel_benchmark_population(
                        [combined],
                        desc=f"Finishing round {round_num}: combined",
                    )
                    self.rebenchmark(
                        [current, combined],
                        desc=f"Finishing round {round_num}: verifying combined",
                    )
                    if math.isfinite(combined.perf) and combined.perf <= current.perf:
                        current = combined
                        simplified = True

            if not simplified and good_candidates:
                current = good_candidates[0]
                simplified = True

            if simplified:
                self.log(
                    f"Finishing round {round_num}: simplified to {current.config}, perf={current.perf:.4f}ms"
                )
            else:
                self.log(
                    f"Finishing round {round_num}: no simplification maintained performance, stopping early"
                )
                break

        # Minimize the final config by removing values that match defaults
        minimal_config = current.config.minimize(self.config_spec)
        current = PopulationMember(
            fn=current.fn,
            perfs=current.perfs,
            flat_values=current.flat_values,
            config=minimal_config,
            status=current.status,
            compile_time=current.compile_time,
        )
        self.log(f"Finishing phase complete: final config={current.config}")
        return current


def population_statistics(population: list[PopulationMember]) -> str:
    """
    Create a summary of the population performance.

    Args:
        population: The population of configurations.

    Returns:
        A string summarizing the performance of the population.
    """
    population = sorted(population, key=performance)
    status_counts: collections.Counter[str] = collections.Counter()
    working: list[PopulationMember] = []
    for member in population:
        status = member.status
        if math.isfinite(member.perf):
            working.append(member)
            if status not in {"ok", "error", "timeout"}:
                status = "ok"
        else:
            if status not in {"error", "timeout"}:
                status = "error"
        if status == "timeout":
            status_counts["timeout"] += 1
        elif status == "error":
            status_counts["error"] += 1
        else:
            status_counts["ok"] += 1
    if len(working) == 0:
        raise exc.NoConfigFound
    parts: list[str] = []
    for label in ("error", "timeout", "ok"):
        count = status_counts.get(label, 0)
        if count:
            parts.append(f"{label}={count}")

    parts.extend(
        (
            f"min={working[0].perf:.4f}",
            f"mid={working[len(working) // 2].perf:.4f}",
            f"max={working[-1].perf:.4f}",
            f"best={pprint.pformat(dict(population[0].config), width=100, compact=True)}",
        )
    )
    return "\n" + "\n".join(parts)


@dataclasses.dataclass
class PrecompileFuture:
    """
    Wraps a child process where we are precompiling a kernel.

    Attributes:
        search (BaseSearch): The search object that initiated the precompilation.
        config (Config): The configuration to be precompiled.
        process (mp.Process | None): The process running the precompilation.
        timeout (float): The timeout for the precompilation.
        start_time (float): The time when the precompilation started.
        end_time (float | None): The time when the precompilation ended.
        ok (bool | None): The result of the precompilation (True if successful, False otherwise).
    """

    search: BaseSearch
    config: Config
    process: mp.Process | None
    timeout: float
    # Set when the process is actually started. For queued futures this is None.
    start_time: float | None = None
    end_time: float | None = None
    ok: bool | None = None
    result_path: str | None = None
    _result_received: bool = False
    remote_error: RemoteError | None = None
    _remote_error_handled: bool = False
    failure_reason: Literal["ok", "error", "timeout"] | None = None
    fork_triton_cache_dir: str | None = None

    @property
    def elapsed(self) -> float:
        """Return the elapsed time since the start of the precompilation."""
        if self.start_time is None:
            return 0.0
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def seconds_left(self) -> float:
        """Return the number of seconds left before the timeout."""
        if self.end_time is not None:
            return 0
        if self.start_time is None:
            return self.timeout
        return self.timeout - (time.time() - self.start_time)

    def is_alive(self) -> bool:
        """Check if the precompilation process is still alive."""
        if (p := self.process) is None:
            return False
        return p.is_alive()

    @property
    def started(self) -> bool:
        """Whether the process has been started."""
        return self.start_time is not None

    def start(self) -> None:
        """Start the underlying process and set the timer if not already started."""
        if self.process is None or self.started:
            return
        if self.fork_triton_cache_dir is not None:
            os.environ["TRITON_CACHE_DIR"] = self.fork_triton_cache_dir
        self.start_time = time.time()
        self.process.start()

    @staticmethod
    def skip(search: BaseSearch, config: Config, ok: bool) -> PrecompileFuture:
        """Dummy precompile future that is already done."""
        ts = time.time()
        return PrecompileFuture(
            search=search,
            config=config,
            process=None,
            timeout=0,
            ok=ok,
            start_time=ts,
            end_time=ts,
            result_path=None,
            _result_received=True,
            remote_error=None,
            _remote_error_handled=True,
            failure_reason="ok" if ok else "error",
        )

    def __call__(self) -> bool:
        """Wait for the precompilation to finish and return true on success."""
        if self.ok is not None:
            return self.ok
        process = self.process
        assert process is not None
        try:
            # Start now if not already started (single-future path)
            if not self.started:
                self.start()
            process.join(self.seconds_left())
        finally:
            self._mark_complete()
        self._consume_result(raise_on_raise=True)
        assert self.ok is not None
        return self.ok

    @staticmethod
    def wait_for_all(
        futures: list[PrecompileFuture],
        desc: str | None = None,
    ) -> list[bool]:
        """
        Wait for all precompile futures to complete.

        Args:
            futures: A list of PrecompileFuture objects.
            desc: Optional description used for the progress display.

        Returns:
            A list of boolean values indicating completion status.
        """
        progress = iter_with_progress(
            range(len(futures)),
            total=len(futures),
            description=desc,
            enabled=desc is not None,
        )
        next(progress, None)  # display the progress bar immediately
        progress_left = len(futures)
        remaining = [f for f in futures if f.ok is None]
        try:
            while remaining:
                remaining = PrecompileFuture._wait_for_all_step(remaining)
                while progress_left > len(remaining):
                    next(progress, None)
                    progress_left -= 1
        except BaseException:
            PrecompileFuture._cancel_all(futures)
            raise
        result = []
        for f in futures:
            assert f.ok is not None
            if f.failure_reason is None:
                f.failure_reason = "ok" if f.ok else "error"
            result.append(f.ok)
        return result

    @staticmethod
    def _wait_for_all_step(
        futures: list[PrecompileFuture],
    ) -> list[PrecompileFuture]:
        """Start up to the concurrency cap, wait for progress, and return remaining futures."""
        cap = futures[0].search._jobs if futures else 1
        running = [f for f in futures if f.started and f.ok is None and f.is_alive()]

        # Start queued futures up to the cap
        queued = collections.deque(f for f in futures if not f.started and f.ok is None)
        while len(running) < cap and queued:
            job = queued.popleft()
            job.start()
            if job.is_alive():
                running.append(job)

        # Wait for at least one to finish or time out
        timeout = min([f.seconds_left() for f in running], default=0.0)
        handles = [f.process.sentinel for f in running if f.process is not None]
        if handles and timeout > 0:
            connection.wait(handles, timeout)
        remaining: list[PrecompileFuture] = []
        for f in futures:
            if f.ok is not None:
                continue
            if f.started and (not f.is_alive() or f.seconds_left() <= 0):
                f._mark_complete()
                f._consume_result(raise_on_raise=True)
            else:
                remaining.append(f)
        return remaining

    @staticmethod
    def _cancel_all(futures: Iterable[PrecompileFuture]) -> None:
        """Cancel any futures that have not completed."""
        active = [future for future in futures if future.ok is None]
        for future in active:
            with contextlib.suppress(Exception):
                future._kill_without_wait()
        for future in active:
            with contextlib.suppress(Exception):
                future.cancel()

    def _kill_without_wait(self) -> None:
        """Issue a hard kill to the underlying process without waiting for exit."""
        process = self.process
        if process is None or not self.started:
            return
        if process.is_alive():
            with contextlib.suppress(Exception):
                process.kill()

    def cancel(self) -> None:
        """Terminate the underlying process (if any) without waiting for success."""
        self.end_time = time.time()
        process = self.process
        if process is not None:
            if self.started:
                with contextlib.suppress(Exception):
                    if process.is_alive():
                        process.kill()
                    process.join()
        if self.ok is None:
            self.ok = False
        if self.failure_reason is None:
            self.failure_reason = "error"
        self._consume_result(raise_on_raise=False)

    def _mark_complete(self) -> bool:
        """
        Mark the precompile future as complete and kill the process if needed.

        Returns:
            True if the precompilation was successful, False otherwise.
        """
        self.end_time = time.time()
        process = self.process
        assert process is not None
        # If the process hasn't been started yet (shouldn't happen in normal flow),
        # start and immediately terminate to maintain invariants.
        if not self.started:
            self.start()
        if not process.is_alive():
            self.ok = process.exitcode == 0
            self._consume_result(raise_on_raise=False)
            if self.ok:
                self.failure_reason = "ok"
            elif self.failure_reason is None:
                self.failure_reason = "error"
            return self.ok
        process.terminate()
        process.join(10)
        msg = f"Timeout after {self.elapsed:.0f}s compiling {self.config}"
        if process.is_alive():
            if not self.search.settings.autotune_ignore_errors:
                self.search.log.warning(
                    msg,
                    "(SIGKILL required)",
                )
            process.kill()
            process.join()
        else:
            if not self.search.settings.autotune_ignore_errors:
                self.search.log.warning(msg)

        self.ok = False
        self.failure_reason = "timeout"
        self._consume_result(raise_on_raise=False)
        return False

    def _consume_result(self, *, raise_on_raise: bool) -> None:
        if not self._result_received and self.result_path is not None:
            message_data: dict[str, object] | None = None
            try:
                with open(self.result_path, "rb") as f:
                    message_data = pickle.load(f)
            except FileNotFoundError:
                message_data = None
            except Exception as err:
                if self.remote_error is None:
                    self.remote_error = RemoteError(
                        exc_type=type(err).__name__,
                        exc_module=type(err).__module__,
                        exc_args=(str(err),),
                        traceback=None,
                        classification="warn",
                    )
            finally:
                with contextlib.suppress(Exception):
                    os.remove(self.result_path)
            if message_data is None:
                if self.failure_reason == "timeout":
                    # Timeout warnings have already been emitted; suppress secondary EOF logs.
                    self.remote_error = None
                    self._remote_error_handled = True
                elif self.remote_error is None:
                    self.remote_error = RemoteError(
                        exc_type="EOFError",
                        exc_module=__name__,
                        exc_args=("No result received from subprocess.",),
                        traceback=None,
                        classification="debug",
                    )
            elif message_data["status"] == "ok":
                if self.ok is None:
                    self.ok = True
                assert self.remote_error is None
            else:
                exc_args_obj = message_data["exc_args"]
                if isinstance(exc_args_obj, tuple):
                    exc_args_tuple: tuple[object, ...] = exc_args_obj
                else:
                    exc_args_tuple = tuple(cast("Iterable[object]", exc_args_obj))
                self.remote_error = RemoteError(
                    exc_type=cast("str", message_data["exc_type"]),
                    exc_module=cast("str | None", message_data["exc_module"]),
                    exc_args=exc_args_tuple,
                    traceback=cast("str | None", message_data["traceback"]),
                    classification=cast("str | None", message_data["classification"]),
                    captured_output=cast(
                        "str | None", message_data.get("captured_output")
                    ),
                )
                self.ok = False
            self.result_path = None
            self._result_received = True

        error = self.remote_error
        if error is None or self._remote_error_handled:
            return
        exc_obj = error.to_exception()
        maybe_dump_triton_failure(
            self.search.kernel,
            self.config,
            exc_obj,
            remote_traceback=error.traceback,
            captured_output=error.captured_output,
        )
        classification = error.classification or classify_triton_exception(exc_obj)
        ignore_errors = self.search.settings.autotune_ignore_errors
        if ignore_errors:
            classification = "debug"
        if classification == "raise":
            if raise_on_raise:
                self._remote_error_handled = True
                decorator = self.search.kernel.format_kernel_decorator(
                    self.config, self.search.settings
                )
                log_generated_triton_code_debug(
                    self.search.log,
                    self.search.kernel,
                    self.config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.search.kernel.maybe_log_repro(
                    self.search.log.error, self.search.args, self.config
                )
                raise exc.TritonError(
                    error=f"{type(exc_obj).__qualname__}: {exc_obj}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from exc_obj
            return

        decorator = self.search.kernel.format_kernel_decorator(
            self.config, self.search.settings
        )
        log_generated_triton_code_debug(
            self.search.log,
            self.search.kernel,
            self.config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        formatted = format_triton_compile_failure(
            self.config, exc_obj, self.search.kernel
        )
        if error.traceback:
            formatted = (
                f"{formatted}\nRemote traceback (spawned process):\n{error.traceback}"
            )
        if classification == "warn":
            self.search.log.warning(formatted)
            self.search.kernel.maybe_log_repro(
                self.search.log.warning, self.search.args, self.config
            )
        elif not ignore_errors:
            self.search.log.debug(formatted)
            self.search.kernel.maybe_log_repro(
                self.search.log.debug, self.search.args, self.config
            )
        self._remote_error_handled = True


def _clone_tree(tree: object) -> object:
    def _clone(leaf: object) -> object:
        if isinstance(leaf, torch.Tensor):
            clone = leaf.detach().clone()
            clone.requires_grad_(leaf.requires_grad)
            return clone
        return leaf

    return tree_map(_clone, tree)


def _assert_args_close(
    actual: Sequence[object],
    expected: Sequence[object],
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    actual_flat, _ = tree_flatten(actual)
    expected_flat, _ = tree_flatten(expected)
    for act, exp in zip(actual_flat, expected_flat, strict=False):
        if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
            torch.testing.assert_close(act, exp, atol=atol, rtol=rtol)


def _write_result_file(result_path: str, message: dict[str, object]) -> None:
    tmp_path = f"{result_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(message, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, result_path)


def _run_kernel_in_subprocess_spawn(
    fn_spec: SerializedCompiledFunction,
    args_path: str,
    result_path: str,
    decorator: str,
) -> None:
    status = 0
    _cap: list[str] = [""]
    try:
        fn = _load_compiled_fn(fn_spec)
        args = torch.load(args_path)
        assert isinstance(args, (tuple, list))
        _bench_device_synchronize()
        with capture_output() as _cap:
            fn(*args)
        _bench_device_synchronize()
        _write_result_file(result_path, {"status": "ok"})
    except Exception as exc:
        status = 1
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                    "captured_output": _cap[0] or None,
                },
            )
    finally:
        os._exit(status)


def _prepare_precompiler_for_fork(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: _AutotunableKernel,
    decorator: str,
    logger: AutotuningLogger,
) -> Callable[[], None] | None:
    def extract_launcher(
        triton_kernel: object,
        grid: tuple[int, ...],
        *launch_args: object,
        **launch_kwargs: object,
    ) -> NoReturn:
        raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

    try:
        fn(*args, _launcher=extract_launcher)
        raise RuntimeError("Expected _ExtractedLaunchArgs to be raised")
    except _ExtractedLaunchArgs as extracted:
        # debug_jitfunction(extracted.kernel)
        precompiler = make_precompiler(
            cast("Any", extracted.kernel),
            config,
            cast("BoundKernel", kernel),
        )(*extracted.args, **extracted.kwargs)
        if precompiler is already_compiled:
            return None
        return precompiler
    except Exception as e:
        maybe_dump_triton_failure(kernel, config, e)
        log_generated_triton_code_debug(
            logger,
            kernel,
            config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        logger.warning(
            "Helion autotuner precompile error for %s. %s",
            decorator,
            SUPPRESSED_TRITON_CODE_MSG,
            exc_info=True,
        )
        raise


def _run_kernel_in_subprocess_fork(
    precompiler: Callable[[], None],
    config: Config,
    kernel: _AutotunableKernel,
    result_path: str,
    decorator: str,
) -> None:
    status = 0
    _cap: list[str] = [""]
    try:
        with capture_output() as _cap:
            precompiler()
        _write_result_file(result_path, {"status": "ok"})
    except Exception as exc:
        status = 1
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                    "captured_output": _cap[0] or None,
                },
            )
    finally:
        os._exit(status)


class _ExtractedLaunchArgs(Exception):
    """Exception that carries kernel launch arguments for precompiler extraction."""

    def __init__(
        self,
        kernel: object,
        grid: tuple[int, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.grid = grid
        self.args = args
        self.kwargs = kwargs


def _unset_fn(*args: object) -> NoReturn:
    raise RuntimeError("Uninitialized function")


@dataclasses.dataclass
class SerializedCompiledFunction:
    function_name: str
    source_code: str
    filename: str | None
    module_name: str | None


@dataclasses.dataclass
class _AutotuneBenchSubprocessPayload:
    fn_spec: SerializedCompiledFunction
    working_args_path: str
    bench_dir: str | None
    accuracy_check: bool
    baseline_bundle_path: str | None
    atol: float
    rtol: float
    mutated_arg_indices: tuple[int, ...]
    bench_kind: Literal["default", "generic"]
    decorator: str


@dataclasses.dataclass
class RemoteError:
    exc_type: str
    exc_module: str | None
    exc_args: tuple[object, ...]
    traceback: str | None
    classification: str | None
    captured_output: str | None = None

    def to_exception(self) -> Exception:
        exc_cls = types.new_class(self.exc_type, (Exception,))
        exc_cls.__module__ = self.exc_module or __name__
        exc_obj = exc_cls(*self.exc_args)
        exc_obj.remote_traceback = self.traceback
        return exc_obj


def _serialize_compiled_fn(fn: CompiledConfig) -> SerializedCompiledFunction:
    if "<locals>" in getattr(fn, "__qualname__", ""):
        raise RuntimeError("Unable to serialize nested compiled functions")
    module_name = getattr(fn, "__module__", None)
    module = sys.modules.get(module_name) if module_name is not None else None
    filename: str | None = None
    source_code: str | None = None
    if module is not None:
        filename = getattr(module, "__file__", None)
        if filename is not None and os.path.exists(filename):
            source_code = Path(filename).read_text(encoding="utf-8")
        if source_code is None:
            with contextlib.suppress(OSError, TypeError):
                source_code = inspect.getsource(module)
    if source_code is None:
        raise RuntimeError("Unable to capture source for compiled kernel")
    return SerializedCompiledFunction(
        function_name=fn.__name__,
        source_code=source_code,
        filename=filename,
        module_name=module_name,
    )


def _load_compiled_fn(fn_spec: SerializedCompiledFunction) -> CompiledConfig:
    module_name = f"_helion_autotune_subprocess_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = fn_spec.filename or "<helion-autotune-subprocess>"
    module.__loader__ = None
    module.__package__ = None
    sys.modules[module_name] = module
    exec(
        compile(fn_spec.source_code, module.__file__, "exec"),
        module.__dict__,
    )
    fn = getattr(module, fn_spec.function_name, None)
    if fn is None:
        raise RuntimeError(
            f"Unable to locate compiled kernel '{fn_spec.function_name}' in generated module"
        )
    return fn


def _autotune_bench_subprocess_worker(payload_path: str, result_path: str) -> None:
    """Run warmup, optional accuracy check, and timing in a spawned process."""
    _captured_output: list[str] = [""]
    payload: _AutotuneBenchSubprocessPayload | None = None
    try:
        with open(payload_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, _AutotuneBenchSubprocessPayload):
            raise TypeError(
                f"Expected _AutotuneBenchSubprocessPayload, got {type(payload)}"
            )
        if payload.bench_dir is not None:
            os.environ["TRITON_CACHE_DIR"] = payload.bench_dir
        fn = _load_compiled_fn(payload.fn_spec)
        working_args = torch.load(payload.working_args_path)
        assert isinstance(working_args, (list, tuple))
        _bench_fn = (
            do_bench_generic if payload.bench_kind == "generic" else default_do_bench()
        )
        _capture_ctx = (
            capture_output()
            if _get_failure_dump_dir()
            else contextlib.nullcontext(_captured_output)
        )
        _bench_device_synchronize()
        with _capture_ctx as _captured_output:
            output = fn(*working_args)
        _bench_device_synchronize()
        if payload.accuracy_check:
            if payload.baseline_bundle_path is None:
                raise RuntimeError(
                    "accuracy_check set but baseline_bundle_path is None"
                )
            baseline_output, baseline_post_args = torch.load(
                payload.baseline_bundle_path
            )
            try:
                _autotune_outputs_match_baseline(
                    output,
                    working_args,
                    baseline_output=baseline_output,
                    baseline_post_args=baseline_post_args,
                    mutated_arg_indices=payload.mutated_arg_indices,
                    atol=payload.atol,
                    rtol=payload.rtol,
                )
            except AssertionError:
                _write_result_file(result_path, {"status": "accuracy_fail"})
                return
        res = _bench_fn(
            functools.partial(fn, *working_args),
            return_mode="median",
            warmup=1,
            rep=50,
        )
        res = sync_object(res)
        assert isinstance(res, float)
        _write_result_file(result_path, {"status": "ok", "perf_ms": res})
    except Exception as exc:
        decorator = payload.decorator if payload is not None else "<unknown>"
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            unrecoverable = match_unrecoverable_runtime_error(exc)
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                    "captured_output": _captured_output[0] or None,
                    "unrecoverable": unrecoverable,
                },
            )
