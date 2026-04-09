from __future__ import annotations

from . import _compat as _compat_module  # noqa: F401  # side-effect import
from . import _logging
from . import exc
from . import language
from . import runtime
from ._utils import cdiv
from ._utils import next_power_of_2
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime.settings import RefMode
from .runtime.settings import Settings
from ._testing import is_npu
__all__ = [
    "Config",
    "Kernel",
    "RefMode",
    "Settings",
    "cdiv",
    "exc",
    "jit",
    "kernel",
    "language",
    "next_power_of_2",
    "runtime",
]

_logging.init_logs()

# Register with Dynamo after all modules are fully loaded
from ._compiler._dynamo.variables import register_dynamo_variable  # noqa: E402

register_dynamo_variable()
if is_npu():
    from torch_npu._inductor.codegen.ir_fx import _patch_npu_inductor_ir
    from torch_npu._inductor.lowering_fx import _register_npu_inductor_fallbacks
    _compat_module.register_npu_backend()
    _compat_module._register_interface_for_device()
    _patch_npu_inductor_ir()
    _register_npu_inductor_fallbacks()