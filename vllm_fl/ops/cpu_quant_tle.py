"""Process-wide ownership for the ARM quantized TLE runtime symbol."""

from collections.abc import Callable
from threading import Lock


_LOCK = Lock()
_ACTIVE_BACKEND: str | None = None
_BUILTIN_IMPORTS: dict[str, str] = {}
_INDUCTOR_PATCHED = False


def ensure_tle_backend(name: str, register: Callable[[], None]) -> None:
    """Register exactly one packed-weight ABI for the shared compiler op.

    W4A8 and W8A8 currently lower through the same external symbol, but their
    packed RHS layouts differ. They may be imported together, while attempting
    to execute both in one process is rejected explicitly.
    """
    global _ACTIVE_BACKEND
    with _LOCK:
        if _ACTIVE_BACKEND == name:
            return
        if _ACTIVE_BACKEND is not None:
            raise RuntimeError(
                "ARM quantized TLE backend is already active as "
                f"{_ACTIVE_BACKEND}; cannot activate {name} in the same process. "
                "Choose one of FL_CPU_INT4 or FL_CPU_INT8 and restart."
            )
        register()
        _ACTIVE_BACKEND = name


def register_inductor_builtin_import(trigger: str, import_line: str) -> None:
    """Register one custom builtin import with a single Inductor source patch."""
    import torch._inductor.async_compile as async_compile

    global _INDUCTOR_PATCHED
    with _LOCK:
        previous = _BUILTIN_IMPORTS.get(trigger)
        if previous is not None and previous != import_line:
            raise RuntimeError(f"conflicting Inductor import for {trigger}")
        _BUILTIN_IMPORTS[trigger] = import_line
        if _INDUCTOR_PATCHED:
            return

        original_triton = async_compile.AsyncCompile.triton

        def compile_triton(self, kernel_name, source_code, device_str="cpu"):
            with _LOCK:
                registered = tuple(_BUILTIN_IMPORTS.items())
            imports = [
                line
                for marker, line in registered
                if marker in source_code and line.strip() not in source_code
            ]
            if imports:
                source_code = "".join(imports) + source_code
            return original_triton(self, kernel_name, source_code, device_str)

        async_compile.AsyncCompile.triton = compile_triton
        _INDUCTOR_PATCHED = True
