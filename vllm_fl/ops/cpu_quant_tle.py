"""Process-wide ownership for the ARM quantized TLE runtime symbol."""

from collections.abc import Callable
from threading import Lock


_LOCK = Lock()
_ACTIVE_BACKEND: str | None = None


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
