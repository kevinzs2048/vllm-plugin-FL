"""FL-owned ARM CPU runtime integration for Qwen W4A8 and GDN.

This module replaces the former ``vllm-triton-cpu-qwen35`` general plugin.
It deliberately keeps FlagGems imports lazy so importing :mod:`vllm_fl` on a
GPU host, an x86 host, or a build machine does not require the ARM stack.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

_ACTIVE_BACKEND: str | None = None
_BACKEND_TO_FLAGGEMS_RUNTIME = {
    "libtriton_jit": "libtriton_jit",
    "tleraw": None,
}


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_int4_backend(configured: str | None = None) -> str:
    """Resolve the supported FL INT4 backend."""
    if configured is None:
        configured = os.getenv("FL_CPU_INT4_BACKEND", "tleraw")

    backend = configured.strip().lower()
    if backend not in _BACKEND_TO_FLAGGEMS_RUNTIME:
        choices = ", ".join(sorted(_BACKEND_TO_FLAGGEMS_RUNTIME))
        raise ValueError(f"FL_CPU_INT4_BACKEND must be one of: {choices}")
    return backend


def _configure_runtime_defaults() -> None:
    defaults = {
        "FLAGGEMS_VENDOR_NAME": "arm",
        "TRITON_BACKENDS_IN_TREE": "1",
        "TRITON_CPU_BACKEND": "1",
        "TRITON_CPU_FIXED_I8MM": "1",
        "FLAGGEMS_GDN_NATIVE_DECODE": "1",
        "FLAGGEMS_GDN_NATIVE_CONV": "1",
        "FLAGGEMS_GDN_NATIVE_PREFILL": "1",
        "FLAGGEMS_GDN_NATIVE_PACKED_DECODE": "1",
        "FLAGGEMS_GDN_NATIVE_NORM": "1",
        "FLAGGEMS_GDN_NATIVE_FAST_FORWARD": "1",
        "FLAGGEMS_Q4_RELEASE_SOURCE_WEIGHTS": "1",
        "FLAGGEMS_RELEASE_BF16_LM_HEAD": "1",
        "FLAGGEMS_GDN_CONV_PREFILL_TRITON": "1",
        "FLAGGEMS_GDN_PREFILL_DECAY_STORE": "1",
        "FLAGGEMS_Q4_FUSED_GDN_INPUT": "1",
        "FLAGGEMS_W8_STEALING_DECODE": "1",
        "FLAGGEMS_W8_STEAL_CHUNK": "64",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)

    homebrew_libomp = Path("/opt/homebrew/opt/libomp")
    if homebrew_libomp.is_dir():
        os.environ.setdefault("TRITON_LOCAL_LIBOMP_PATH", str(homebrew_libomp))


def _configure_darwin_gdn_safety() -> str:
    configured = os.getenv(
        "FL_CPU_QWEN_GDN_BACKEND",
        "torch" if platform.system() == "Darwin" else "triton",
    )
    backend = configured.strip().lower()
    if backend not in {"torch", "triton"}:
        raise ValueError(
            "FL_CPU_QWEN_GDN_BACKEND must be 'torch' or 'triton'"
        )

    unsafe_acknowledged = _enabled("FL_CPU_QWEN_ALLOW_UNSAFE_GDN")
    if (
        platform.system() == "Darwin"
        and backend == "triton"
        and not unsafe_acknowledged
    ):
        raise RuntimeError(
            "The Triton GDN path is quarantined on Darwin after a reproducible "
            "SoC watchdog reset. Use FL_CPU_QWEN_GDN_BACKEND=torch, or "
            "explicitly acknowledge the experimental path with "
            "FL_CPU_QWEN_ALLOW_UNSAFE_GDN=1."
        )

    if platform.system() == "Darwin":
        os.environ["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] = (
            "1" if _enabled("FLAGGEMS_GDN_NATIVE_PACKED_DECODE", "1") else "0"
        )
    return backend


def _gdn_label(gdn_backend: str, runtime: str | None) -> str:
    parts = [gdn_backend]
    if runtime != "libtriton_jit":
        return "+".join(parts)

    flags = (
        ("FLAGGEMS_GDN_NATIVE_DECODE", "native-decode"),
        ("FLAGGEMS_GDN_NATIVE_CONV", "native-conv"),
        ("FLAGGEMS_GDN_NATIVE_PREFILL", "native-prefill"),
        ("FLAGGEMS_GDN_CONV_PREFILL_TRITON", "vector-conv-prefill"),
        ("FLAGGEMS_GDN_PREFILL_DECAY_STORE", "decay-store-prefill"),
        ("FLAGGEMS_GDN_NATIVE_PACKED_DECODE", "native-packed-decode"),
        ("FLAGGEMS_GDN_NATIVE_NORM", "native-norm"),
        ("FLAGGEMS_GDN_NATIVE_FAST_FORWARD", "native-fast-forward"),
        ("FLAGGEMS_Q4_FUSED_GDN_INPUT", "fused-q4-input"),
    )
    parts.extend(label for name, label in flags if _enabled(name))
    return "+".join(parts)


def _print_runtime_banner(
    *, backend: str, runtime: str | None, gdn_backend: str
) -> None:
    if _enabled("FLAGGEMS_Q4_STEALING_DECODE"):
        q4_schedule = "stealing"
    elif _enabled("FLAGGEMS_Q4_WEIGHTED_DECODE"):
        q4_schedule = "weighted"
    else:
        q4_schedule = "static"
    released_sources = _enabled("FLAGGEMS_Q4_RELEASE_SOURCE_WEIGHTS") and (
        not _enabled("FL_INT8_LMHEAD")
        or _enabled("FLAGGEMS_RELEASE_BF16_LM_HEAD")
    )
    print(
        "[vllm_fl] ARM Qwen runtime active "
        f"(quant_backend={backend}, q4_runtime={runtime or 'vllm'}, "
        f"gdn={_gdn_label(gdn_backend, runtime)}, "
        f"q4_decode_schedule={q4_schedule}, "
        "w8_decode_schedule="
        f"{'stealing' if _enabled('FLAGGEMS_W8_STEALING_DECODE') else 'static'}, "
        "packed_gdn="
        f"{'on' if _enabled('FLAGGEMS_GDN_NATIVE_PACKED_DECODE') else 'off'}, "
        f"release_sources={'on' if released_sources else 'off'})",
        flush=True,
    )


def enable_qwen_runtime(
    *, backend: str, verbose: bool = True
) -> bool:
    """Install the former standalone Qwen plugin inside vllm-plugin-FL.

    ``tleraw`` keeps its existing linear implementation while installing the
    Darwin GDN safety layer. ``libtriton_jit`` additionally activates the
    FlagGems quantized router.
    """
    global _ACTIVE_BACKEND
    backend = resolve_int4_backend(backend)
    if _ACTIVE_BACKEND is not None:
        if _ACTIVE_BACKEND != backend:
            raise RuntimeError(
                "FL ARM Qwen runtime is already active with INT4 backend "
                f"{_ACTIVE_BACKEND}; cannot switch to {backend} in-process"
            )
        return False

    _configure_runtime_defaults()
    gdn_backend = _configure_darwin_gdn_safety()
    if gdn_backend == "torch":
        from vllm_fl.ops.cpu_qwen_gdn import install_vllm_gdn_fallback

        install_vllm_gdn_fallback()

    runtime = _BACKEND_TO_FLAGGEMS_RUNTIME[backend]
    if runtime is not None:
        from flag_gems.runtime.backend._arm.q4 import enable_vllm_q4_codegen

        enable_vllm_q4_codegen(verbose=verbose, runtime=runtime)

    _ACTIVE_BACKEND = backend
    if verbose:
        _print_runtime_banner(
            backend=backend,
            runtime=runtime,
            gdn_backend=gdn_backend,
        )
    return True


__all__ = [
    "enable_qwen_runtime",
    "resolve_int4_backend",
]
