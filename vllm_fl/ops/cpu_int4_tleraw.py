"""ARM CPU W4A8 Linear through FlagTree TLE-raw CPU ops.

The compiled TLE op accepts arbitrary M and selects KleidiAI dotprod for
decode (M=1) or i8mm for prefill (M>1) inside its C backing function.  Shape
dispatch must not happen in Python: vLLM CPU uses DYNAMO_TRACE_ONCE and drops
the warmup graph's shape guards before reusing it for decode.
"""

import ctypes
import logging
import os
import pathlib

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.core import _unwrap_if_constexpr, builtin
from triton.language.extra.cpu import neon
from vllm_fl.ops.cpu_quant_linear import (
    install_cpu_quantized_linear,
    native_asset_cache_abi,
    require_arm_quant_extensions,
)
from vllm_fl.ops.cpu_quant_tle import (
    ensure_tle_backend,
    register_inductor_builtin_import,
)
from vllm_fl.patches.dynamo_metrics import patch_dynamo_metrics_serialization

logger = logging.getLogger("vllm_fl.cpu_int4_tleraw")
INCLUDE_LM_HEAD = os.environ.get("FL_INT4_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT4_STRICT", "1") != "0"
BL = 32

_HERE = pathlib.Path(__file__).resolve().parent
# Our own sources; the KleidiAI microkernels they call are built separately.
_WRAPPER_SOURCE = _HERE / "cpu_int4_tle_wrapper.c"
_PACK_SOURCE = _HERE / "cpu_int4_pack.c"
_TLE_SYMBOL = "sdot_gemm_q4_0_v2_smmla_bf16"
_REGISTERED = False

# KleidiAI is not vendored.  Build its microkernel object with FlagTree's
# python/scripts/build_kai_w4a8_assets.sh, then point these at the result.
_BUILD_HINT = (
    "Build the KleidiAI assets first:\n"
    "    bash python/scripts/build_kai_w4a8_assets.sh   # in a FlagTree checkout\n"
    "then export the variables it prints (FL_KAI_W4A8_DIR, KLEIDIAI_ROOT).\n"
    "See examples/minicpm/arm_cpu_int4.md."
)
_KAI_DIR = os.environ.get("FL_KAI_W4A8_DIR", "")
_KLEIDIAI_ROOT = os.environ.get("KLEIDIAI_ROOT", "")


def _ukernel_object() -> pathlib.Path:
    """Path to libkai_w4a8_ukernels.o, linked into every compiled TLE kernel."""
    if not _KAI_DIR:
        raise RuntimeError(f"FL_KAI_W4A8_DIR is not set.\n{_BUILD_HINT}")
    path = pathlib.Path(_KAI_DIR) / "libkai_w4a8_ukernels.o"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.\n{_BUILD_HINT}")
    return path


def _build_pack_library() -> pathlib.Path:
    """Compile cpu_int4_pack.c against the KleidiAI headers, cached by content.

    Loaded through ctypes at model-load time to pack weights into the KleidiAI
    qsi4c32p layout; it is not on the inference path.
    """
    import hashlib
    import shutil
    import subprocess
    import tempfile

    override = os.environ.get("FL_KAI_W4A8_PACK_SO")
    if override:
        path = pathlib.Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"FL_KAI_W4A8_PACK_SO={path} does not exist")
        return path

    ukernel = _UKERNEL_OBJECT
    if not _KLEIDIAI_ROOT:
        raise RuntimeError(
            f"KLEIDIAI_ROOT is not set (needed for the KleidiAI headers).\n{_BUILD_HINT}"
        )
    include_dir = pathlib.Path(_KLEIDIAI_ROOT)
    if not (include_dir / "kai").is_dir():
        raise FileNotFoundError(
            f"KLEIDIAI_ROOT={include_dir} does not look like a KleidiAI root (no kai/)"
        )

    digest = hashlib.sha256(
        _PACK_SOURCE.read_bytes() + ukernel.read_bytes()
    ).hexdigest()[:16]
    cache_dir = pathlib.Path(
        os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
    ) / "flagos-kai-w4a8"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"libkai_w4a8_pack_{digest}.so"
    if out.is_file():
        return out

    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("No C compiler found; set CC")
    # Build to a temporary path first so concurrent workers cannot observe a
    # half-written .so at the cache location.
    with tempfile.NamedTemporaryFile(suffix=".so", dir=cache_dir, delete=False) as tmp:
        tmp_path = pathlib.Path(tmp.name)
    try:
        subprocess.check_call([
            cc, "-O3", "-fPIC", "-shared",
            "-march=armv8.6-a+bf16+i8mm+dotprod",
            f"-I{include_dir}",
            str(_PACK_SOURCE), str(ukernel),
            "-o", str(tmp_path),
        ])
        tmp_path.replace(out)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("[vllm_fl] built KleidiAI pack library: %s", out)
    return out


require_arm_quant_extensions("FL ARM W4A8 TLE backend")
_UKERNEL_OBJECT = _ukernel_object()
TLE_CACHE_ABI = native_asset_cache_abi(_WRAPPER_SOURCE, _UKERNEL_OBJECT)
_PACK_LIBRARY = _build_pack_library()
# RTLD_GLOBAL: the compiled TLE kernel resolves fl_w4a8_profile_record_v2 here
# when FL_W4A8_PROFILE is enabled.
_PACK = ctypes.CDLL(str(_PACK_LIBRARY), mode=ctypes.RTLD_GLOBAL)
_PACK.fl_w4a8_rhs_packed_size.restype = ctypes.c_size_t
_PACK.fl_w4a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 3
_PACK.fl_w4a8_pack_rhs.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_PACK.fl_w4a8_profile_count.restype = ctypes.c_size_t
_PACK.fl_w4a8_profile_get.argtypes = [
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
]
_PACK.fl_w4a8_profile_get.restype = ctypes.c_int


def _pointer(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def quant_native_qs4c32(weight, block_length=BL):
    """Quantize [N,K] BF16 weights to native signed-int4 blocks."""
    N, K = weight.shape
    if K % 2 or K % block_length:
        raise ValueError(
            f"W4A8 requires K divisible by {block_length}; got {(N, K)}"
        )
    blocks = weight.detach().float().reshape(N, K // block_length, block_length)
    max_indices = blocks.abs().argmax(dim=-1, keepdim=True)
    signed_max = torch.gather(blocks, -1, max_indices)
    scales = signed_max / -8.0
    reciprocal = torch.where(
        scales != 0, 1.0 / scales, torch.zeros_like(scales)
    )
    quantized = (blocks * reciprocal).round().clamp_(-8, 7).to(torch.int32)
    unsigned = (quantized + 8).to(torch.uint8).reshape(N, K)
    native = (
        unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    ).contiguous()
    bf16_scales = scales.squeeze(-1).to(torch.bfloat16).contiguous()
    return native, bf16_scales


def _pack_rhs(native, scales, N, K):
    packed = torch.empty(
        _PACK.fl_w4a8_rhs_packed_size(N, K, BL), dtype=torch.uint8
    )
    _PACK.fl_w4a8_pack_rhs(
        N,
        K,
        BL,
        _pointer(native),
        _pointer(scales),
        _pointer(packed),
    )
    return packed


def _register_tle_w4a8() -> None:
    """Register the TLE backing symbol and link its KleidiAI ukernels."""
    global _REGISTERED
    if _REGISTERED:
        return

    def register() -> None:
        if triton.runtime.driver.active.get_current_target().backend != "cpu":
            raise RuntimeError("FL ARM int4 TLE backend requires triton-cpu")
        if not _WRAPPER_SOURCE.is_file():
            raise FileNotFoundError(
                f"Required ARM int4 TLE source is missing: {_WRAPPER_SOURCE}"
            )
        ukernel_object = str(_UKERNEL_OBJECT)
        neon.register_c_function(
            _TLE_SYMBOL,
            _WRAPPER_SOURCE.read_text(encoding="utf-8"),
            extra_cflags=["-funroll-loops"],
        )
        original_get_objects = neon.get_all_object_files

        def get_all_object_files():
            objects = original_get_objects()
            if ukernel_object not in objects:
                objects.append(ukernel_object)
            return objects

        neon.get_all_object_files = get_all_object_files

    ensure_tle_backend("w4a8", register)
    _REGISTERED = True


def _as_i64(value, builder):
    value = _unwrap_if_constexpr(value)
    return value.handle if hasattr(value, "handle") else builder.get_int64(value)


@builtin
def gemm_w4a8_i8mm(
    x_ptr,
    rhs_ptr,
    out_ptr,
    M,
    K,
    N,
    _builder=None,
):
    """Emit FlagTree's CPU W4A8 GEMM op with a runtime M dimension."""
    _builder.create_cpu_gemm_q4_0_v2_smmla_bf16(
        x_ptr.handle,
        rhs_ptr.handle,
        out_ptr.handle,
        _as_i64(M, _builder),
        _as_i64(K, _builder),
        _as_i64(N, _builder),
    )


_BUILTIN_IMPORT = (
    "from vllm_fl.ops.cpu_int4_tleraw import gemm_w4a8_i8mm\n"
)


patch_dynamo_metrics_serialization()
register_inductor_builtin_import("gemm_w4a8_i8mm(", _BUILTIN_IMPORT)


@triton.jit
def _linear_kernel(
    x_ptr,
    rhs_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    CACHE_ABI: tl.constexpr,
):
    M_i64 = tl.full((), M, tl.int64)
    gemm_w4a8_i8mm(x_ptr, rhs_ptr, out_ptr, M_i64, K, N)
    # Inductor's mutation analysis does not know the custom CPU op.  The
    # dependent store makes the output mutation explicit without changing it.
    value = tl.load(out_ptr)
    tl.store(out_ptr, value)


@triton_op("fltleraw::linear_w4a8", mutates_args={})
def linear_w4a8(
    x: torch.Tensor,
    rhs: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
    _register_tle_w4a8()
    x_bf16 = x.to(torch.bfloat16).reshape(-1, K).contiguous()
    M = x_bf16.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    wrap_triton(_linear_kernel)[(1,)](
        x_bf16,
        rhs,
        out,
        M,
        K=K,
        N=N,
        CACHE_ABI=TLE_CACHE_ABI,
        num_warps=1,
        num_stages=1,
    )
    return out.reshape(*x.shape[:-1], N)


@linear_w4a8.register_fake
def _(x, rhs, N, K):
    return x.new_empty((*x.shape[:-1], N), dtype=torch.bfloat16)


def _make_cpu_linear(rhs, N, K):
    def cpu_linear(x, weight, bias):
        out = torch.ops.fltleraw.linear_w4a8(x, rhs, N, K)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def _prepare_linear(weight):
    N, K = weight.shape
    native, scales = quant_native_qs4c32(
        weight.detach().to(torch.bfloat16), BL
    )
    return _make_cpu_linear(_pack_rhs(native, scales, N, K), N, K)


def enable_int4(verbose=True):
    installed = install_cpu_quantized_linear(
        backend="ARM W4A8 TLE",
        prepare_linear=_prepare_linear,
        supports_shape=lambda n, k: k % BL == 0 and n % 8 == 0,
        include_lm_head=INCLUDE_LM_HEAD,
        strict=STRICT,
        logger=logger,
        initialize=_register_tle_w4a8,
    )
    if installed and verbose:
        logger.info(
            "[vllm_fl] CPU int4 TLE-raw enabled "
            "(decode=dotprod, prefill=i8mm)"
        )


def profile_reset():
    _PACK.fl_w4a8_profile_reset()


def profile_stats():
    result = []
    for index in range(_PACK.fl_w4a8_profile_count()):
        values = (ctypes.c_uint64 * 6)()
        if not _PACK.fl_w4a8_profile_get(index, values):
            continue
        m, n, k, calls, lhs_pack_ns, gemv_ns = map(int, values)
        result.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "calls": calls,
                "lhs_pack_ms": lhs_pack_ns / 1e6,
                "gemv_ms": gemv_ns / 1e6,
            }
        )
    return result
