"""ARM CPU W8A8 Linear through the FlagTree TLE-raw CPU op.

Mirrors the int4 TLE-raw integration: the compiled TLE op
(`create_cpu_gemm_q4_0_v2_smmla_bf16`) passes raw pointers plus m/k/n to a
fixed extern symbol, and we back that symbol with a W8A8 implementation
(KleidiAI qai8dxp x qsi8cxp dotprod GEMV / i8mm GEMM) registered via
`neon.register_c_function`.  Shape dispatch (decode vs prefill) happens inside
the C backing function because vLLM CPU uses DYNAMO_TRACE_ONCE and drops the
warmup graph's shape guards before reusing it for decode.

Weights are quantized online (per-row symmetric int8) and packed to
qsi8cxp4x8 via ctypes (libkai_w8a8.so) at load time; runtime compute goes
through the TLE op inside a @triton.jit kernel wrapped with
torch.library.triton_op for Inductor.
"""

import ctypes
import logging
import os
import pathlib
import platform

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.core import _unwrap_if_constexpr, builtin
from triton.language.extra.cpu import neon

logger = logging.getLogger("vllm_fl.cpu_int8_tleraw")
STATS = {"int8_linears": 0}
INCLUDE_LM_HEAD = os.environ.get("FL_INT8_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT8_STRICT", "1") != "0"
TLE_CACHE_ABI = 20260720

_HERE = pathlib.Path(__file__).resolve().parent
_WRAPPER_SOURCE = _HERE / "cpu_int8_tle_wrapper.c"
_UKERNEL_OBJECT = _HERE / "libkai_w8a8_ukernels.o"
_PACK_LIBRARY = _HERE / "libkai_w8a8.so"
_TLE_SYMBOL = "sdot_gemm_q4_0_v2_smmla_bf16"
_REGISTERED = False


def _validate_cpu_features() -> None:
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("FL ARM int8 TLE backend requires AArch64")
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return
    feature_sets = []
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("features"):
            feature_sets.append(set(line.partition(":")[2].split()))
    features = set.intersection(*feature_sets) if feature_sets else set()
    required = {"asimddp", "i8mm", "bf16"}
    missing = required - features
    if missing:
        raise RuntimeError(
            "FL ARM int8 requires dotprod, i8mm, and BF16 CPU extensions; "
            f"missing: {', '.join(sorted(missing))}"
        )


_validate_cpu_features()

for _path in (_WRAPPER_SOURCE, _UKERNEL_OBJECT, _PACK_LIBRARY):
    if not _path.is_file():
        raise FileNotFoundError(
            f"Required ARM int8 TLE asset is missing: {_path}; "
            "run tools/build_arm_int8_assets.sh"
        )

_PACK = ctypes.CDLL(str(_PACK_LIBRARY))
_PACK.fl_w8a8_rhs_packed_size.restype = ctypes.c_size_t
_PACK.fl_w8a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 2
_PACK.fl_w8a8_pack_rhs.argtypes = [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 3


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _quantize_pack(weight):
    """[N,K] bf16 -> KleidiAI qsi8cxp packed rhs blob (uint8)."""
    N, K = weight.shape
    w = weight.detach().to(torch.float32)
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    qw = (w / scale[:, None]).round().clamp(-128, 127).to(torch.int8).contiguous()
    scale_f32 = scale.to(torch.float32).contiguous()
    packed = torch.empty(_PACK.fl_w8a8_rhs_packed_size(N, K), dtype=torch.uint8)
    _PACK.fl_w8a8_pack_rhs(N, K, _ptr(qw), _ptr(scale_f32), _ptr(packed))
    return packed


def _register_tle_w8a8() -> None:
    """Register the W8A8 backing symbol and link its KleidiAI ukernels."""
    global _REGISTERED
    if _REGISTERED:
        return
    if triton.runtime.driver.active.get_current_target().backend != "cpu":
        raise RuntimeError("FL ARM int8 TLE backend requires triton-cpu")

    neon.register_c_function(
        _TLE_SYMBOL,
        _WRAPPER_SOURCE.read_text(encoding="utf-8"),
        extra_cflags=["-funroll-loops"],
    )
    original_get_objects = neon.get_all_object_files

    def get_all_object_files():
        objects = original_get_objects()
        object_path = str(_UKERNEL_OBJECT)
        if object_path not in objects:
            objects.append(object_path)
        return objects

    neon.get_all_object_files = get_all_object_files
    _REGISTERED = True


def _as_i64(value, builder):
    value = _unwrap_if_constexpr(value)
    return value.handle if hasattr(value, "handle") else builder.get_int64(value)


@builtin
def gemm_w8a8_tle(
    x_ptr,
    rhs_ptr,
    out_ptr,
    M,
    K,
    N,
    _builder=None,
):
    """Emit FlagTree's CPU GEMM op; the W8A8 backing symbol interprets rhs."""
    _builder.create_cpu_gemm_q4_0_v2_smmla_bf16(
        x_ptr.handle,
        rhs_ptr.handle,
        out_ptr.handle,
        _as_i64(M, _builder),
        _as_i64(K, _builder),
        _as_i64(N, _builder),
    )


_register_tle_w8a8()


_BUILTIN_IMPORT = "from vllm_fl.ops.cpu_int8_tleraw import gemm_w8a8_tle\n"


def _patch_inductor_builtin_import() -> None:
    """Teach Inductor-generated Triton source about the custom TLE builtin."""
    import torch._inductor.async_compile as async_compile

    if getattr(async_compile.AsyncCompile, "_fl_w8a8_builtin_patched", False):
        return
    original_triton = async_compile.AsyncCompile.triton

    def compile_triton(self, kernel_name, source_code, device_str="cpu"):
        if (
            "gemm_w8a8_tle(" in source_code
            and _BUILTIN_IMPORT.strip() not in source_code
        ):
            source_code = _BUILTIN_IMPORT + source_code
        return original_triton(self, kernel_name, source_code, device_str)

    async_compile.AsyncCompile.triton = compile_triton
    async_compile.AsyncCompile._fl_w8a8_builtin_patched = True


from vllm_fl.patches.dynamo_metrics import patch_dynamo_metrics_serialization

patch_dynamo_metrics_serialization()
_patch_inductor_builtin_import()


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
    gemm_w8a8_tle(x_ptr, rhs_ptr, out_ptr, M_i64, K, N)
    # Inductor's mutation analysis does not know the custom CPU op.  The
    # dependent store makes the output mutation explicit without changing it.
    value = tl.load(out_ptr)
    tl.store(out_ptr, value)


@triton_op("flint8tle::linear_w8a8", mutates_args={})
def linear_w8a8(
    x: torch.Tensor,
    rhs: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
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


@linear_w8a8.register_fake
def _(x, rhs, N, K):
    return x.new_empty((*x.shape[:-1], N), dtype=torch.bfloat16)


def _make_cpu_linear(rhs, N, K):
    def cpu_linear(x, weight, bias):
        out = torch.ops.flint8tle.linear_w8a8(x, rhs, N, K)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def enable_int8(verbose=True):
    import vllm.model_executor.layers.utils as layer_utils
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if getattr(layer_utils, "_fl_int8tle_enabled", False):
        return
    original_dispatch = layer_utils.dispatch_cpu_unquantized_gemm

    def dispatch(layer, remove_weight):
        weight = getattr(layer, "weight", None)
        prefix = getattr(layer, "prefix", "") or ""
        is_lm_head = isinstance(layer, ParallelLMHead)
        is_input_embedding = type(layer) is VocabParallelEmbedding
        if (
            weight is not None
            and weight.ndim == 2
            and weight.shape[1] % 8 == 0
            and not is_input_embedding
            and (INCLUDE_LM_HEAD or not is_lm_head)
        ):
            try:
                N, K = weight.shape
                packed = _quantize_pack(weight)
                layer.cpu_linear = _make_cpu_linear(packed, N, K)
                if remove_weight:
                    layer.weight = torch.nn.Parameter(
                        torch.empty(0), requires_grad=False
                    )
                STATS["int8_linears"] += 1
                try:
                    with open("/tmp/fl_int8tle_marker.txt", "w") as f:
                        f.write(f"int8_linears={STATS['int8_linears']}\n")
                except OSError:
                    pass
                return
            except Exception as exc:
                message = (
                    f"failed to prepare ARM W8A8 TLE weight {prefix} "
                    f"{tuple(weight.shape)}"
                )
                if STRICT:
                    raise RuntimeError(message) from exc
                logger.warning("%s; falling back to BF16: %s", message, exc)
        return original_dispatch(layer, remove_weight)

    layer_utils.dispatch_cpu_unquantized_gemm = dispatch
    layer_utils._fl_int8tle_enabled = True
    if verbose:
        logger.info(
            "[vllm_fl] ARM W8A8 enabled (FlagTree TLE-raw op, KleidiAI backing)"
        )


def stats():
    return dict(STATS)
