"""ARM CPU W4A8 Linear through FlagTree TLE-raw CPU ops.

The compiled TLE op accepts arbitrary M and selects KleidiAI dotprod for
decode (M=1) or i8mm for prefill (M>1) inside its C backing function.  Shape
dispatch must not happen in Python: vLLM CPU uses DYNAMO_TRACE_ONCE and drops
the warmup graph's shape guards before reusing it for decode.
"""

import ctypes
import logging
import os

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra.cpu import kleidiai
from triton.language.extra.cpu.tle_ops import (
    kleidiai_w4a8_linear as gemm_w4a8_i8mm,
)
from vllm_fl.ops.cpu_quant_linear import (
    install_cpu_quantized_linear,
    require_arm_quant_extensions,
)
from vllm_fl.ops.cpu_quant_tle import (
    register_inductor_builtin_import,
)
from vllm_fl.patches.dynamo_metrics import patch_dynamo_metrics_serialization

logger = logging.getLogger("vllm_fl.cpu_int4_tleraw")
INCLUDE_LM_HEAD = os.environ.get("FL_INT4_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT4_STRICT", "1") != "0"
BL = 32

require_arm_quant_extensions("FL ARM W4A8 TLE backend")
TLE_CACHE_ABI = kleidiai.runtime_abi("w4a8")
_PACK_LIBRARY = kleidiai.build_runtime("w4a8")
# The generated TLE kernel resolves FlagTree's stable W4A8 runtime symbol from
# this source-built, process-global library.
_PACK = ctypes.CDLL(str(_PACK_LIBRARY), mode=ctypes.RTLD_GLOBAL)
_PACK.flagtree_kai_w4a8_rhs_packed_size.restype = ctypes.c_size_t
_PACK.flagtree_kai_w4a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 3
_PACK.flagtree_kai_w4a8_pack_rhs.restype = None
_PACK.flagtree_kai_w4a8_pack_rhs.argtypes = [
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_PACK.flagtree_kai_w4a8_profile_reset.restype = None
_PACK.flagtree_kai_w4a8_profile_count.restype = ctypes.c_size_t
_PACK.flagtree_kai_w4a8_profile_get.argtypes = [
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
]
_PACK.flagtree_kai_w4a8_profile_get.restype = ctypes.c_int


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
        _PACK.flagtree_kai_w4a8_rhs_packed_size(N, K, BL), dtype=torch.uint8
    )
    _PACK.flagtree_kai_w4a8_pack_rhs(
        N,
        K,
        BL,
        _pointer(native),
        _pointer(scales),
        _pointer(packed),
    )
    return packed


def _register_tle_w4a8() -> None:
    """Verify that the W4A8 TLE op is running on Triton CPU."""
    if triton.runtime.driver.active.get_current_target().backend != "cpu":
        raise RuntimeError("FL ARM int4 TLE backend requires triton-cpu")


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
    _PACK.flagtree_kai_w4a8_profile_reset()


def profile_stats():
    result = []
    for index in range(_PACK.flagtree_kai_w4a8_profile_count()):
        values = (ctypes.c_uint64 * 6)()
        if not _PACK.flagtree_kai_w4a8_profile_get(index, values):
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
