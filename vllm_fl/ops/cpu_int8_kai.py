"""ARM CPU W8A8 Linear via KleidiAI qai8dxp x qsi8cxp NEON ukernels.

Per-row symmetric int8 weights (packed once at load, qsi8cxp4x8) with dynamic
per-row int8 activations (qai8dxp, packed inside the C wrapper each call).
Decode (M=1) runs the 1x4 NEON dotprod GEMV; prefill (M>1) the 16x4 NEON i8mm
GEMM.  The hand-tuned kernels keep packed int8 weights in their native layout
and avoid materializing a dequantized weight tensor on the inference path.

The ctypes call is wrapped in an opaque torch.library.custom_op so vLLM's
DYNAMO_TRACE_ONCE + Inductor graph treats it as a black box (the proven
compile-safe pattern from the int4 backend).  Weights quantized online from
the original BF16 checkpoint — no torchao checkpoint needed.
"""
import ctypes
import logging
import os
import pathlib

import torch

from vllm_fl.ops.cpu_quant_linear import (
    install_cpu_quantized_linear,
    require_arm_quant_extensions,
)

logger = logging.getLogger("vllm_fl.cpu_int8_kai")
INCLUDE_LM_HEAD = os.environ.get("FL_INT8_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT8_STRICT", "1") != "0"

_HERE = pathlib.Path(__file__).resolve().parent
_LIB_PATH = _HERE / "libkai_w8a8.so"
require_arm_quant_extensions("FL ARM W8A8 KleidiAI backend")
if not _LIB_PATH.is_file():
    raise FileNotFoundError(
        f"missing {_LIB_PATH}; run tools/build_arm_int8_assets.sh"
    )
_LIB = ctypes.CDLL(str(_LIB_PATH))
_LIB.fl_w8a8_rhs_packed_size.restype = ctypes.c_size_t
_LIB.fl_w8a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 2
_LIB.fl_w8a8_pack_rhs.argtypes = [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 3
_LIB.fl_w8a8_linear.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 3


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _quantize_pack(weight):
    """[N,K] bf16 -> KleidiAI qsi8cxp packed rhs (uint8 blob)."""
    N, K = weight.shape
    w = weight.detach().to(torch.float32)
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    qw = (w / scale[:, None]).round().clamp(-128, 127).to(torch.int8).contiguous()
    scale_f32 = scale.to(torch.float32).contiguous()
    packed = torch.empty(
        _LIB.fl_w8a8_rhs_packed_size(N, K), dtype=torch.uint8
    )
    _LIB.fl_w8a8_pack_rhs(N, K, _ptr(qw), _ptr(scale_f32), _ptr(packed))
    return packed


@torch.library.custom_op("fl_cpu::linear_w8a8", mutates_args=())
def linear_w8a8(x: torch.Tensor, packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    x_bf16 = x.to(torch.bfloat16).reshape(-1, K).contiguous()
    M = x_bf16.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16)
    _LIB.fl_w8a8_linear(M, N, K, _ptr(x_bf16), _ptr(packed), _ptr(out))
    return out.reshape(*x.shape[:-1], N)


@linear_w8a8.register_fake
def _(x, packed, N, K):
    return x.new_empty((*x.shape[:-1], N), dtype=torch.bfloat16)


def _make_cpu_linear(packed, N, K):
    def cpu_linear(x, weight, bias):
        out = torch.ops.fl_cpu.linear_w8a8(x, packed, N, K)
        return out + bias.to(out.dtype) if bias is not None else out

    return cpu_linear


def _prepare_linear(weight):
    N, K = weight.shape
    return _make_cpu_linear(_quantize_pack(weight), N, K)


def enable_int8(verbose=True):
    installed = install_cpu_quantized_linear(
        backend="ARM W8A8 KleidiAI",
        prepare_linear=_prepare_linear,
        supports_shape=lambda n, k: k % 8 == 0,
        include_lm_head=INCLUDE_LM_HEAD,
        strict=STRICT,
        logger=logger,
    )
    if installed and verbose:
        logger.info("[vllm_fl] ARM W8A8 enabled (KleidiAI dotprod/i8mm)")
