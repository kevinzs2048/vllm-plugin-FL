"""ARM CPU W8A8 Linear via KleidiAI qai8dxp x qsi8cxp NEON ukernels.

Per-row symmetric int8 weights (packed once at load, qsi8cxp4x8) with dynamic
per-row int8 activations (qai8dxp, packed inside the C wrapper each call).
Decode (M=1) runs the 1x4 NEON dotprod GEMV; prefill (M>1) the 16x4 NEON i8mm
GEMM.  ~2.7x faster than torch._weight_int8pack_mm on this SoC because the
hand-tuned asm streams the int8 weights at near memory bandwidth.

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

logger = logging.getLogger("vllm_fl.cpu_int8_kai")
STATS = {"int8_linears": 0}
INCLUDE_LM_HEAD = os.environ.get("FL_INT8_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT8_STRICT", "1") != "0"

_HERE = pathlib.Path(__file__).resolve().parent
_LIB_PATH = _HERE / "libkai_w8a8.so"
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


def enable_int8(verbose=True):
    import vllm.model_executor.layers.utils as layer_utils
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if getattr(layer_utils, "_fl_int8kai_enabled", False):
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
            and weight.shape[1] % 8 == 0  # kr alignment
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
                    with open("/tmp/fl_int8kai_marker.txt", "w") as f:
                        f.write(f"int8_linears={STATS['int8_linears']}\n")
                except OSError:
                    pass
                return
            except Exception as exc:
                message = (
                    f"failed to prepare ARM W8A8 weight {prefix} "
                    f"{tuple(weight.shape)}"
                )
                if STRICT:
                    raise RuntimeError(message) from exc
                logger.warning("%s; falling back to BF16: %s", message, exc)
        return original_dispatch(layer, remove_weight)

    layer_utils.dispatch_cpu_unquantized_gemm = dispatch
    layer_utils._fl_int8kai_enabled = True
    if verbose:
        logger.info("[vllm_fl] ARM W8A8 enabled (KleidiAI dotprod/i8mm)")


def stats():
    return dict(STATS)
