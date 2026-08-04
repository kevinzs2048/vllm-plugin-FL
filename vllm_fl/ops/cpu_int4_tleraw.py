"""Offline channelwise ARM CPU W4A8 through FlagTree TLE-raw CPU ops.

Weights and per-channel scales come directly from a standard vLLM W4A8
checkpoint.  The compiled op accepts arbitrary M and selects KleidiAI dotprod
for decode (M=1) or i8mm for prefill (M>1) inside its C backing function.
"""

import ctypes
import logging
import platform

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra.cpu import kleidiai
from triton.language.extra.cpu.tle_ops import (
    kleidiai_w4a8_linear as gemm_w4a8_i8mm,
)
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    _POSSIBLE_KERNELS,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm.scalar_type import scalar_types
from vllm_fl.ops.cpu_quant_linear import (
    require_arm_quant_extensions,
)
from vllm_fl.ops.cpu_quant_tle import (
    register_inductor_builtin_import,
)
from vllm_fl.patches.dynamo_metrics import patch_dynamo_metrics_serialization

logger = logging.getLogger("vllm_fl.cpu_int4_tleraw")

require_arm_quant_extensions("FL ARM W4A8 TLE backend")
TLE_CACHE_ABI = kleidiai.runtime_abi("w4a8")
_PACK_LIBRARY = kleidiai.build_runtime("w4a8")
# The generated TLE kernel resolves FlagTree's stable W4A8 runtime symbol from
# this source-built, process-global library.
_PACK = ctypes.CDLL(str(_PACK_LIBRARY), mode=ctypes.RTLD_GLOBAL)
_PACK.flagtree_kai_w4a8_rhs_packed_size.restype = ctypes.c_size_t
_PACK.flagtree_kai_w4a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 2
_PACK.flagtree_kai_w4a8_pack_rhs.restype = None
_PACK.flagtree_kai_w4a8_pack_rhs.argtypes = [
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


def _pack_rhs(weight, scales):
    """Pack checkpoint-native signed INT4 values and channelwise scales."""
    if weight.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("ARM W4A8 packing requires CPU weights and scales")
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("ARM W4A8 weight must be a 2D torch.int8 tensor")
    N, K = weight.shape
    if K % 2:
        raise ValueError(f"ARM W4A8 requires an even K dimension, got {K}")
    if scales.numel() != N:
        raise ValueError(
            f"ARM W4A8 requires one weight scale per row, got {scales.shape}"
        )
    if torch.any(weight < -8) or torch.any(weight > 7):
        raise ValueError("ARM W4A8 checkpoint weights must be in [-8, 7]")

    unsigned = weight.detach().add(8).to(torch.uint8)
    native = (
        unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    ).contiguous()
    scales_f32 = scales.detach().reshape(-1).to(torch.float32).contiguous()
    packed = torch.empty(
        _PACK.flagtree_kai_w4a8_rhs_packed_size(N, K), dtype=torch.uint8
    )
    _PACK.flagtree_kai_w4a8_pack_rhs(
        N,
        K,
        _pointer(native),
        _pointer(scales_f32),
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


class FlagTreeKleidiAIInt4LinearKernel(MPLinearKernel):
    """Direct KAI kernel for offline channelwise W4A8 checkpoints."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 1

    @classmethod
    def can_implement(
        cls, config: MPLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU"
        if platform.machine().lower() not in {"aarch64", "arm64"}:
            return False, "requires AArch64"
        if config.weight_type != scalar_types.int4:
            return False, "requires signed INT4 checkpoint weights"
        if config.act_type != torch.bfloat16:
            return False, "requires BF16 activations"
        if config.zero_points or config.has_g_idx:
            return False, "requires symmetric weights without activation ordering"
        if config.group_size != config.partition_weight_shape[0]:
            return False, "requires channelwise weight scales"
        if config.partition_weight_shape[0] % 2:
            return False, "requires an even input dimension"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = getattr(layer, self.w_q_name)
        scales = getattr(layer, self.w_s_name)
        N, K = weight.shape
        packed = _pack_rhs(weight, scales)
        layer.register_buffer("_fl_w4a8_packed_rhs", packed, persistent=False)
        self.N = N
        self.K = K

        # The packed buffer is the runtime representation; release checkpoint
        # tensors just as vLLM's native CPU kernel does after its own repack.
        setattr(layer, self.w_q_name, None)
        setattr(layer, self.w_s_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = linear_w4a8(x, layer._fl_w4a8_packed_rhs, self.N, self.K)
        return out + bias.to(out.dtype) if bias is not None else out


def register_checkpoint_int4_kernel() -> bool:
    """Prioritize direct KAI for compatible standard W4A8 checkpoints."""
    candidates = _POSSIBLE_KERNELS.setdefault(PlatformEnum.CPU, [])
    if FlagTreeKleidiAIInt4LinearKernel in candidates:
        return False
    candidates.insert(0, FlagTreeKleidiAIInt4LinearKernel)
    return True


def enable_int4(verbose=True):
    _register_tle_w4a8()
    registered = register_checkpoint_int4_kernel()
    if registered and verbose:
        logger.info(
            "[vllm_fl] offline channelwise W4A8 enabled "
            "(FlagTree TLE-raw; decode=dotprod, prefill=i8mm)"
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
