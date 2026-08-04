"""ARM CPU W8A8 Linear through FlagTree's direct KleidiAI TLE op."""

import ctypes
import logging
import os
import platform

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra.cpu import kleidiai
from triton.language.extra.cpu.tle_ops import (
    kleidiai_w8a8_linear as gemm_w8a8_tle,
)
from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    _POSSIBLE_INT8_KERNELS,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    Int8ScaledMMLinearLayerConfig,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm_fl.ops.cpu_quant_linear import (
    install_cpu_quantized_linear,
    require_arm_quant_extensions,
)
from vllm_fl.ops.cpu_quant_tle import (
    register_inductor_builtin_import,
)
from vllm_fl.patches.dynamo_metrics import patch_dynamo_metrics_serialization

logger = logging.getLogger("vllm_fl.cpu_int8_tleraw")
INCLUDE_LM_HEAD = os.environ.get("FL_INT8_LMHEAD", "0") == "1"
STRICT = os.environ.get("FL_CPU_INT8_STRICT", "1") != "0"

require_arm_quant_extensions("FL ARM W8A8 TLE backend")
TLE_CACHE_ABI = kleidiai.runtime_abi("w8a8")
_PACK_LIBRARY = kleidiai.build_runtime("w8a8")
# The generated TLE kernel resolves FlagTree's stable W8A8 runtime symbol from
# this source-built, process-global library.
_PACK = ctypes.CDLL(str(_PACK_LIBRARY), mode=ctypes.RTLD_GLOBAL)
_PACK.flagtree_kai_w8a8_rhs_packed_size.restype = ctypes.c_size_t
_PACK.flagtree_kai_w8a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 2
_PACK.flagtree_kai_w8a8_pack_rhs.restype = None
_PACK.flagtree_kai_w8a8_pack_rhs.argtypes = (
    [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 3
)


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _quantize_pack(weight):
    """Online fallback: [N,K] BF16 -> packed channelwise symmetric INT8."""
    N, K = weight.shape
    w = weight.detach().to(torch.float32)
    scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    qw = (w / scale[:, None]).round().clamp(-128, 127).to(torch.int8).contiguous()
    return _pack_rhs(qw, scale)


def _pack_rhs(weight, scale):
    """Pack checkpoint-native INT8 weights without requantizing them."""
    if weight.device.type != "cpu" or scale.device.type != "cpu":
        raise ValueError("ARM W8A8 packing requires CPU weights and scales")
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise ValueError("ARM W8A8 weight must be a 2D torch.int8 tensor")
    N, K = weight.shape
    if scale.numel() != N:
        raise ValueError(
            f"ARM W8A8 requires one weight scale per row, got {scale.shape}"
        )

    weight_i8 = weight.detach().contiguous()
    scale_f32 = scale.detach().reshape(-1).to(torch.float32).contiguous()
    packed = torch.empty(
        _PACK.flagtree_kai_w8a8_rhs_packed_size(N, K), dtype=torch.uint8
    )
    _PACK.flagtree_kai_w8a8_pack_rhs(
        N, K, _ptr(weight_i8), _ptr(scale_f32), _ptr(packed)
    )
    return packed


def _register_tle_w8a8() -> None:
    """Verify that the W8A8 TLE op is running on Triton CPU."""
    if triton.runtime.driver.active.get_current_target().backend != "cpu":
        raise RuntimeError("FL ARM int8 TLE backend requires triton-cpu")


_BUILTIN_IMPORT = "from vllm_fl.ops.cpu_int8_tleraw import gemm_w8a8_tle\n"


patch_dynamo_metrics_serialization()
register_inductor_builtin_import("gemm_w8a8_tle(", _BUILTIN_IMPORT)


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
    _register_tle_w8a8()
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


def _prepare_linear(weight):
    N, K = weight.shape
    return _make_cpu_linear(_quantize_pack(weight), N, K)


class FlagTreeKleidiAIInt8LinearKernel(Int8ScaledMMLinearKernel):
    """Direct KAI pack/compute for dynamic-symmetric channelwise W8A8."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cpu():
            return False, "requires CPU"
        if platform.machine().lower() not in {"aarch64", "arm64"}:
            return False, "requires AArch64"
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not config.is_channelwise:
            return False, "requires channelwise weight scales"
        if config.is_static_input_scheme:
            return False, "requires dynamic activation quantization"
        if not config.input_symmetric:
            return False, "requires symmetric activation quantization"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_name, scale_name, _, _, _ = self.layer_param_names
        weight = getattr(layer, weight_name)
        scale = getattr(layer, scale_name)
        N, K = weight.shape
        packed = _pack_rhs(weight, scale)
        layer.register_buffer("_fl_w8a8_packed_rhs", packed, persistent=False)
        self.N = N
        self.K = K

        # Loading is complete and the packed buffer owns all required data.
        # Match vLLM's oneDNN kernel by releasing the checkpoint matrix.
        setattr(layer, weight_name, None)
        setattr(layer, scale_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = linear_w8a8(x, layer._fl_w8a8_packed_rhs, self.N, self.K)
        return out + bias.to(out.dtype) if bias is not None else out


def register_checkpoint_int8_kernel() -> bool:
    """Prioritize direct KAI for compatible standard W8A8 checkpoints."""
    candidates = _POSSIBLE_INT8_KERNELS.setdefault(PlatformEnum.CPU, [])
    if FlagTreeKleidiAIInt8LinearKernel in candidates:
        return False
    candidates.insert(0, FlagTreeKleidiAIInt8LinearKernel)
    return True


def enable_int8(verbose=True):
    source = os.environ.get("FL_CPU_INT8_SOURCE", "auto").lower()
    if source not in {"auto", "checkpoint", "online"}:
        raise ValueError(
            "FL_CPU_INT8_SOURCE must be 'auto', 'checkpoint' or 'online'"
        )

    _register_tle_w8a8()
    checkpoint_registered = False
    online_installed = False
    if source in {"auto", "checkpoint"}:
        checkpoint_registered = register_checkpoint_int8_kernel()
    if source in {"auto", "online"}:
        online_installed = install_cpu_quantized_linear(
            backend="ARM W8A8 TLE",
            prepare_linear=_prepare_linear,
            supports_shape=lambda n, k: k % 8 == 0,
            include_lm_head=INCLUDE_LM_HEAD,
            strict=STRICT,
            logger=logger,
        )

    if (checkpoint_registered or online_installed) and verbose:
        logger.info(
            "[vllm_fl] ARM W8A8 enabled (source=%s, FlagTree TLE-raw op, "
            "KleidiAI backing)",
            source,
        )
