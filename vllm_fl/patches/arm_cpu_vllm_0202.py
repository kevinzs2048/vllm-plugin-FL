"""ARM CPU compatibility layer for the stock vLLM 0.20.2 package.

Keep model/runtime compatibility fixes in vllm-plugin-FL rather than carrying
a permanently modified vLLM source tree.  The integration intentionally fails
closed on a different vLLM version because it patches private 0.20.2 APIs.
"""

from __future__ import annotations

import gc
from importlib import metadata

import torch


_INSTALLED = False


def _require_vllm_0202() -> None:
    installed = metadata.version("vllm")
    base = installed.split("+", 1)[0]
    if base != "0.20.2":
        raise RuntimeError(
            "The FL ARM CPU compatibility layer requires vLLM 0.20.2; "
            f"found {installed}"
        )


def _install_packed_w4a8() -> None:
    from compressed_tensors.config import CompressionFormat
    from vllm.logger import init_logger
    from vllm.model_executor.kernels.linear import (
        MPLinearLayerConfig,
        choose_mp_linear_kernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as config_module,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a8_int as scheme_module,
    )
    from vllm.model_executor.parameter import (
        BasevLLMParameter,
        ChannelQuantScaleParameter,
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    scheme_cls = scheme_module.CompressedTensorsW4A8Int
    config_cls = config_module.CompressedTensorsConfig
    if getattr(scheme_cls, "_vllm_fl_packed_w4a8", False):
        return

    logger = init_logger(__name__)
    original_init = scheme_cls.__init__
    original_create_weights = scheme_cls.create_weights
    original_get_scheme = config_cls._get_scheme_from_parts

    def scheme_init(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
        packed: bool = False,
    ) -> None:
        original_init(
            self,
            strategy=strategy,
            num_bits=num_bits,
            group_size=group_size,
            is_static_input_scheme=is_static_input_scheme,
            input_symmetric=input_symmetric,
        )
        self._vllm_fl_checkpoint_packed = packed
        self._vllm_fl_pack_factor = 32 // num_bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        if not getattr(self, "_vllm_fl_checkpoint_packed", False):
            return original_create_weights(
                self,
                layer,
                output_size,
                input_size,
                output_partition_sizes,
                input_size_per_partition,
                params_dtype,
                weight_loader,
                **kwargs,
            )

        output_size_per_partition = sum(output_partition_sizes)
        row_parallel = input_size != input_size_per_partition
        effective_group_size = (
            input_size_per_partition if self.group_size == -1 and row_parallel
            else input_size if self.group_size == -1
            else self.group_size
        )
        if input_size_per_partition % effective_group_size:
            raise ValueError(
                f"input partition {input_size_per_partition} is not divisible "
                f"by W4 group size {effective_group_size}"
            )

        kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,
            zero_points=False,
            has_g_idx=False,
        )
        kernel_type = choose_mp_linear_kernel(kernel_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info(
                "Using %s for packed CompressedTensorsW4A8Int",
                kernel_type.__name__,
            )
            self._kernel_backends_being_used.add(kernel_type.__name__)

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self._vllm_fl_pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self._vllm_fl_pack_factor,
            packed_dim=1,
        )
        layer.register_parameter("weight_packed", weight)

        scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                input_size_per_partition // effective_group_size,
                dtype=params_dtype,
            ),
        }
        if self.group_size == -1 and row_parallel:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0, **scale_args
            )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **scale_args
            )
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter(
            "weight_shape",
            BasevLLMParameter(
                data=torch.empty(2, dtype=torch.int64),
                weight_loader=weight_loader,
            ),
        )
        self.kernel = kernel_type(
            kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name=None,
            w_gidx_param_name=None,
        )

    def get_scheme_from_parts(
        self,
        weight_quant,
        input_quant,
        format: str | None = None,
        layer_name: str | None = None,
    ):
        resolved_format = format if format is not None else self.quant_format
        if (
            resolved_format == CompressionFormat.pack_quantized.value
            and self._is_dynamic_token_w4a8_int(weight_quant, input_quant)
        ):
            return scheme_cls(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                group_size=weight_quant.group_size,
                is_static_input_scheme=False,
                input_symmetric=input_quant.symmetric,
                packed=True,
            )
        return original_get_scheme(
            self,
            weight_quant,
            input_quant,
            format=format,
            layer_name=layer_name,
        )

    scheme_cls.__init__ = scheme_init
    scheme_cls.create_weights = create_weights
    config_cls._get_scheme_from_parts = get_scheme_from_parts
    scheme_cls._vllm_fl_packed_w4a8 = True


def _install_cpu_gemm_guard() -> None:
    from vllm.model_executor.layers import utils as layer_utils

    original = layer_utils.dispatch_cpu_unquantized_gemm
    if getattr(original, "_vllm_fl_ndim_guard", False):
        return

    def guarded(layer: torch.nn.Module, remove_weight: bool) -> None:
        weight = getattr(layer, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim != 2:
            return
        return original(layer, remove_weight)

    guarded._vllm_fl_ndim_guard = True
    layer_utils.dispatch_cpu_unquantized_gemm = guarded


def _install_text_only_vision_guard() -> None:
    from contextvars import ContextVar
    from vllm.model_executor.models import qwen3_5 as qwen

    if getattr(qwen, "_vllm_fl_text_only_vision", False):
        return

    text_only_build = ContextVar("vllm_fl_qwen_text_only_build", default=False)
    original_vision_init = qwen.Qwen3_VisionTransformer.__init__

    def vision_init(self, *args, **kwargs) -> None:
        if text_only_build.get():
            kwargs["quant_config"] = None
        original_vision_init(self, *args, **kwargs)

    qwen.Qwen3_VisionTransformer.__init__ = vision_init

    for model_cls in (
        qwen.Qwen3_5ForConditionalGeneration,
        qwen.Qwen3_5MoeForConditionalGeneration,
    ):
        original_model_init = model_cls.__init__

        def model_init(
            self,
            *,
            vllm_config,
            prefix: str = "model",
            _original=original_model_init,
        ) -> None:
            language_only = bool(
                vllm_config.model_config.multimodal_config.language_model_only
            )
            token = text_only_build.set(language_only)
            try:
                _original(self, vllm_config=vllm_config, prefix=prefix)
            finally:
                text_only_build.reset(token)

        model_cls.__init__ = model_init

    qwen._vllm_fl_text_only_vision = True


def _install_cpu_attention_block_constraint() -> None:
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    if hasattr(CPUAttentionBackend, "get_supported_kernel_block_sizes"):
        return

    CPUAttentionBackend.get_supported_kernel_block_sizes = staticmethod(
        lambda: [MultipleOf(32)]
    )


def _install_cpu_cleanup_guard() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner._cleanup_profiling_kv_cache
    if getattr(original, "_vllm_fl_cpu_guard", False):
        return

    def cleanup(self) -> None:
        if self.device.type != "cpu":
            return original(self)
        if hasattr(self, "kv_caches") and self.kv_caches:
            for index in range(len(self.kv_caches)):
                self.kv_caches[index] = None
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None
        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
            if hasattr(layer, "impl"):
                if hasattr(layer.impl, "_k_scale_cache"):
                    layer.impl._k_scale_cache = None
                if hasattr(layer.impl, "_v_scale_cache"):
                    layer.impl._v_scale_cache = None
        gc.collect()

    cleanup._vllm_fl_cpu_guard = True
    GPUModelRunner._cleanup_profiling_kv_cache = cleanup


def install_arm_cpu_vllm_0202_compat() -> bool:
    """Install all Python-only compatibility hooks exactly once."""
    global _INSTALLED
    if _INSTALLED:
        return False
    _require_vllm_0202()
    _install_packed_w4a8()
    _install_cpu_gemm_guard()
    _install_text_only_vision_guard()
    _install_cpu_attention_block_constraint()
    _install_cpu_cleanup_guard()
    _INSTALLED = True
    return True


__all__ = ["install_arm_cpu_vllm_0202_compat"]
