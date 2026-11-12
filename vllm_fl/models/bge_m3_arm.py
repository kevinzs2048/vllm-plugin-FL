# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 BGE-M3 Arm integration. No checkpoint or vLLM changes."""

import copy
import os
from types import MethodType

import torch
from vllm import ModelRegistry
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.models.roberta import BgeM3EmbeddingModel


def _sme_attention(
    self,
    layer,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
    output_scale=None,
    output_block_scale=None,
):
    if output_scale is not None or output_block_scale is not None:
        raise ValueError("BGE attention is unquantized")
    if attn_metadata is None:
        return output
    if attn_metadata.causal:
        raise ValueError("BGE requires noncausal attention")
    from flag_gems.runtime.backend._arm.fused.vision_attention import (
        vision_attention_sme_bf16,
    )

    bounds = attn_metadata.query_start_loc.tolist()
    if bounds[0] != 0 or bounds[-1] != attn_metadata.num_actual_tokens:
        raise ValueError("Invalid BGE sequence boundaries")
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            raise ValueError("Empty BGE sequence")
        q, k, v = [
            t[start:end]
            .reshape(end - start, self.num_heads, self.head_size)
            .transpose(0, 1)
            .unsqueeze(0)
            for t in (query, key, value)
        ]
        result = vision_attention_sme_bf16(
            q,
            k,
            v,
            scale=self.scale,
            threads=int(os.environ.get("FL_BGE_ATTN_THREADS", "4")),
            backend="raw",
        )
        output[start:end].reshape(end - start, self.num_heads, self.head_size).copy_(
            result.squeeze(0).transpose(0, 1)
        )
    layer._fl_bge_attention_calls += 1
    return output


def _fused_output(self, hidden_states, input_tensor):
    hidden_states, _ = self.dense(hidden_states)
    if hidden_states.numel() // hidden_states.shape[-1] < 1024:
        return self.LayerNorm(hidden_states + input_tensor)
    from flag_gems.runtime.backend._arm.fused.add_layer_norm import add_layer_norm

    result = add_layer_norm(
        hidden_states,
        input_tensor,
        self.LayerNorm.weight,
        self.LayerNorm.bias,
        self.LayerNorm.eps,
        threads=int(os.environ.get("FL_BGE_NORM_THREADS", "8")),
    )
    self._fl_bge_norm_calls += 1
    return result


class FlagOSBgeM3EmbeddingModel(BgeM3EmbeddingModel):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config
        if os.environ.get("FL_BGE_W8_FFN_RAW", "0") != "0":
            raise ValueError("W8 FFN Raw requires the quantized BGE architecture")
        if any(
            os.environ.get(k, "native") != "native"
            for k in ("FL_BGE_NORM", "FL_BGE_ATTENTION", "FL_BGE_LINEAR")
        ):
            if (
                config.dtype != torch.bfloat16
                or config.quantization is not None
                or not config.enforce_eager
            ):
                raise ValueError(
                    "BGE optimized operators require unquantized BF16 eager inference"
                )
        norm = os.environ.get("FL_BGE_NORM", "native")
        if norm not in {"native", "triton"}:
            raise ValueError("FL_BGE_NORM must be native or triton")
        if norm == "triton":
            from vllm.model_executor.models.bert import BertOutput, BertSelfOutput

            for module in self.modules():
                if isinstance(module, (BertOutput, BertSelfOutput)):
                    module._fl_bge_norm_calls = 0
                    module.forward = MethodType(_fused_output, module)
        attention = os.environ.get("FL_BGE_ATTENTION", "native")
        if attention not in {"native", "sme"}:
            raise ValueError("FL_BGE_ATTENTION must be native or sme")
        if attention == "sme":
            from vllm.model_executor.layers.attention import EncoderOnlyAttention

            for module in self.modules():
                if isinstance(module, EncoderOnlyAttention):
                    if (
                        module.impl.head_size != 64
                        or module.impl.num_heads != module.impl.num_kv_heads
                    ):
                        raise ValueError(
                            "Expected BGE equal Q/K/V 64-dimensional heads"
                        )
                    module._fl_bge_attention_calls = 0
                    module.impl.forward = MethodType(_sme_attention, module.impl)
        elementwise = os.environ.get("FL_BGE_BF16_ELEMENTWISE", "0")
        if elementwise not in {"0", "1"}:
            raise ValueError("FL_BGE_BF16_ELEMENTWISE must be0 or1")
        if elementwise == "1":
            if (
                config.dtype != torch.bfloat16
                or config.quantization is not None
                or not config.enforce_eager
            ):
                raise ValueError(
                    "Prepared BGE elementwise operators require BF16 eager inference"
                )
            if norm != "native" or os.environ.get("FL_BGE_W8_ADD_NORM", "0") != "0":
                raise ValueError(
                    "BF16 elementwise selection requires native norm mode and no W8 fusion"
                )
            from .bge_m3_quant import _configure_elementwise

            _configure_elementwise(self)
        route = os.environ.get("FL_BGE_LINEAR", "native")
        if route not in {"native", "sme"}:
            raise ValueError("FL_BGE_LINEAR must be native or sme")
        self._fl_bge_route = route
        if route == "native":
            return
        if (
            config.dtype != torch.bfloat16
            or config.quantization is not None
            or not config.enforce_eager
        ):
            raise ValueError("BGE SME requires unquantized BF16 eager inference")
        from flag_gems.runtime.backend._arm.fused.bf16_sme import PreparedBF16SME

        threads = int(os.environ.get("FL_BGE_SME_THREADS", "8"))
        min_rows = int(os.environ.get("FL_BGE_BF16_SME_MIN_M", "32"))
        if not 1 <= min_rows <= 32:
            raise ValueError("BF16 SME minimum row count must be in [1,32]")
        kernel = os.environ.get("FL_BGE_SME_KERNEL", "bf16_16x64_striped")
        if kernel not in {"bf16_32x32", "bf16_16x64", "bf16_16x64_striped"}:
            raise ValueError("Select a BF16 SME kernel")
        for name, layer in self.named_modules():
            method = getattr(layer, "quant_method", None)
            if not isinstance(layer, LinearBase) or not isinstance(
                method, UnquantizedLinearMethod
            ):
                continue
            if min(layer.weight.shape) < 128:
                continue
            replacement = copy.copy(method)

            def process(
                self, layer, original=type(method).process_weights_after_loading
            ):
                original(self, layer)
                layer._fl_bge_plan = PreparedBF16SME(
                    layer.weight, threads, kernel=kernel
                )

            def apply(self, layer, x, bias=None, original=type(method).apply):
                if x.numel() // x.shape[-1] >= min_rows:
                    result = layer._fl_bge_plan(x, bias)
                    layer._fl_bge_calls += 1
                    return result
                return original(self, layer, x, bias)

            replacement.process_weights_after_loading = MethodType(process, replacement)
            replacement.apply = MethodType(apply, replacement)
            layer.quant_method = replacement
            layer._fl_bge_calls = 0


def register():
    ModelRegistry.register_model(
        "FlagOSBgeM3EmbeddingModel",
        "vllm_fl.models.bge_m3_arm:FlagOSBgeM3EmbeddingModel",
    )
    ModelRegistry.register_model(
        "FlagOSBgeM3QuantModel", "vllm_fl.models.bge_m3_quant:FlagOSBgeM3QuantModel"
    )
