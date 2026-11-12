# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit BGE quantized body candidates; embedding/norm/bias/heads stay BF16."""

import copy
import json
from functools import lru_cache, partial
import os
import platform
from pathlib import Path
from types import MethodType
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.models.roberta import BgeM3EmbeddingModel


def configure_short_native_attention(
    model, attention_op, max_tokens=128, *, encoder_guarded=False
):
    """Experimental typed bridge, retaining native arithmetic and guarded fallback."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention import EncoderOnlyAttention

    for module in model.modules():
        if not isinstance(module, EncoderOnlyAttention):
            continue
        impl = module.impl
        if (
            impl.num_heads != impl.num_kv_heads
            or impl.head_size != 64
            or impl.scale != 0.125
            or impl.isa != "neon"
            or impl.logits_soft_cap != 0.0
            or impl.kv_cache_dtype != "auto"
            or impl.alibi_slopes is not None
            or impl.sinks is not None
            or module._k_scale_float != 1.0
            or module._v_scale_float != 1.0
        ):
            raise ValueError(
                "Short Arm bridge requires native equal-head BF16 attention"
            )
        original = module.forward

        def forward(self, query, key, value, original=original, **kwargs):
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                metadata = next(iter(metadata.values()), None)
            if (
                kwargs
                or metadata is None
                or metadata.causal
                or metadata.seq_lens.numel() != 1
                or query.shape[0] > max_tokens
            ):
                return original(query, key, value, **kwargs)
            # The compiled encoder may move this check to its Python entry,
            # where changing token counts do not specialize the dynamic graph.
            # Direct callers retain the check, including during compilation.
            if (
                not encoder_guarded or not torch.compiler.is_compiling()
            ) and metadata.num_actual_tokens != query.shape[0]:
                return original(query, key, value, **kwargs)
            q, k, v = [
                t.view(t.shape[0], self.num_heads, self.head_size)
                for t in (query, key, value)
            ]
            output = attention_op(
                q,
                k,
                v,
                metadata.slot_mapping,
                metadata.query_start_loc,
                metadata.seq_lens,
                metadata.block_table,
                metadata.scheduler_metadata,
                metadata.encoder_cache.shape[0],
                metadata.encoder_cache.shape[2],
                self.impl.sliding_window,
            )
            return output.reshape(query.shape[0], self.num_heads * self.head_size)

        module.forward = MethodType(forward, module)


def guard_short_compiled_encoder(
    original_encoder, compiled_encoder, *, dynamic_encoder=None, max_tokens=128
):
    """Check request metadata before entering any compiled encoder graph.

    Pair with configure_short_native_attention(..., encoder_guarded=True).
    Ordinary Dynamo guards remain enabled. Unsupported requests use the eager
    encoder, whose attention entry still performs its own token-count check.
    """
    from vllm.forward_context import get_forward_context

    def forward(hidden_states, *args, **kwargs):
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        rows = hidden_states.shape[0]
        if (
            metadata is None
            or metadata.causal
            or metadata.seq_lens.numel() != 1
            or rows > max_tokens
            or metadata.num_actual_tokens != rows
        ):
            return original_encoder(hidden_states, *args, **kwargs)
        if dynamic_encoder is not None and rows not in (32, 64, 128):
            return dynamic_encoder(hidden_states, *args, **kwargs)
        return compiled_encoder(hidden_states, *args, **kwargs)

    return forward


def _w8_rules():
    text = os.environ.get("FL_BGE_W8_SCHEDULE", "")
    if not text:
        return ()
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("BGE W8 schedule must be a list")
    result = []
    previous = 0
    categories = {"qkv", "attention_out", "ffn_up", "ffn_down"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != categories | {"max_tokens"}:
            raise ValueError("BGE W8 schedule requires all four matrix categories")
        limit = row["max_tokens"]
        if type(limit) is not int or limit <= previous:
            raise ValueError("BGE W8 schedule limits must strictly increase")
        workers = {}
        for category in categories:
            pair = row[category]
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(x) is not int for x in pair)
                or not 1 <= pair[0] <= (os.cpu_count() or 1)
                or not 0 <= pair[1] <= pair[0]
            ):
                raise ValueError("Invalid BGE W8 worker pair")
            workers[category] = tuple(pair)
        result.append((limit, workers))
        previous = limit
    return tuple(result)


def _w8_override(m, category, rules):
    for limit, workers in rules:
        if m <= limit:
            return workers[category]
    return None


class FlagOSBgeM3QuantModel(BgeM3EmbeddingModel):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        c = vllm_config.model_config
        if (
            c.dtype != torch.bfloat16
            or c.quantization is not None
            or not c.enforce_eager
        ):
            raise ValueError(
                "Quantized BGE candidates require a BF16 source and eager mode"
            )
        if os.environ.get("FL_BGE_NORM", "native") != "native":
            raise ValueError(
                "Quantized BGE uses FL_BGE_RESIDUAL_NORM; FL_BGE_NORM is BF16-only"
            )
        if os.environ.get("FL_BGE_ATTENTION", "native") != "native":
            raise ValueError(
                "Quantized BGE currently requires native attention; SME candidate is not qualified"
            )
        mode = os.environ["FL_BGE_QUANT"]
        panel_packing = os.environ.get("FL_BGE_W8_PANEL_PACK", "0")
        if panel_packing not in {"0", "1"}:
            raise ValueError("FL_BGE_W8_PANEL_PACK must be0 or1")
        if mode in {"w8a8", "w8a8-sq", "w8a8-sq-fit"}:
            backend = os.environ.get(
                "FL_BGE_QUANT_BACKEND",
                "sme" if platform.system() == "Darwin" else "i8mm",
            )
            if backend == "sme":
                from flag_gems.runtime.backend._arm.fused.prepared_w8a8_sme import (
                    PreparedW8A8SME as Plan,
                )

                Plan = partial(
                    Plan,
                    kernel=os.environ.get("FL_BGE_W8_KERNEL", "scratch"),
                    sme_threads=int(os.environ.get("FL_BGE_HYBRID_SME_THREADS", "3")),
                    panel_packing=panel_packing == "1",
                )
            elif backend == "i8mm":
                from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import (
                    PreparedW8A8I8MM as Plan,
                )

                Plan = partial(
                    Plan,
                    kernel=os.environ.get("FL_BGE_I8MM_KERNEL", "staged"),
                    order=os.environ.get("FL_BGE_I8MM_ORDER", "n"),
                )
            else:
                raise ValueError("W8 backend must be sme or i8mm")
        else:
            raise ValueError("Unknown BGE quantization")
        if panel_packing == "1" and (not mode.startswith("w8a8") or backend != "sme"):
            raise ValueError("Panel packing requires the SME W8 backend")
        threads = int(os.environ.get("FL_BGE_SME_THREADS", "8"))
        calibration = None
        if mode in {"w8a8-sq", "w8a8-sq-fit"}:
            path = Path(
                os.environ["FL_BGE_SQ_FIT"]
                if mode == "w8a8-sq-fit"
                else os.environ["FL_BGE_CALIBRATION"]
            )
            calibration = torch.load(path, map_location="cpu", weights_only=True)
        self._fl_bge_quant = mode
        native_rules = _w8_rules()
        if native_rules and (not mode.startswith("w8a8") or backend != "sme"):
            raise ValueError("BGE W8 scheduling rules require the SME backend")
        schedule = os.environ.get("FL_BGE_LINEAR_SCHEDULE", "default")
        if schedule not in {"default", "short-v1"}:
            raise ValueError("Unknown BGE linear schedule")
        if schedule != "default" and (not mode.startswith("w8a8") or backend != "sme"):
            raise ValueError(
                "Short-input schedule requires the prepared W8 SME backend"
            )
        for name, layer in self.named_modules():
            method = getattr(layer, "quant_method", None)
            if not isinstance(layer, LinearBase) or not isinstance(
                method, UnquantizedLinearMethod
            ):
                continue
            replacement = copy.copy(method)

            def process(self, layer, name=name):
                if calibration is None:
                    layer._fl_bge_quant_plan = Plan(layer.weight, threads)
                elif mode == "w8a8-sq-fit":
                    layer._fl_bge_quant_plan = Plan(
                        layer.weight, threads, smoothing=calibration[name]
                    )
                elif mode == "w8a8-sq":
                    activations = calibration[name].float().clamp_min(1e-5)
                    weights = (
                        layer.weight.detach().float().abs().amax(0).clamp_min(1e-5)
                    )
                    smoothing = (activations.sqrt() / weights.sqrt()).clamp(1e-4, 1e4)
                    layer._fl_bge_quant_plan = Plan(
                        layer.weight, threads, smoothing=smoothing
                    )
                else:
                    layer._fl_bge_quant_plan = Plan(
                        layer.weight, threads, calibration=calibration[name]
                    )
                del layer.weight

            category = (
                "qkv"
                if "qkv_proj" in name
                else (
                    "attention_out"
                    if "attention.output.dense" in name
                    else "ffn_up" if "intermediate.dense" in name else "ffn_down"
                )
            )

            def apply(self, layer, x, bias=None, name=name, category=category):
                plan = layer._fl_bge_quant_plan
                m = x.numel() // x.shape[-1]
                override = (
                    _w8_override(m, category, native_rules)
                    if native_rules and m <= native_rules[-1][0]
                    else None
                )
                if override is not None:
                    result = plan(x, bias, threads=override[0], sme_threads=override[1])
                elif schedule != "default" and m <= 32:
                    result = plan(x, bias, threads=4, sme_threads=1)
                elif (
                    schedule != "default"
                    and m <= 128
                    and "attention.output.dense" in name
                ):
                    result = plan(x, bias, threads=6, sme_threads=2)
                else:
                    result = plan(x, bias)
                if not torch.compiler.is_compiling():
                    layer._fl_bge_quant_calls += 1
                return result

            replacement.process_weights_after_loading = MethodType(process, replacement)
            replacement.apply = MethodType(apply, replacement)
            layer.quant_method = replacement
            layer._fl_bge_quant_calls = 0
        if backend == "i8mm":
            _configure_i8mm_elementwise(self)
            _configure_i8mm_ffn(self)
            _configure_short_encoder(self)
        else:
            _configure_elementwise(self, native_rules)
            _configure_ffn(self, native_rules)


def _configure_elementwise(model, native_rules=()):
    """Explicit candidates, off by default; normal long-input route is retained."""
    fused_linear = os.environ.get("FL_BGE_W8_ADD_NORM", "0") == "1"
    norm = os.environ.get("FL_BGE_RESIDUAL_NORM", "native")
    gelu = os.environ.get("FL_BGE_GELU", "native")
    limits = dict(
        norm_max=int(os.environ.get("FL_BGE_FUSED_NORM_MAX_M", "128")),
        gelu_max=int(os.environ.get("FL_BGE_GELU_MAX_M", "128")),
        norm_threads=int(
            os.environ.get(
                "FL_BGE_ELEMENTWISE_LONG_THREADS",
                os.environ.get("FL_BGE_NORM_THREADS", "4"),
            )
        ),
        gelu_threads=int(
            os.environ.get(
                "FL_BGE_ELEMENTWISE_LONG_THREADS",
                os.environ.get("FL_BGE_GELU_THREADS", "8"),
            )
        ),
    )
    model._fl_bge_long_elementwise = limits

    if norm not in {"native", "raw"} or gelu not in {"native", "lookup"}:
        raise ValueError("Invalid BGE elementwise backend")
    from vllm.model_executor.models.bert import BertOutput, BertSelfOutput

    if norm != "native":
        from flag_gems.runtime.backend._arm.fused.prepared_elementwise import (
            PreparedAddLayerNormRaw as Plan,
        )

        threads = int(os.environ.get("FL_BGE_NORM_THREADS", "4"))
        max_m = int(os.environ.get("FL_BGE_FUSED_NORM_MAX_M", "128"))
        max_fused_m = int(os.environ.get("FL_BGE_W8_ADD_NORM_MAX_M", str(max_m)))
        for module in model.modules():
            if not isinstance(module, (BertOutput, BertSelfOutput)):
                continue
            module._fl_bge_norm_plan = None
            module._fl_bge_norm_calls = 0
            module._fl_bge_w8_norm_plan = None
            module._fl_bge_norm_attention = isinstance(module, BertSelfOutput)

            def forward(self, hidden_states, input_tensor):
                m = hidden_states.numel() // hidden_states.shape[-1]
                if fused_linear and m <= max_fused_m:
                    plan = self.dense._fl_bge_quant_plan
                    if self._fl_bge_w8_norm_plan is None:
                        from flag_gems.runtime.backend._arm.fused.prepared_w8a8_add_norm import (
                            PreparedW8A8AddNorm,
                        )
                        from vllm.distributed import (
                            get_tensor_model_parallel_world_size,
                        )

                        if get_tensor_model_parallel_world_size() != 1:
                            raise ValueError("Experimental fused BGE norm requires TP1")
                        self._fl_bge_w8_norm_plan = PreparedW8A8AddNorm(
                            plan,
                            self.LayerNorm.weight,
                            self.LayerNorm.bias,
                            self.LayerNorm.eps,
                            norm_threads=threads,
                        )
                    options = {}
                    category = (
                        "attention_out" if self._fl_bge_norm_attention else "ffn_down"
                    )
                    override = (
                        _w8_override(m, category, native_rules)
                        if native_rules and m <= native_rules[-1][0]
                        else None
                    )
                    if override is not None:
                        options = dict(threads=override[0], sme_threads=override[1])
                    elif (
                        os.environ.get("FL_BGE_LINEAR_SCHEDULE", "default")
                        == "short-v1"
                    ):
                        if m <= 32:
                            options = dict(threads=4, sme_threads=1)
                        elif self._fl_bge_norm_attention:
                            options = dict(threads=6, sme_threads=2)
                    output = self._fl_bge_w8_norm_plan(
                        hidden_states, input_tensor, self.dense.bias, **options
                    )
                    if not torch.compiler.is_compiling():
                        self.dense._fl_bge_quant_calls += 1
                    if not torch.compiler.is_compiling():
                        self._fl_bge_norm_calls += 1
                    return output
                hidden_states, _ = self.dense(hidden_states)
                if (
                    hidden_states.numel() // hidden_states.shape[-1]
                    > limits["norm_max"]
                ):
                    return self.LayerNorm(hidden_states + input_tensor)
                if self._fl_bge_norm_plan is None:
                    self._fl_bge_norm_plan = Plan(
                        self.LayerNorm.weight,
                        self.LayerNorm.bias,
                        self.LayerNorm.eps,
                        threads=limits["norm_threads"] if m > max_fused_m else threads,
                    )
                if not torch.compiler.is_compiling():
                    self._fl_bge_norm_calls += 1
                return self._fl_bge_norm_plan(hidden_states, input_tensor)

            module.forward = MethodType(forward, module)
    if gelu == "lookup":
        # vLLM's activation registry caches module instances. Own a model-local
        # copy before changing forward or attaching plans; retain sharing among
        # this model's layers without modifying other models or the registry.
        from vllm.model_executor.models.bert import BertIntermediate

        activations = {}
        for module in model.modules():
            if isinstance(module, BertIntermediate):
                original = module.intermediate_act_fn
                if id(original) not in activations:
                    activations[id(original)] = copy.deepcopy(original)
                module.intermediate_act_fn = activations[id(original)]
        from flag_gems.runtime.backend._arm.fused.prepared_elementwise import (
            PreparedGeluErfRaw,
        )

        plan = PreparedGeluErfRaw(
            threads=int(os.environ.get("FL_BGE_GELU_THREADS", "8")),
            reference_threads=torch.get_num_threads(),
        )
        long_plan = [
            None
        ]  # Shared by all layers; prepare only on the first long input.
        for name, module in model.named_modules():
            if not name.endswith("intermediate_act_fn"):
                continue
            from vllm.model_executor.layers.activation import GELU

            if (
                not isinstance(module, (torch.nn.GELU, GELU))
                or getattr(module, "approximate", "none") != "none"
            ):
                raise ValueError("Raw exact-erf candidate requires a standard erf GELU")
            original = module.forward
            module._fl_bge_gelu_calls = 0
            module._fl_bge_gelu_plan = plan

            def forward(self, x, original=original):
                if x.numel() // x.shape[-1] > limits["gelu_max"]:
                    return original(x)
                if not torch.compiler.is_compiling():
                    self._fl_bge_gelu_calls += 1
                if x.numel() // x.shape[-1] <= 128:
                    return plan(x)
                if long_plan[0] is None:
                    from flag_gems.runtime.backend._arm.fused.prepared_elementwise import (
                        PreparedGeluErfTriton,
                    )

                    long_plan[0] = PreparedGeluErfTriton(
                        threads=limits["gelu_threads"],
                        reference_threads=plan.reference_threads,
                        library_path=os.environ["FLAGGEMS_ARM_GELU_TABLES_LIBRARY"],
                    )
                return long_plan[0](x)

            module.forward = MethodType(forward, module)


def _configure_ffn(model, native_rules):
    enabled = os.environ.get("FL_BGE_W8_FFN_RAW", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("FL_BGE_W8_FFN_RAW must be0 or1")
    if enabled == "0":
        return
    max_rows = int(os.environ.get("FL_BGE_W8_FFN_MAX_TOKENS", "32"))
    if max_rows not in {32, 64, 128}:
        raise ValueError("FFN Raw supports qualified limits32,64 or128 only")
    if (
        not native_rules
        or native_rules[-1][0] < max_rows
        or os.environ.get("FL_BGE_W8_ADD_NORM", "0") != "1"
        or os.environ.get("FL_BGE_RESIDUAL_NORM", "native") != "raw"
        or os.environ.get("FL_BGE_GELU", "native") != "lookup"
        or int(os.environ.get("FL_BGE_FUSED_NORM_MAX_M", "128")) < max_rows
        or int(os.environ.get("FL_BGE_GELU_MAX_M", "128")) < max_rows
    ):
        raise ValueError(
            "FFN Raw requires W8 rules, raw norm and exact GELU covering its row limit"
        )
    from vllm.model_executor.models.bert import BertLayer
    from vllm.distributed import get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1:
        raise ValueError("Prepared FFN Raw requires TP1")
    norm_threads = int(os.environ.get("FL_BGE_NORM_THREADS", "4"))
    for layer in model.modules():
        if not isinstance(layer, BertLayer):
            continue
        layer._fl_bge_ffn_plan = None
        layer._fl_bge_ffn_calls = 0
        original = layer.forward

        def forward(self, hidden_states, original=original):
            # vLLM encoders use flattened [scheduled_tokens, hidden] tensors.
            m = hidden_states.size(0)
            if m > max_rows:
                return original(hidden_states)
            attention = self.attention(hidden_states)
            up = self.intermediate.dense
            output = self.output
            if self._fl_bge_ffn_plan is None:
                from flag_gems.runtime.backend._arm.fused.prepared_w8a8_ffn import (
                    PreparedW8A8FFN,
                )
                from flag_gems.runtime.backend._arm.fused.prepared_w8a8_add_norm import (
                    PreparedW8A8AddNorm,
                )

                if output._fl_bge_w8_norm_plan is None:
                    output._fl_bge_w8_norm_plan = PreparedW8A8AddNorm(
                        output.dense._fl_bge_quant_plan,
                        output.LayerNorm.weight,
                        output.LayerNorm.bias,
                        output.LayerNorm.eps,
                        norm_threads=norm_threads,
                    )
                self._fl_bge_ffn_plan = PreparedW8A8FFN(
                    up._fl_bge_quant_plan,
                    output._fl_bge_w8_norm_plan,
                    self.intermediate.intermediate_act_fn._fl_bge_gelu_plan,
                )
            ut, us = _w8_override(m, "ffn_up", native_rules)
            dt, ds = _w8_override(m, "ffn_down", native_rules)
            result = self._fl_bge_ffn_plan(
                attention,
                attention,
                up.bias,
                output.dense.bias,
                up_threads=ut,
                up_sme_threads=us,
                down_threads=dt,
                down_sme_threads=ds,
            )
            if not torch.compiler.is_compiling():
                up._fl_bge_quant_calls += 1
            if not torch.compiler.is_compiling():
                output.dense._fl_bge_quant_calls += 1
            if not torch.compiler.is_compiling():
                output._fl_bge_norm_calls += 1
            if not torch.compiler.is_compiling():
                self.intermediate.intermediate_act_fn._fl_bge_gelu_calls += 1
            if not torch.compiler.is_compiling():
                self._fl_bge_ffn_calls += 1
            return result

        layer.forward = MethodType(forward, layer)


def configure_bge_m3_cpu_inference(model, worker, *, compile_short=True):
    """Configure the loaded BGE instance before serving.

    I8MM initialization already owns its guarded encoder. The following
    additional preparation is specific to the validated M5 SME profile.

    Local candidate API. Keeps native short attention and selects the qualified
    SME Raw attention for >=512 tokens. No process-wide function replacement.
    The caller must not mutate this model or share the worker concurrently.
    """
    from dataclasses import replace
    from vllm.model_executor.layers.attention import EncoderOnlyAttention
    from .bge_m3_arm import _sme_attention

    if not isinstance(model, FlagOSBgeM3QuantModel):
        raise ValueError("Expected the prepared quantized BGE model")
    if platform.system() != "Darwin":
        # The I8MM model configures its guarded encoder during initialization.
        return
    if getattr(model, "_fl_bge_inference_configured", False):
        raise RuntimeError("BGE inference was already configured")
    if worker.thread_policy is None or worker.thread_policy.prefill != 18:
        raise ValueError("M5 profile requires the matching 18-thread loading policy")
    plans = [getattr(m, "_fl_bge_ffn_plan", None) for m in model.modules()]
    plans = [p for p in plans if p is not None]
    if len(plans) != 24 or any(p.gelu.reference_threads != 18 for p in plans):
        raise ValueError(
            "Warm the 32-token model once before preparing compiled execution"
        )
    worker.thread_policy = replace(
        worker.thread_policy,
        small_prefill_tokens=0,
        small_prefill_threads=0,
        prefill_ranges=((64, 8), (128, 12)),
    )
    attention = [m for m in model.modules() if isinstance(m, EncoderOnlyAttention)]
    if len(attention) != 24:
        raise ValueError("Expected 24 encoder attention layers")
    for module in attention:
        original = module.impl.forward
        module._fl_bge_attention_calls = 0

        def forward(
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
            original=original,
        ):
            if query.shape[0] >= 512:
                return _sme_attention(
                    self,
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            return original(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        module.impl.forward = MethodType(forward, module.impl)
    bridge = Path(os.environ["FL_BGE_ATTENTION_BRIDGE_LIBRARY"])
    if not bridge.is_file():
        raise ValueError("Matching attention bridge library is missing")
    torch.ops.load_library(str(bridge))

    @torch.library.register_fake("vllm_fl_cpu_native::encoder_attention.arm")
    def bridge_fake(
        q,
        k,
        v,
        slots,
        starts,
        lengths,
        blocks,
        schedule,
        cache_blocks,
        block_size,
        window,
    ):
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    configure_short_native_attention(
        model,
        torch.ops.vllm_fl_cpu_native.encoder_attention.arm,
        encoder_guarded=compile_short,
    )
    if os.environ.get("FL_BGE_EXTEND_ELEMENTWISE", "0") == "1":
        if any(
            getattr(module, "_fl_bge_norm_plan", None) is not None
            for module in model.modules()
        ):
            raise RuntimeError(
                "Long elementwise must be enabled after the original initialization"
            )
        model._fl_bge_long_elementwise.update(
            norm_max=8192, gelu_max=8192, norm_threads=18, gelu_threads=18
        )
    if compile_short:
        original = model.model.encoder.forward
        options = {"size_asserts": True, "cpp_wrapper": True}
        static = torch.compile(
            original, backend="inductor", fullgraph=True, dynamic=False, options=options
        )
        dynamic = torch.compile(
            original, backend="inductor", fullgraph=True, dynamic=True, options=options
        )
        model.model.encoder.forward = guard_short_compiled_encoder(
            original, static, dynamic_encoder=dynamic
        )
    model._fl_bge_inference_configured = True


def _configure_i8mm_elementwise(model):
    """Explicit candidates, off by default; normal long-input route is retained."""
    fused_linear = os.environ.get("FL_BGE_W8_ADD_NORM", "0") == "1"
    fused_max_m = int(os.environ.get("FL_BGE_W8_ADD_NORM_MAX_M", "128"))
    if fused_max_m < 1:
        raise ValueError("Positive fused norm row limit required")
    norm = os.environ.get("FL_BGE_RESIDUAL_NORM", "native")
    if fused_linear:
        from vllm.distributed import get_tensor_model_parallel_world_size

        if norm != "raw" or get_tensor_model_parallel_world_size() != 1:
            raise ValueError("I8MM fused norm requires raw norm and TP1")
    gelu = os.environ.get("FL_BGE_GELU", "native")
    if norm not in {"native", "raw"} or gelu not in {"native", "lookup"}:
        raise ValueError("Invalid BGE elementwise backend")
    from vllm.model_executor.models.bert import BertOutput, BertSelfOutput

    if norm != "native":
        from flag_gems.runtime.backend._arm.fused.prepared_elementwise import (
            PreparedAddLayerNormRaw as Plan,
        )

        threads = int(os.environ.get("FL_BGE_NORM_THREADS", "4"))
        max_m = int(os.environ.get("FL_BGE_FUSED_NORM_MAX_M", "128"))
        for module in model.modules():
            if not isinstance(module, (BertOutput, BertSelfOutput)):
                continue
            module._fl_bge_norm_plan = None
            module._fl_bge_norm_calls = 0
            module._fl_bge_w8_norm_plan = None
            module._fl_bge_fused_norm_calls = 0

            def forward(self, hidden_states, input_tensor):
                if fused_linear and hidden_states.numel() // hidden_states.shape[
                    -1
                ] <= min(max_m, fused_max_m):
                    if self._fl_bge_w8_norm_plan is None:
                        from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import (
                            PreparedW8A8I8MMAddNorm,
                        )

                        if self.dense.skip_bias_add:
                            raise ValueError(
                                "Fused BGE norm requires inline linear bias"
                            )
                        self._fl_bge_w8_norm_plan = PreparedW8A8I8MMAddNorm(
                            self.dense._fl_bge_quant_plan,
                            self.LayerNorm.weight,
                            self.LayerNorm.bias,
                            self.LayerNorm.eps,
                            norm_threads=threads,
                        )
                    result = self._fl_bge_w8_norm_plan(
                        hidden_states, input_tensor, self.dense.bias
                    )
                    if not torch.compiler.is_compiling():
                        self.dense._fl_bge_quant_calls += 1
                        self._fl_bge_norm_calls += 1
                        self._fl_bge_fused_norm_calls += 1
                    return result
                hidden_states, _ = self.dense(hidden_states)
                if hidden_states.numel() // hidden_states.shape[-1] > max_m:
                    return self.LayerNorm(hidden_states + input_tensor)
                if self._fl_bge_norm_plan is None:
                    self._fl_bge_norm_plan = Plan(
                        self.LayerNorm.weight,
                        self.LayerNorm.bias,
                        self.LayerNorm.eps,
                        threads=threads,
                    )
                if not torch.compiler.is_compiling():
                    self._fl_bge_norm_calls += 1
                return self._fl_bge_norm_plan(hidden_states, input_tensor)

            module.forward = MethodType(forward, module)
    if gelu == "lookup":
        from flag_gems.runtime.backend._arm.fused.prepared_elementwise import (
            PreparedGeluErfRaw,
        )

        plan = PreparedGeluErfRaw(
            threads=int(os.environ.get("FL_BGE_GELU_THREADS", "8")),
            reference_threads=torch.get_num_threads(),
        )
        max_gelu_m = int(os.environ.get("FL_BGE_GELU_MAX_M", "128"))
        for name, module in model.named_modules():
            if not name.endswith("intermediate_act_fn"):
                continue
            from vllm.model_executor.layers.activation import GELU

            if (
                not isinstance(module, (torch.nn.GELU, GELU))
                or getattr(module, "approximate", "none") != "none"
            ):
                raise ValueError("Raw exact-erf candidate requires a standard erf GELU")
            original = module.forward
            module._fl_bge_gelu_calls = 0

            def forward(self, x, original=original):
                if x.numel() // x.shape[-1] > max_gelu_m:
                    return original(x)
                self._fl_bge_gelu_calls += 1
                return plan(x)

            module.forward = MethodType(forward, module)


def _configure_short_encoder(model):
    """Opt-in graph capture of the short-input encoder; TLE math is unchanged."""
    if os.environ.get("FL_BGE_SHORT_ENCODER_COMPILE", "0") != "1":
        return
    if (
        os.environ.get("FL_BGE_QUANT_BACKEND") != "i8mm"
        or os.environ.get("FLAGGEMS_ARM_I8MM_TENSOR_RAW") != "1"
        or os.environ.get("FL_BGE_W8_ADD_NORM") != "1"
    ):
        raise ValueError("Short encoder capture requires typed I8MM Raw and fused norm")
    max_m = int(os.environ.get("FL_BGE_SHORT_ENCODER_MAX_M", "128"))
    if not 1 <= max_m <= int(os.environ.get("FL_BGE_W8_ADD_NORM_MAX_M", "128")):
        raise ValueError("Compiled row limit must be within fused norm coverage")
    backend = os.environ.get("FL_BGE_SHORT_ENCODER_BACKEND", "inductor")
    if backend not in {"eager", "inductor"}:
        raise ValueError("Unsupported short encoder compiler backend")
    fast_cpu = os.environ.get("FL_BGE_SHORT_ENCODER_FASTPATH", "0") == "1"
    specialize = os.environ.get("FL_BGE_SHORT_ENCODER_SPECIALIZE", "0")
    if specialize not in {"0", "1"}:
        raise ValueError("FL_BGE_SHORT_ENCODER_SPECIALIZE must be 0 or 1")
    specialize = specialize == "1"
    if specialize and (not fast_cpu or backend != "inductor"):
        raise ValueError("Short specialization requires the Inductor CPU fast path")
    if fast_cpu:
        _configure_short_cpu_attention(model, max_m)
    from vllm.forward_context import get_forward_context

    encoder = model.model.encoder
    original = encoder.forward
    encoder._fl_bge_compiled_forward = None
    encoder._fl_bge_eager_forward = original

    def forward(self, hidden_states, *args, **kwargs):
        if hidden_states.numel() // hidden_states.shape[-1] > max_m:
            return original(hidden_states, *args, **kwargs)
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        # The v0.24 CPU attention path reads request boundaries from context.
        # Capture has only been validated for one sequence; preserve eager
        # attention metadata handling for multi-request and profiling calls.
        if (
            metadata is None
            or metadata.seq_lens.numel() != 1
            or fast_cpu
            and metadata.num_actual_tokens != hidden_states.shape[0]
        ):
            return original(hidden_states, *args, **kwargs)
        if self._fl_bge_compiled_forward is None:
            # Prepare lazy plans before tracing. First-call preparation/compilation
            # belongs to warmup; no output buffers are cached across calls.
            original(hidden_states, *args, **kwargs)
            # Keep both Dynamo input/module guards and generated size assertions.
            # Short graph specialization does not require disabling either check.
            options = (
                {"size_asserts": True} if fast_cpu and backend == "inductor" else None
            )
            if specialize:
                # Pass compiler options to this encoder only. Do not replace
                # torch.compile or mutate global Inductor configuration.
                options = dict(options or {}, cpp_wrapper=True, size_asserts=True)
                dynamic_forward = torch.compile(
                    original,
                    backend=backend,
                    fullgraph=True,
                    dynamic=True,
                    options=options,
                )
                static_forward = torch.compile(
                    original,
                    backend=backend,
                    fullgraph=True,
                    dynamic=False,
                    options=options,
                )

                def selected(hidden_states, *args, **kwargs):
                    target = (
                        static_forward
                        if hidden_states.shape[0] in (32, 64, 128)
                        else dynamic_forward
                    )
                    return target(hidden_states, *args, **kwargs)

                self._fl_bge_compiled_forward = selected
            else:
                self._fl_bge_compiled_forward = torch.compile(
                    original,
                    backend=backend,
                    fullgraph=True,
                    dynamic=True,
                    options=options,
                )
        return self._fl_bge_compiled_forward(hidden_states, *args, **kwargs)

    encoder.forward = MethodType(forward, encoder)


@lru_cache(maxsize=1)
def _short_cpu_attention_operator():
    """Typed framework bridge to the existing vLLM CPU attention arithmetic."""
    path = os.environ.get("FL_BGE_ATTENTION_BRIDGE_LIBRARY")
    if path:
        from flag_gems.runtime.backend._arm.tle_raw.native_raw import (
            initialize_native_raw,
        )

        initialize_native_raw(os.environ["FLAGGEMS_ARM_W8G32_I8MM_LIBRARY"])
        torch.ops.load_library(path)

        @torch.library.register_fake("vllm_fl_cpu_native::encoder_attention.arm")
        def meta(
            q,
            k,
            v,
            slots,
            starts,
            lengths,
            blocks,
            schedule,
            cache_blocks,
            block_size,
            window,
        ):
            return torch.empty(q.shape, dtype=q.dtype, device=q.device)

        op = torch.ops.vllm_fl_cpu_native.encoder_attention.arm

        def selected(
            q,
            k,
            v,
            slots,
            starts,
            lengths,
            blocks,
            schedule,
            scale,
            isa,
            cache_blocks,
            block_size,
            window,
            cap,
        ):
            if scale != 0.125 or isa != "neon" or cap != 0.0:
                raise ValueError(
                    "Qualified Arm attention requires scale1/8, Neon and no logit cap"
                )
            return op(
                q,
                k,
                v,
                slots,
                starts,
                lengths,
                blocks,
                schedule,
                cache_blocks,
                block_size,
                window,
            )

        return path, selected
    import threading

    workspace = threading.local()
    library = torch.library.Library("vllm_fl_bge_cpu", "FRAGMENT")
    library.define(
        "encoder_attention(Tensor q, Tensor k, Tensor v, Tensor slots, Tensor starts, Tensor lengths, Tensor blocks, Tensor schedule, float scale, str isa, int cache_blocks, int block_size, int window, float cap) -> Tensor"
    )
    reshape_and_cache = torch.ops._C.cpu_attn_reshape_and_cache.default
    attention = torch.ops._C.cpu_attention_with_kv_cache.default

    def cpu(
        q,
        k,
        v,
        slots,
        starts,
        lengths,
        blocks,
        schedule,
        scale,
        isa,
        cache_blocks,
        block_size,
        window,
        cap,
    ):
        if (
            q.ndim != 3
            or q.dtype != torch.bfloat16
            or not q.is_cpu
            or k.shape != q.shape
            or v.shape != q.shape
            or k.dtype != q.dtype
            or v.dtype != q.dtype
            or not k.is_cpu
            or not v.is_cpu
        ):
            raise ValueError("Expected matching CPU BF16 encoder Q/K/V")
        shape = (cache_blocks, q.shape[1], block_size, 2 * q.shape[2])
        key = (shape, q.dtype)
        if getattr(workspace, "key", None) != key:
            workspace.buffer = torch.zeros(shape, dtype=q.dtype, device=q.device)
            workspace.kc, workspace.vc = workspace.buffer.view(
                cache_blocks, q.shape[1], 2 * block_size, q.shape[2]
            ).chunk(2, dim=2)
            workspace.key = key
        # These views and their buffer are thread-local scratch storage.
        kc, vc = workspace.kc, workspace.vc
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        reshape_and_cache(k, v, kc, vc, slots, isa, 1.0, 1.0, "auto")
        attention(
            q,
            kc,
            vc,
            output,
            starts,
            lengths,
            scale,
            False,
            None,
            window,
            blocks,
            cap,
            schedule,
            None,
            None,
            1.0,
            1.0,
            "auto",
        )
        return output

    library.impl("encoder_attention", cpu, "CPU")

    @torch.library.register_fake("vllm_fl_bge_cpu::encoder_attention", lib=library)
    def meta(
        q,
        k,
        v,
        slots,
        starts,
        lengths,
        blocks,
        schedule,
        scale,
        isa,
        cache_blocks,
        block_size,
        window,
        cap,
    ):
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    # Retain registration and workspace ownership for the process lifetime.
    return library, torch.ops.vllm_fl_bge_cpu.encoder_attention.default


def _configure_short_cpu_attention(model, max_m):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention import EncoderOnlyAttention

    attention_op = _short_cpu_attention_operator()[1]
    for module in model.modules():
        if not isinstance(module, EncoderOnlyAttention):
            continue
        impl = module.impl
        if (
            impl.num_heads != impl.num_kv_heads
            or impl.head_size != 64
            or impl.scale != impl.head_size**-0.5
            or impl.kv_cache_dtype != "auto"
            or impl.alibi_slopes is not None
            or impl.sinks is not None
            or module._k_scale_float != 1.0
            or module._v_scale_float != 1.0
        ):
            raise ValueError(
                "Short BGE CPU attention requires unquantized equal Q/K/V heads"
            )
        original = module.forward

        def forward(self, query, key, value, original=original, **kwargs):
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                metadata = next(iter(metadata.values()), None)
            if (
                kwargs
                or metadata is None
                or metadata.causal
                or metadata.seq_lens.numel() != 1
                or query.shape[0] > max_m
            ):
                return original(query, key, value, **kwargs)
            if (
                not torch.compiler.is_compiling()
                and metadata.num_actual_tokens != query.shape[0]
            ):
                return original(query, key, value, **kwargs)
            q, k, v = [
                t.view(t.shape[0], self.num_heads, self.head_size)
                for t in (query, key, value)
            ]
            # Derive the validated scale from an integer to avoid float Tensor guards.
            y = attention_op(
                q,
                k,
                v,
                metadata.slot_mapping,
                metadata.query_start_loc,
                metadata.seq_lens,
                metadata.block_table,
                metadata.scheduler_metadata,
                self.head_size**-0.5,
                self.impl.isa,
                metadata.encoder_cache.shape[0],
                metadata.encoder_cache.shape[2],
                self.impl.sliding_window,
                self.impl.logits_soft_cap,
            )
            return y.reshape(query.shape[0], self.num_heads * self.head_size)

        module.forward = MethodType(forward, module)


def _configure_i8mm_ffn(model):
    """Opt-in short-input fusion; the original layer owns long-input behavior."""
    if os.environ.get("FL_BGE_W8_FFN_RAW", "0") != "1":
        return
    if (
        os.environ.get("FL_BGE_QUANT_BACKEND") != "i8mm"
        or os.environ.get("FLAGGEMS_ARM_I8MM_TENSOR_RAW") != "1"
        or os.environ.get("FL_BGE_W8_ADD_NORM") != "1"
    ):
        raise ValueError("FFN fusion requires typed I8MM Raw and fused norm")
    from vllm.model_executor.models.bert import BertLayer
    from flag_gems.runtime.backend._arm.fused.prepared_w8a8_i8mm import (
        PreparedW8A8I8MMAddNorm,
        PreparedW8A8I8MMFFN,
    )

    max_m = int(os.environ.get("FL_BGE_W8_ADD_NORM_MAX_M", "128"))
    threads = int(os.environ.get("FL_BGE_NORM_THREADS", "8"))
    tables = {}
    for layer in model.modules():
        if not isinstance(layer, BertLayer):
            continue
        original = layer.forward
        layer._fl_bge_ffn = None

        def forward(self, hidden_states, original=original):
            if hidden_states.numel() // hidden_states.shape[-1] > max_m:
                return original(hidden_states)
            x = self.attention(hidden_states)
            if self._fl_bge_ffn is None:
                up, down = self.intermediate.dense, self.output.dense
                if up.skip_bias_add or down.skip_bias_add:
                    raise ValueError("FFN fusion requires inline bias")
                activation = self.intermediate.intermediate_act_fn
                from vllm.model_executor.layers.activation import GELU

                if (
                    not isinstance(activation, (torch.nn.GELU, GELU))
                    or getattr(activation, "approximate", "none") != "none"
                ):
                    raise ValueError("FFN fusion requires standard erf GELU")
                # Capture the original implementation's complete BF16 mapping;
                # this is preparation, outside graph capture and timed inference.
                key = (type(activation), getattr(activation, "approximate", "none"))
                if key not in tables:
                    captured = getattr(activation, "_fl_bge_gelu_plan", None)
                    if os.environ.get("FL_BGE_GELU") == "table_raw":
                        tables[key] = captured.table
                    else:
                        values = (
                            torch.arange(65536, dtype=torch.int32)
                            .to(torch.int16)
                            .view(torch.bfloat16)
                        )
                        tables[key] = activation(values).contiguous()
                norm = PreparedW8A8I8MMAddNorm(
                    down._fl_bge_quant_plan,
                    self.output.LayerNorm.weight,
                    self.output.LayerNorm.bias,
                    self.output.LayerNorm.eps,
                    norm_threads=threads,
                )
                self._fl_bge_ffn = PreparedW8A8I8MMFFN(
                    up._fl_bge_quant_plan, norm, tables[key]
                )
            return self._fl_bge_ffn(
                x, self.intermediate.dense.bias, self.output.dense.bias
            )

        layer.forward = MethodType(forward, layer)
