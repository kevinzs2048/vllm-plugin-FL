# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU loading hooks for the shared MiniCPM-V 4.7 implementation.

No second model/processor registry or vision implementation is maintained here.
"""

import os
import threading
from functools import wraps

_lock = threading.Lock()


def _has_w4_experts(model):
    for block in model.language_model.model.layers:
        experts = getattr(getattr(block, "mlp", None), "experts", None)
        routed = getattr(experts, "routed_experts", None)
        method = getattr(routed, "quant_method", None)
        if type(method).__name__ == "CompressedTensorsW4A8Int8MoEMethod":
            return True
    return False


def install_cpu_model_hooks(model_cls=None):
    if model_cls is None:
        from vllm_fl.models.minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration

        model_cls = MiniCPMV4_7ForConditionalGeneration
    with _lock:
        if getattr(model_cls, "_fl_arm_loading_hooks", False):
            return False
        original = model_cls.load_weights

        @wraps(original)
        def load_weights(self, weights):
            stream = None
            w4_experts = _has_w4_experts(self)
            if os.getenv("MINICPM_STREAM_EXPERT_LOADING") == "1":
                if not w4_experts:
                    raise ValueError(
                        "Streamed expert loading requires CPU W4A8 experts"
                    )
                from vllm_fl.quantization.streamed_experts import StreamedExperts

                stream = StreamedExperts(self)
                weights = stream.weights(weights)
            elif w4_experts:
                from vllm_fl.quantization.streamed_experts import unpack_expert_weights

                weights = unpack_expert_weights(weights)
            loaded = original(self, weights)
            if stream is not None:
                loaded.update(stream.loaded_names)
                self.streamed_loading_report = stream.report
            if w4_experts and os.getenv("FL_ARM_NATIVE") == "1":
                from vllm_fl.ops.minicpmv47_moe import (
                    install,
                    install_direct_model_moe,
                )

                install()
                install_direct_model_moe(self)
            return loaded

        @classmethod
        def state_dtype(cls, vllm_config):
            from vllm.model_executor.layers.mamba.mamba_utils import (
                MambaStateDtypeCalculator,
            )

            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                "float32",
            )

        model_cls.load_weights = load_weights
        model_cls.get_mamba_state_dtype_from_config = state_dtype
        model_cls._fl_arm_loading_hooks = True
    return True
