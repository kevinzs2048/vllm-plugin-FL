# Copyright (c) 2026 BAAI. All rights reserved.
"""Optional CPU Gemma norm binding; tensor arithmetic belongs to FlagGems."""
import os

import torch


def install():
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    original = GemmaRMSNorm.forward_cpu
    if getattr(original, '_fl_arm_gemma_norm', False):
        return False
    operators = flaggems_ops('gdn')

    def forward(self, x, residual=None):
        weight = self.weight
        eligible = (
            os.getenv('FL_ARM_GEMMA_NORM', '1') == '1'
            and os.getenv('FLAGGEMS_GDN_NATIVE_NORM', '1') == '1'
            and x.device.type == weight.device.type == 'cpu'
            and x.dtype == weight.dtype == torch.bfloat16
            and x.ndim >= 1 and x.shape[-1] > 0
            and weight.ndim == 1 and x.shape[-1] == weight.numel()
            and x.is_contiguous() and weight.is_contiguous()
        )
        if residual is not None:
            eligible = eligible and (
                residual.device == x.device and residual.dtype == x.dtype
                and residual.shape == x.shape and residual.is_contiguous()
            )
        if not eligible:
            return original(self, x, residual)
        if residual is None:
            return operators.gemma_rmsnorm(x, weight, self.variance_epsilon)
        # Match the previously validated ARM path: round the residual sum to
        # BF16 before its norm, and preserve the fused in-place tensor contract.
        operators.gemma_fused_add_rmsnorm(
            x, residual, weight, self.variance_epsilon)
        return x, residual

    forward._fl_arm_gemma_norm = True
    GemmaRMSNorm.forward_cpu = forward
    return True
