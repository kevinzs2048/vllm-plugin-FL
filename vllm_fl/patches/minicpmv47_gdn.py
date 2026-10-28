"""Thin vLLM bridge to the FlagGems Qwen GDN integration."""

from __future__ import annotations


def install_vllm_gdn_bridge() -> None:
    """Install the version-adapted FlagGems GDN runtime lazily."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    if getattr(gdn, "_minicpm47_compat_installed", False):
        return
    from flag_gems.runtime.backend._arm.gdn import inference as kernels

    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    # Load the chosen namespace before binding callbacks. Generic tensor
    # arithmetic belongs to FlagGems; vLLM class/cache ABI belongs here.
    flaggems_ops("gdn")
    import os

    kernels._operators(os.environ.get("FL_ARM_OPERATOR_BACKEND", "raw"))
    os.environ["FLAGGEMS_ARM_GDN_BACKEND"] = os.environ.get(
        "FL_ARM_OPERATOR_BACKEND", "raw"
    )
    gdn.causal_conv1d_fn = kernels.torch_causal_conv1d_fn
    gdn.causal_conv1d_update = kernels.torch_causal_conv1d_update
    gdn.fla_chunk_gated_delta_rule = kernels.torch_chunk_gated_delta_rule
    gdn.fused_sigmoid_gating_delta_rule_update = (
        kernels.torch_fused_sigmoid_gating_delta_rule_update
    )
    _install_cpu_gated_norm()

    # Preserve the existing plugin policy; operator routing stays in FlagGems.
    os.environ.setdefault("FLAGGEMS_GDN_CONV_PREFILL_TRITON", "1")
    conv_mode = os.environ.get("MINICPM_GDN_CONV_PREFILL_IMPL")
    if conv_mode is not None:
        if conv_mode not in ("native", "triton"):
            raise ValueError("MINICPM_GDN_CONV_PREFILL_IMPL must be native or triton")
        os.environ["FLAGGEMS_GDN_CONV_PREFILL_TRITON"] = str(int(conv_mode == "triton"))

    # ChunkGatedDeltaRule.forward_native in 0.24 can provide a preallocated
    # output buffer.  The existing CPU implementation returns the same tensor
    # instead, so bridge the new optional ABI and preserve both behaviours.
    flag_gems_chunk_gated_delta_rule = gdn.fla_chunk_gated_delta_rule

    def chunk_gated_delta_rule_024(*args, core_attn_out=None, **kwargs):
        output, final_state = flag_gems_chunk_gated_delta_rule(*args, **kwargs)
        if core_attn_out is not None:
            output_flat = output.squeeze(0).reshape(-1)
            target_flat = core_attn_out.reshape(-1)
            target_flat[: output_flat.numel()].copy_(output_flat)
        return output, final_state

    gdn.fla_chunk_gated_delta_rule = chunk_gated_delta_rule_024

    # vLLM 0.24 selects a new CPU-only GDN implementation in ``__init__``.
    # That path bypasses the FlagGems functions installed above and explicitly
    # rejects speculative metadata.  Keep using the generic Qwen GDN core on
    # Apple ARM: despite the historical ``forward_cuda`` name, the patched
    # implementation dispatches only CPU tensors to FlagGems/libtriton_jit.
    # Install this after FlagGems so we retain its projection setup wrapper.
    original_init = gdn.QwenGatedDeltaNetAttention.__init__

    def install_arm_cpu_forward(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        self._forward_method = self.forward_cuda

    gdn.QwenGatedDeltaNetAttention.__init__ = install_arm_cpu_forward

    # vLLM 0.24 passes an additional token-count argument to this hook.
    # FlagGems intentionally disables it because the accelerator JIT warmup
    # is neither needed nor valid for the Apple CPU runtime.
    def no_op_prefill_warmup(self, *args, **kwargs) -> None:
        return None

    gdn.QwenGatedDeltaNetAttention._warmup_prefill_kernels = no_op_prefill_warmup
    _install_fused_cpu_decode(gdn)
    gdn._minicpm47_compat_installed = True


def _install_cpu_gated_norm():
    """Retain the validated gated-norm fast path for prefill and fallback decode."""
    import os

    import torch
    from vllm.model_executor.layers.layernorm import RMSNormGated

    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    original = RMSNormGated.forward_cpu
    if getattr(original, "_fl_arm_gated_norm", False):
        return False
    operators = flaggems_ops("gdn")

    def forward(self, x, z=None):
        if (
            os.getenv("FLAGGEMS_GDN_NATIVE_NORM", "1") == "1"
            and z is not None
            and self.group_size is None
            and self.norm_before_gate
            and self.activation in ("silu", "swish", "sigmoid")
            and x.device.type == z.device.type == self.weight.device.type == "cpu"
            and x.dtype == z.dtype == self.weight.dtype == torch.bfloat16
            and x.shape == z.shape
            and x.shape[-1] == self.weight.numel()
            and x.is_contiguous()
            and z.is_contiguous()
            and self.weight.is_contiguous()
        ):
            return operators.gdn_rmsnorm_gated(
                x, self.weight, z, self.eps, self.activation == "sigmoid"
            )
        return original(self, x, z)

    forward._fl_arm_gated_norm = True
    RMSNormGated.forward_cpu = forward
    return True


def _install_fused_cpu_decode(gdn) -> None:
    """Use FlagGems' existing recurrence-plus-gated-norm op for CPU decode."""
    import os

    import torch

    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    operators = flaggems_ops("gdn")
    original = gdn.QwenGatedDeltaNetAttention.forward_cuda

    def fused_decode(self, hidden_states, output):
        norm = self.norm
        if (
            os.getenv("MINICPM_GDN_FUSED_NORM", "0") != "1"
            or os.getenv("FLAGGEMS_GDN_TRITON_DECODE", "0") == "1"
            or os.getenv("FLAGGEMS_GDN_NATIVE_NORM", "1") != "1"
            or hidden_states.device.type != "cpu"
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or hidden_states.shape[0] != 1
            or not hasattr(self, "in_proj_qkvz")
            or self.tp_size != 1
            or norm.group_size is not None
            or not norm.norm_before_gate
            or norm.activation not in ("silu", "swish", "sigmoid")
            or norm.weight.dtype != torch.bfloat16
            or not norm.weight.is_contiguous()
        ):
            return original(self, hidden_states, output)
        by_layer = gdn.get_forward_context().attn_metadata
        metadata = by_layer.get(self.prefix) if isinstance(by_layer, dict) else None
        if (
            not isinstance(metadata, gdn.GDNAttentionMetadata)
            or not self.enable_packed_recurrent_decode
            or metadata.spec_sequence_masks is not None
            or metadata.num_prefills != 0
            or metadata.num_decodes <= 0
            or metadata.num_actual_tokens != 1
            or metadata.non_spec_state_indices_tensor is None
        ):
            return original(self, hidden_states, output)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        mixed_qkv, z = mixed_qkvz.split(
            [self.key_dim * 2 + self.value_dim, self.value_dim], dim=-1
        )
        z = z.reshape(1, -1, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
        conv_state = self.kv_cache[0]
        if not gdn.is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weight = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        indices = (
            metadata.non_spec_state_indices_tensor[:1]
            .to(dtype=torch.int32)
            .contiguous()
        )
        normalized = torch.empty_like(z)
        packed_decode_norm = operators.gdn_packed_decode_norm
        packed_decode_norm(
            mixed_qkv.contiguous(),
            a.contiguous(),
            b.contiguous(),
            self.A_log,
            self.dt_bias,
            conv_state,
            conv_weight,
            self.conv1d.bias,
            self.kv_cache[1],
            indices,
            z.contiguous(),
            norm.weight,
            norm.eps,
            norm.activation == "sigmoid",
            normalized,
            True,
        )
        output[:1], _ = self.out_proj(normalized.reshape(1, -1))

    gdn.QwenGatedDeltaNetAttention.forward_cuda = fused_decode


__all__ = ["install_vllm_gdn_bridge"]
