"""Route supported MiniCPM CPU prefill calls to packaged FlagGems operators."""

from __future__ import annotations

import os
from collections import Counter
from functools import wraps

CALLS = Counter()


def install_candidate_gdn_prefill(gdn) -> None:
    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    candidate = flaggems_ops("gdn")
    import torch

    # The existing CPU runtime uses Triton for this convolution by default.
    # Keep that policy when switching operator ownership to FlagGems; the
    # native C++ convolution remains an explicit diagnostic alternative.
    triton_enabled = os.environ.get(
        "FLAGGEMS_GDN_CONV_PREFILL_TRITON", "1"
    ).lower() not in ("0", "false", "off")
    conv_mode = os.environ.get(
        "MINICPM_GDN_CONV_PREFILL_IMPL", "triton" if triton_enabled else "native"
    )
    if conv_mode not in ("native", "triton"):
        raise ValueError("MINICPM_GDN_CONV_PREFILL_IMPL must be native or triton")
    original_conv = gdn.causal_conv1d_fn
    original_recurrence = gdn.fla_chunk_gated_delta_rule

    def enabled():
        return os.environ.get("FLAGGEMS_GDN_NATIVE_PREFILL", "1").lower() not in (
            "0", "false", "off"
        )

    @wraps(original_conv)
    def conv(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=None, has_initial_state=None, activation="silu",
        pad_slot_id=-1, block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None, initial_state_idx=None,
        num_computed_tokens=None, block_size_to_align=0, metadata=None,
        validate_data=False,
    ):
        supported = (
            enabled() and x.device.type == "cpu" and x.dtype == torch.bfloat16
            and x.ndim == 2 and weight.ndim == 2
            and block_idx_last_scheduled_token is None
            and initial_state_idx is None and cache_indices is not None
            and cache_indices.ndim == 1 and has_initial_state is not None
            and query_start_loc.dtype == torch.int32
            and cache_indices.dtype == torch.int32
            and activation in ("silu", "swish")
        )
        if not supported:
            CALLS["conv_legacy_fallback"] += 1
            return original_conv(
                x, weight, bias, conv_states, query_start_loc, cache_indices,
                has_initial_state, activation, pad_slot_id,
                block_idx_first_scheduled_token, block_idx_last_scheduled_token,
                initial_state_idx, num_computed_tokens, block_size_to_align,
                metadata, validate_data,
            )
        args = (
            x, conv_states, weight, bias, query_start_loc.contiguous(),
            cache_indices.contiguous(), has_initial_state.contiguous(), True,
        )
        if conv_mode == "triton" and weight.shape[1] == 4:
            from flag_gems.runtime.backend._arm.gdn.conv_prefill import (
                gdn_conv1d_prefill_triton,
            )
            output = gdn_conv1d_prefill_triton(*args)
            CALLS["conv_triton"] += 1
            return output
        output = candidate.gdn_conv1d_prefill(*args)
        CALLS["conv_candidate"] += 1
        return output

    @wraps(original_recurrence)
    def recurrence(
        q, k, v, g, beta, initial_state, output_final_state,
        cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
        use_qk_l2norm_in_kernel=True,
    ):
        supported = (
            enabled() and q.device.type == "cpu" and q.ndim == 4
            and q.shape[0] == 1 and output_final_state
            and initial_state is not None and initial_state.dtype == torch.float32
            and cu_seqlens is not None and cu_seqlens.dtype == torch.int32
        )
        if not supported:
            CALLS["recurrence_legacy_fallback"] += 1
            return original_recurrence(
                q, k, v, g, beta, initial_state, output_final_state,
                cu_seqlens, chunk_indices, chunk_offsets,
                use_qk_l2norm_in_kernel,
            )
        state = initial_state.contiguous()
        output = candidate.gdn_prefill(
            q, k, v, g, beta, state, cu_seqlens.contiguous(),
            use_qk_l2norm_in_kernel,
        )
        CALLS["recurrence_candidate"] += 1
        return output, state

    # The 0.24 core_attn_out compatibility wrapper is installed by the caller
    # after these functions. Exceptions from candidate kernels propagate.
    gdn.causal_conv1d_fn = conv
    gdn.fla_chunk_gated_delta_rule = recurrence
