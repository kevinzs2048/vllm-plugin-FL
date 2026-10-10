# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in vLLM 0.24 ARM CPU MiniCPM lifecycle compatibility."""
import gc
from importlib import metadata

import torch

_ACTIVE = False


def install():
    global _ACTIVE
    if _ACTIVE:
        return False
    version = metadata.version('vllm')
    if version.split('+',1)[0] != '0.24.0':
        raise RuntimeError(f'ARM MiniCPM runtime requires vLLM 0.24.0; found {version}')
    _install_cpu_gemm_guard()
    _install_cpu_cleanup_guard()
    _install_cpu_hybrid_kv_zeroing()
    _install_cpu_sampling_rng()
    from vllm.utils.cpu_triton_utils import compute_slot_mapping_kernel
    from vllm.v1.worker import block_table
    block_table._compute_slot_mapping_kernel = compute_slot_mapping_kernel
    from vllm_fl.quantization.arm_cpu_native import install_w8a8
    from vllm_fl.patches.minicpmv47_gdn import install_vllm_gdn_bridge
    from vllm_fl.ops.minicpmv47_moe import install as install_moe
    install_w8a8()
    install_moe()
    install_vllm_gdn_bridge()
    from vllm_fl.patches.arm_gemma_norm import install as install_gemma_norm
    install_gemma_norm()
    _ACTIVE = True
    return True


def _install_cpu_gemm_guard():
    from vllm.model_executor.layers import utils
    original = utils.dispatch_cpu_unquantized_gemm
    if getattr(original, '_fl_arm_ndim_guard', False):
        return

    def dispatch(layer, remove_weight):
        weight = getattr(layer, 'weight', None)
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        import os
        if (isinstance(layer, ParallelLMHead) and os.getenv('FL_INT8_LMHEAD') == '1'
                and isinstance(weight,torch.Tensor) and not weight.is_meta):
            from flag_gems.runtime.backend._arm.fused.w8a8_prepare import prepare_w8a8_weight
            from vllm_fl.ops.arm_operator_backend import flaggems_ops
            operators = flaggems_ops('w8a8')
            n,k = weight.shape
            packed = prepare_w8a8_weight(weight)
            layer._fl_arm_head_packed = torch.nn.Parameter(packed,requires_grad=False)

            def linear(x, unused_weight, bias):
                if x.numel() == 0:
                    return x.new_empty((*x.shape[:-1],n))
                output = operators.linear(x.reshape(-1,k).float().contiguous(),
                    layer._fl_arm_head_packed,n,False,32,128)
                output = output.to(x.dtype).reshape(*x.shape[:-1],n)
                return output if bias is None else output+bias.to(output.dtype)

            layer.cpu_linear = linear
            if remove_weight:
                layer.weight = None
            return
        if isinstance(weight, torch.Tensor) and weight.ndim != 2:
            return
        return original(layer, remove_weight)

    dispatch._fl_arm_ndim_guard = True
    utils.dispatch_cpu_unquantized_gemm = dispatch

def _install_cpu_cleanup_guard() -> None:
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    original = CPUModelRunner._cleanup_profiling_kv_cache
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
    CPUModelRunner._cleanup_profiling_kv_cache = cleanup


def _zero_cpu_attention_blocks(self, block_ids: list[int]) -> None:
    """Clear recycled attention pages for vLLM's hybrid CPU cache.

    vLLM 0.20.2 asks the runner to clear newly allocated pages whenever a
    model has Mamba/GDN layers.  Its CPU runner discards that request, based
    on the assumption that invalid attention positions are always masked.
    With a hybrid page size larger than the CPU attention kernel block, stale
    K/V data can still affect a recycled physical page.  The generic zeroer
    already knows the hybrid page mapping and intentionally skips Mamba state;
    GDN handles fresh state through ``has_initial_state`` during prefill.
    """
    zeroer = getattr(self, "_kv_block_zeroer", None)
    if zeroer is not None:
        zeroer.zero_block_ids(block_ids)


def _install_cpu_hybrid_kv_zeroing() -> None:
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    current = CPUModelRunner._zero_block_ids
    if getattr(current, "_vllm_fl_hybrid_kv_zeroing", False):
        return
    _zero_cpu_attention_blocks._vllm_fl_hybrid_kv_zeroing = True
    CPUModelRunner._zero_block_ids = _zero_cpu_attention_blocks


def _install_cpu_sampling_rng() -> None:
    """Undo discarded prefill draws with CPU RNG state, rather than CUDA offsets."""
    from functools import wraps
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    if getattr(CPUModelRunner, '_vllm_fl_cpu_sampling_rng', False):
        return
    original_sample = CPUModelRunner._sample
    original_bookkeeping = CPUModelRunner._bookkeeping_sync

    def discarded_generators(runner):
        mask = runner.discard_request_mask.np[:runner.input_batch.num_reqs]
        return {
            index: gen for index, gen in runner.input_batch.generators.items()
            if index < len(mask) and mask[index] and gen.device.type == 'cpu'
        }

    @wraps(original_sample)
    def sample(self, *args, **kwargs):
        # Only discarded requests need a snapshot. Valid decode draws retain
        # their normal progression, including in mixed prefill/decode batches.
        snapshots = [(gen, gen.get_state()) for gen in discarded_generators(self).values()]
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            for gen, state in snapshots:
                gen.set_state(state)

    @wraps(original_bookkeeping)
    def bookkeeping(self, *args, **kwargs):
        # The inherited method rewinds CUDA generators by four offsets. CPU
        # draws have already been restored above; exclude only those generators
        # from that rewind while preserving upstream token/logprob bookkeeping.
        generators = self.input_batch.generators
        restored = discarded_generators(self)
        for index in restored:
            del generators[index]
        try:
            return original_bookkeeping(self, *args, **kwargs)
        finally:
            generators.update(restored)

    CPUModelRunner._sample = sample
    CPUModelRunner._bookkeeping_sync = bookkeeping
    CPUModelRunner._vllm_fl_cpu_sampling_rng = True
