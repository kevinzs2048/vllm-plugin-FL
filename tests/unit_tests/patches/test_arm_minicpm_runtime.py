# Copyright (c) 2026 BAAI. All rights reserved.
"""Arm MiniCPM runtime: norm dispatch/aliasing and chunked sampling RNG."""

import os
import platform
import unittest
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_fl.patches.arm_minicpm_runtime import _install_cpu_sampling_rng

requires_arm_operators = pytest.mark.skipif(
    platform.machine().lower() not in ("arm64", "aarch64")
    or not os.getenv("FLAGGEMS_ARM_NATIVE_LIBRARY"),
    reason="requires the optional FlagGems ARM native build",
)


@requires_arm_operators
@pytest.mark.parametrize("backend", ["native", "raw"])
@pytest.mark.parametrize("hidden", [1024, 2048])
@pytest.mark.parametrize("gemma_backend", ["inherit", "triton"])
def test_gemma_cpu_dispatch_and_aliasing(monkeypatch, backend, hidden, gemma_backend):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    from vllm_fl.ops.arm_operator_backend import flaggems_ops
    from vllm_fl.patches.arm_gemma_norm import install

    monkeypatch.setenv("FL_ARM_OPERATOR_BACKEND", backend)
    monkeypatch.setenv("FL_ARM_GEMMA_NORM", "1")
    monkeypatch.setenv("FL_ARM_GEMMA_BACKEND", gemma_backend)
    original = GemmaRMSNorm.forward_cpu
    monkeypatch.setattr(GemmaRMSNorm, "forward_cpu", original)
    assert install()
    assert not install()
    with set_current_vllm_config(VllmConfig()):
        layer = GemmaRMSNorm(hidden).to(dtype=torch.bfloat16)
    torch.manual_seed(191)
    layer.weight.data.uniform_(-0.2, 0.2)
    x = torch.randn(3, hidden, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    ops = flaggems_ops("gdn")
    assert torch.equal(
        layer.forward_cpu(x), ops.gemma_rmsnorm(x, layer.weight, layer.variance_epsilon)
    )
    assert torch.equal(layer(x), layer.forward_cpu(x))
    expected, expected_state = x.clone(), residual.clone()
    ops.gemma_fused_add_rmsnorm(
        expected, expected_state, layer.weight, layer.variance_epsilon
    )
    output, state = layer.forward_cpu(x.clone(), residual)
    assert state is residual
    assert torch.equal(output, expected) and torch.equal(state, expected_state)
    # Preserve the stock implementation for unsupported tensor layouts/dtypes.
    value = x.float()
    assert torch.equal(layer.forward_cpu(value), original(layer, value))
    noncontiguous = x.T.contiguous().T
    assert not noncontiguous.is_contiguous()
    assert torch.equal(layer.forward_cpu(noncontiguous), original(layer, noncontiguous))
    monkeypatch.setenv("FL_ARM_GEMMA_NORM", "0")
    assert torch.equal(layer.forward_cpu(x), original(layer, x))
    monkeypatch.setenv("FL_ARM_GEMMA_NORM", "1")
    monkeypatch.setenv("FLAGGEMS_GDN_NATIVE_NORM", "0")
    assert torch.equal(layer.forward_cpu(x), original(layer, x))


@requires_arm_operators
@pytest.mark.parametrize("backend", ["native", "raw"])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_gated_norm_bridge_and_unsupported_group_fallback(
    monkeypatch, backend, activation
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNormGated

    from vllm_fl.patches.minicpmv47_gdn import _install_cpu_gated_norm

    monkeypatch.setenv("FL_ARM_OPERATOR_BACKEND", backend)
    monkeypatch.setenv("FLAGGEMS_GDN_NATIVE_NORM", "1")
    # Restore the class after each test, including the installed wrapper.
    monkeypatch.setattr(RMSNormGated, "forward_cpu", RMSNormGated.forward_cpu)
    assert _install_cpu_gated_norm()
    assert not _install_cpu_gated_norm()
    torch.manual_seed(144)
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNormGated(
            64,
            dtype=torch.bfloat16,
            device="cpu",
            norm_before_gate=True,
            activation=activation,
        )
    layer.weight.data.uniform_(0.2, 1.5)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    z = torch.randn_like(x)
    expected = layer.forward_native(x, z)
    torch.testing.assert_close(layer.forward_cpu(x, z), expected, rtol=0.01, atol=0.01)
    layer.group_size = 16
    assert torch.equal(layer.forward_cpu(x, z), layer.forward_native(x, z))
    layer.group_size = None
    layer.norm_before_gate = False
    assert torch.equal(layer.forward_cpu(x, z), layer.forward_native(x, z))


class TestCPUChunkedSampling(unittest.TestCase):
    def test_discarded_chunk_and_mixed_decode(self):
        gpu_sample = GPUModelRunner._sample
        gpu_bookkeeping = GPUModelRunner._bookkeeping_sync
        _install_cpu_sampling_rng()
        runner = object.__new__(CPUModelRunner)
        generators = {i: torch.Generator().manual_seed(1234 + i) for i in range(2)}
        baseline = {i: gen.get_state().clone() for i, gen in generators.items()}
        runner.input_batch = NS(
            num_reqs=2,
            generators=generators,
            sampling_metadata=NS(generators=generators),
            update_async_output_token_ids=lambda: None,
            req_ids=["prefill", "decode"],
            req_id_to_index={"prefill": 0, "decode": 1},
            token_ids_cpu=np.zeros((2, 8), dtype=np.int64),
            is_token_ids=np.zeros((2, 8), dtype=bool),
            num_tokens_no_spec=np.array([0, 0]),
        )
        runner.discard_request_mask = NS(np=np.array([True, False]))
        runner.use_async_scheduling = False
        runner.routed_experts_initialized = False
        runner.max_model_len = 8
        runner.requests = {
            key: NS(output_token_ids=[]) for key in runner.input_batch.req_ids
        }
        runner._to_list = lambda tensor: tensor.tolist()
        runner._get_prompt_logprobs_dict = lambda *args: {}

        def sampler(logits, sampling_metadata):
            tokens = [
                torch.multinomial(row.softmax(-1), 1, generator=generators[i])
                for i, row in enumerate(logits)
            ]
            return NS(sampled_token_ids=torch.stack(tokens), logprobs_tensors=None)

        runner.sampler = sampler
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        expected_gen = torch.Generator().set_state(baseline[1])
        for _ in range(2):
            output = runner._sample(logits, None)
            expected = torch.multinomial(
                logits[1].softmax(-1), 1, generator=expected_gen
            ).item()
            result = runner._bookkeeping_sync(
                NS(num_scheduled_tokens={}), output, logits, torch.zeros((2, 1)), 2
            )
            self.assertEqual(result[2], [[], [expected]])
            self.assertTrue(torch.equal(generators[0].get_state(), baseline[0]))
            self.assertTrue(
                torch.equal(generators[1].get_state(), expected_gen.get_state())
            )
            self.assertEqual(set(generators), {0, 1})
        runner.discard_request_mask.np[:] = False
        output = runner._sample(logits, None)
        fresh = torch.Generator().set_state(baseline[0])
        expected = torch.multinomial(logits[0].softmax(-1), 1, generator=fresh).item()
        self.assertEqual(output.sampled_token_ids[0].item(), expected)
        self.assertTrue(torch.equal(generators[0].get_state(), fresh.get_state()))
        self.assertIs(GPUModelRunner._sample, gpu_sample)
        self.assertIs(GPUModelRunner._bookkeeping_sync, gpu_bookkeeping)
