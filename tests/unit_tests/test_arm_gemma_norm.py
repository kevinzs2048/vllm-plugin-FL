# Copyright (c) 2026 BAAI. All rights reserved.
import os
import platform

import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ('arm64', 'aarch64')
    or not os.getenv('FLAGGEMS_ARM_NATIVE_LIBRARY'),
    reason='requires the optional FlagGems ARM native build',
)


@pytest.mark.parametrize('backend', ['native', 'raw'])
@pytest.mark.parametrize('hidden', [1024, 2048])
def test_gemma_cpu_dispatch_and_aliasing(monkeypatch, backend, hidden):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm_fl.patches.arm_gemma_norm import install
    from vllm_fl.ops.arm_operator_backend import flaggems_ops

    monkeypatch.setenv('FL_ARM_OPERATOR_BACKEND', backend)
    monkeypatch.setenv('FL_ARM_GEMMA_NORM', '1')
    original = GemmaRMSNorm.forward_cpu
    monkeypatch.setattr(GemmaRMSNorm, 'forward_cpu', original)
    assert install()
    assert not install()
    with set_current_vllm_config(VllmConfig()):
        layer = GemmaRMSNorm(hidden).to(dtype=torch.bfloat16)
    torch.manual_seed(191)
    layer.weight.data.uniform_(-.2, .2)
    x = torch.randn(3, hidden, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    ops = flaggems_ops('gdn')
    assert torch.equal(layer.forward_cpu(x),
                       ops.gemma_rmsnorm(x, layer.weight, layer.variance_epsilon))
    assert torch.equal(layer(x), layer.forward_cpu(x))
    expected, expected_state = x.clone(), residual.clone()
    ops.gemma_fused_add_rmsnorm(expected, expected_state,
                              layer.weight, layer.variance_epsilon)
    output, state = layer.forward_cpu(x.clone(), residual)
    assert state is residual
    assert torch.equal(output, expected) and torch.equal(state, expected_state)
    # Preserve the stock implementation for unsupported tensor layouts/dtypes.
    value = x.float()
    assert torch.equal(layer.forward_cpu(value), original(layer, value))
    noncontiguous = x.T.contiguous().T
    assert not noncontiguous.is_contiguous()
    assert torch.equal(layer.forward_cpu(noncontiguous), original(layer, noncontiguous))
    monkeypatch.setenv('FL_ARM_GEMMA_NORM', '0')
    assert torch.equal(layer.forward_cpu(x), original(layer, x))
    monkeypatch.setenv('FL_ARM_GEMMA_NORM', '1')
    monkeypatch.setenv('FLAGGEMS_GDN_NATIVE_NORM', '0')
    assert torch.equal(layer.forward_cpu(x), original(layer, x))
