# Copyright (c) 2026 BAAI. All rights reserved.
import pytest
import platform
import os
import torch


pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in ('arm64','aarch64')
    or not os.getenv('FLAGGEMS_ARM_NATIVE_LIBRARY'),
    reason='requires the optional FlagGems ARM native build',
)


@pytest.mark.parametrize('backend', ['native','raw'])
@pytest.mark.parametrize('activation', ['silu','sigmoid'])
def test_gated_norm_bridge_and_unsupported_group_fallback(monkeypatch,backend,activation):
    from vllm.model_executor.layers.layernorm import RMSNormGated
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_fl.patches.minicpmv47_gdn import _install_cpu_gated_norm
    monkeypatch.setenv('FL_ARM_OPERATOR_BACKEND',backend)
    monkeypatch.setenv('FLAGGEMS_GDN_NATIVE_NORM','1')
    # Restore the class after each test, including the installed wrapper.
    monkeypatch.setattr(RMSNormGated,'forward_cpu',RMSNormGated.forward_cpu)
    assert _install_cpu_gated_norm()
    assert not _install_cpu_gated_norm()
    torch.manual_seed(144)
    with set_current_vllm_config(VllmConfig()):
        layer = RMSNormGated(64,dtype=torch.bfloat16,device='cpu',
            norm_before_gate=True,activation=activation)
    layer.weight.data.uniform_(.2,1.5)
    x = torch.randn(3,64,dtype=torch.bfloat16)
    z = torch.randn_like(x)
    expected = layer.forward_native(x,z)
    torch.testing.assert_close(layer.forward_cpu(x,z),expected,rtol=.01,atol=.01)
    layer.group_size = 16
    assert torch.equal(layer.forward_cpu(x,z),layer.forward_native(x,z))
    layer.group_size = None
    layer.norm_before_gate = False
    assert torch.equal(layer.forward_cpu(x,z),layer.forward_native(x,z))
