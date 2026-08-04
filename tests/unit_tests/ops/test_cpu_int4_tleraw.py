# Copyright (c) 2026 BAAI. All rights reserved.

import platform

import pytest
import torch
from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.scalar_type import scalar_types


pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"aarch64", "arm64"},
    reason="ARM TLE-raw W4A8 requires AArch64",
)


@pytest.fixture(scope="module")
def int4_case():
    from vllm_fl.ops import cpu_int4_tleraw as int4

    torch.manual_seed(19)
    n, k = 64, 128
    checkpoint_weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    checkpoint_scale = torch.rand(n, 1, dtype=torch.bfloat16) * 0.1 + 1e-3
    rhs = int4._pack_rhs(checkpoint_weight, checkpoint_scale)
    dequantized = checkpoint_weight.float() * checkpoint_scale.float()
    return (
        int4,
        rhs,
        checkpoint_weight,
        checkpoint_scale,
        dequantized,
        n,
        k,
    )


@pytest.mark.parametrize("m", [1, 3, 7, 128, 129, 512])
def test_linear_matches_dequantized_reference(int4_case, m):
    int4, rhs, _, _, weight, n, k = int4_case
    torch.manual_seed(100 + m)
    x = torch.randn(m, k, dtype=torch.bfloat16)

    actual = int4.linear_w4a8(x, rhs, n, k).float()
    expected = x.float() @ weight.T

    relative_error = torch.linalg.vector_norm(actual - expected) / (
        torch.linalg.vector_norm(expected) + 1e-12
    )
    assert relative_error.item() < 0.015


@pytest.mark.slow
def test_compiled_prefill_graph_handles_decode_shape(int4_case):
    int4, rhs, _, _, _, n, k = int4_case

    def linear(x):
        return int4.linear_w4a8(x, rhs, n, k)

    compiled = torch.compile(linear, fullgraph=True, dynamic=True)
    for m in (7, 3, 1, 7, 1):
        torch.manual_seed(200 + m)
        x = torch.randn(m, k, dtype=torch.bfloat16)
        torch.testing.assert_close(compiled(x), linear(x), rtol=0, atol=0)


def _config(n=64, k=128, group_size=None, act_type=torch.bfloat16):
    return MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.int4,
        act_type=act_type,
        group_size=k if group_size is None else group_size,
        zero_points=False,
        has_g_idx=False,
    )


def test_checkpoint_kernel_accepts_only_bf16_channelwise_int4():
    from vllm_fl.ops.cpu_int4_tleraw import (
        FlagTreeKleidiAIInt4LinearKernel as kernel,
    )

    assert kernel.can_implement(_config()) == (True, None)
    assert kernel.can_implement(_config(group_size=32))[0] is False
    assert kernel.can_implement(_config(act_type=torch.float32))[0] is False


def test_checkpoint_kernel_consumes_loaded_weight_without_requantizing(int4_case):
    int4, rhs, weight, scale, _, n, k = int4_case
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight_packed", torch.nn.Parameter(weight.clone(), requires_grad=False)
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(scale.clone(), requires_grad=False)
    )
    kernel = int4.FlagTreeKleidiAIInt4LinearKernel(
        _config(n, k), "weight_packed", "weight_scale"
    )

    kernel.process_weights_after_loading(layer)

    assert layer.weight_packed is None
    assert layer.weight_scale is None
    assert "_fl_w4a8_packed_rhs" not in layer.state_dict()
    torch.testing.assert_close(layer._fl_w4a8_packed_rhs, rhs, rtol=0, atol=0)

    x = torch.randn(3, k, dtype=torch.bfloat16)
    torch.testing.assert_close(
        kernel.apply_weights(layer, x),
        int4.linear_w4a8(x, rhs, n, k),
        rtol=0,
        atol=0,
    )


def test_online_group32_quantizer_is_removed():
    from vllm_fl.ops import cpu_int4_tleraw as int4

    assert not hasattr(int4, "quant_native_qs4c32")
    assert not hasattr(int4, "_prepare_linear")
