# Copyright (c) 2026 BAAI. All rights reserved.

import platform

import pytest
import torch


pytestmark = pytest.mark.skipif(
    platform.machine().lower() not in {"aarch64", "arm64"},
    reason="ARM TLE-raw W4A8 requires AArch64",
)


@pytest.fixture(scope="module")
def int4_case():
    from vllm_fl.ops import cpu_int4_tleraw as int4

    torch.manual_seed(19)
    n, k = 64, 128
    weight = torch.randn(n, k, dtype=torch.bfloat16)
    native, scales = int4.quant_native_qs4c32(weight)
    rhs = int4._pack_rhs(native, scales, n, k)

    unsigned = torch.empty((n, k), dtype=torch.uint8)
    unsigned[:, 0::2] = native & 0xF
    unsigned[:, 1::2] = native >> 4
    dequantized = (unsigned.float() - 8) * scales.float().repeat_interleave(
        int4.BL, dim=1
    )
    return int4, rhs, dequantized, n, k


@pytest.mark.parametrize("m", [1, 3, 7])
def test_linear_matches_dequantized_reference(int4_case, m):
    int4, rhs, weight, n, k = int4_case
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
    int4, rhs, _, n, k = int4_case

    def linear(x):
        return int4.linear_w4a8(x, rhs, n, k)

    compiled = torch.compile(linear, fullgraph=True, dynamic=True)
    for m in (7, 3, 1, 7, 1):
        torch.manual_seed(200 + m)
        x = torch.randn(m, k, dtype=torch.bfloat16)
        torch.testing.assert_close(compiled(x), linear(x), rtol=0, atol=0)
