# Copyright (c) 2026 BAAI. All rights reserved.

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


IS_ARM64 = platform.machine().lower() in {"aarch64", "arm64"}
REPO_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(IS_ARM64, "ARM quantized CPU backends require AArch64")
class TestCpuInt8Backends(unittest.TestCase):
    def run_in_fresh_cache(self, code):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["OMP_NUM_THREADS"] = "2"
        # GNU as on this target does not accept LLVM's two-string .file form.
        env["TRITON_DISABLE_LINE_INFO"] = "1"
        with (
            tempfile.TemporaryDirectory(prefix="fl-triton-cache-") as triton_cache,
            tempfile.TemporaryDirectory(prefix="fl-inductor-cache-") as inductor_cache,
            tempfile.TemporaryDirectory(prefix="fl-kleidiai-cache-") as kai_cache,
        ):
            env["TRITON_CACHE_DIR"] = triton_cache
            env["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
            env["TRITON_KLEIDIAI_CACHE_DIR"] = kai_cache
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_kleidiai_concurrent_calls_use_isolated_scratch(self):
        from vllm_fl.ops import cpu_int8_kai as op

        torch.manual_seed(1234)
        n, k = 512, 1024
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        packed = op._quantize_pack(weight)
        inputs = [
            torch.randn(8, k, dtype=torch.bfloat16),
            torch.randn(8, k, dtype=torch.bfloat16),
        ]
        references = [op.linear_w8a8(x, packed, n, k).clone() for x in inputs]

        def run(index):
            return op.linear_w8a8(inputs[index], packed, n, k).clone()

        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(100):
                outputs = list(pool.map(run, (0, 1)))
                for actual, expected in zip(outputs, references):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_kleidiai_matches_dequantized_reference(self):
        from vllm_fl.ops import cpu_int8_kai as op

        torch.manual_seed(20260803)
        n, k = 64, 128
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        scale = (weight.float().abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        quantized = (weight.float() / scale[:, None]).round().clamp(-128, 127)
        dequantized = quantized * scale[:, None]
        packed = op._quantize_pack(weight)
        for m in (1, 3, 7):
            x = torch.randn(m, k, dtype=torch.bfloat16)
            actual = op.linear_w8a8(x, packed, n, k).float()
            expected = x.float() @ dequantized.T
            relative_error = torch.linalg.vector_norm(actual - expected) / (
                torch.linalg.vector_norm(expected) + 1e-12
            )
            self.assertLess(float(relative_error), 0.02)

    def test_torchpack_matches_dequantized_reference(self):
        from vllm_fl.ops import cpu_int8_pack as op

        torch.manual_seed(20260804)
        n, k = 64, 128
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        quantized, scale = op._quantize_int8(weight)
        linear = op._make_cpu_linear(quantized, scale, n, k)
        dequantized = quantized.float() * scale.float()[:, None]
        for m in (1, 3, 7):
            x = torch.randn(m, k, dtype=torch.bfloat16)
            actual = linear(x, weight, None).float()
            expected = x.float() @ dequantized.T
            relative_error = torch.linalg.vector_norm(actual - expected) / (
                torch.linalg.vector_norm(expected) + 1e-12
            )
            self.assertLess(float(relative_error), 0.02)

    def test_w8_library_exports_only_flagtree_abi(self):
        nm = shutil.which("nm")
        if nm is None:
            self.skipTest("nm is required for the native ABI check")
        from triton.language.extra.cpu import kleidiai

        library = kleidiai.build_runtime("w8a8")
        result = subprocess.run(
            [nm, "-D", "--defined-only", str(library)],
            check=True,
            text=True,
            capture_output=True,
        )
        exported = {
            line.split()[-1]
            for line in result.stdout.splitlines()
            if line.split()
        }
        self.assertEqual(
            exported,
            {
                "flagtree_kai_w8a8_linear",
                "flagtree_kai_w8a8_pack_rhs",
                "flagtree_kai_w8a8_rhs_packed_size",
            },
        )

    def test_tleraw_checkpoint_kernel_packs_native_int8_weights(self):
        from vllm.model_executor.kernels.linear import init_int8_linear_kernel
        from vllm_fl.ops import cpu_int8_tleraw as op

        op.register_checkpoint_int8_kernel()
        kernel = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name="test_checkpoint_w8a8",
        )
        self.assertIsInstance(kernel, op.FlagTreeKleidiAIInt8LinearKernel)

        torch.manual_seed(0)
        n, k = 64, 130
        layer = torch.nn.Module()
        weight = torch.randint(-128, 128, (n, k), dtype=torch.int8)
        scale = torch.rand(n, 1, dtype=torch.float32) * 0.02 + 1e-4
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(scale, requires_grad=False)
        )
        layer.register_parameter("input_scale", None)
        layer.register_parameter("input_zero_point", None)
        layer.register_parameter("azp_adj", None)

        kernel.process_weights_after_loading(layer)
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.weight_scale)
        self.assertEqual(layer._fl_w8a8_packed_rhs.dtype, torch.uint8)
        self.assertNotIn("_fl_w8a8_packed_rhs", layer.state_dict())

    def test_tleraw_checkpoint_kernel_rejects_other_int8_schemes(self):
        from vllm.model_executor.kernels.linear.scaled_mm import (
            Int8ScaledMMLinearLayerConfig,
        )
        from vllm_fl.ops.cpu_int8_tleraw import (
            FlagTreeKleidiAIInt8LinearKernel as kernel,
        )

        supported = Int8ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        self.assertEqual(kernel.can_implement(supported), (True, None))
        for config in (
            Int8ScaledMMLinearLayerConfig(False, False, True),
            Int8ScaledMMLinearLayerConfig(False, True, False),
            Int8ScaledMMLinearLayerConfig(True, True, True),
        ):
            self.assertFalse(kernel.can_implement(config)[0])

    def test_tleraw_checkpoint_kernel_is_fullgraph_compile_safe(self):
        code = r'''
import torch
from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm_fl.ops import cpu_int8_tleraw as op

op.register_checkpoint_int8_kernel()
kernel = init_int8_linear_kernel(
    is_channelwise=True,
    is_static_input_scheme=False,
    input_symmetric=True,
    module_name="test_checkpoint_w8a8_fullgraph",
)
n, k = 64, 128
layer = torch.nn.Module()
torch.manual_seed(0)
layer.register_parameter(
    "weight",
    torch.nn.Parameter(torch.randint(-128, 128, (n, k), dtype=torch.int8), False),
)
layer.register_parameter(
    "weight_scale",
    torch.nn.Parameter(torch.rand(n, 1, dtype=torch.float32) * 0.02 + 1e-4, False),
)
for name in ("input_scale", "input_zero_point", "azp_adj"):
    layer.register_parameter(name, None)
kernel.process_weights_after_loading(layer)

def linear(x):
    return kernel.apply_weights(layer, x)

compiled = torch.compile(linear, fullgraph=True, dynamic=True)
for m in (7, 3, 1, 7, 1):
    x = torch.randn(m, k, dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(x), linear(x), rtol=0, atol=0)
'''
        self.run_in_fresh_cache(code)

    def test_w4_and_w8_tle_modules_can_execute_in_one_process(self):
        code = r'''
import torch
from vllm_fl.ops import cpu_int4_tleraw as w4
from vllm_fl.ops import cpu_int8_tleraw as w8

torch.manual_seed(42)
n, k = 64, 128
weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
scales = torch.rand(n, 1, dtype=torch.bfloat16) * 0.1 + 1e-3
rhs = w4._pack_rhs(weight, scales)
dequant = weight.float() * scales.float()
x = torch.randn(1, k, dtype=torch.bfloat16)
actual = w4.linear_w4a8(x, rhs, n, k).float()
expected = x.float() @ dequant.T
relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
assert float(relative_error) < 0.015, float(relative_error)

w8_weight = torch.randn(n, k, dtype=torch.bfloat16)
w8_scale = (w8_weight.float().abs().amax(dim=1) / 127.0).clamp(min=1e-8)
w8_quantized = (w8_weight.float() / w8_scale[:, None]).round().clamp(-128, 127)
w8_packed = w8._quantize_pack(w8_weight)
w8_actual = w8.linear_w8a8(x, w8_packed, n, k).float()
w8_expected = x.float() @ (w8_quantized * w8_scale[:, None]).T
w8_error = torch.linalg.vector_norm(w8_actual - w8_expected) / torch.linalg.vector_norm(w8_expected)
assert float(w8_error) < 0.02, float(w8_error)
'''
        self.run_in_fresh_cache(code)

    def test_w8a8_tle_fresh_cache_uses_pack_library_symbols(self):
        code = r'''
from concurrent.futures import ThreadPoolExecutor

import torch
from vllm_fl.ops import cpu_int8_tleraw as op

torch.manual_seed(20260802)
n, k = 64, 128
weight = torch.randn(n, k, dtype=torch.bfloat16)
scale = (weight.float().abs().amax(dim=1) / 127.0).clamp(min=1e-8)
quantized = (weight.float() / scale[:, None]).round().clamp(-128, 127)
dequantized = quantized * scale[:, None]
packed = op._quantize_pack(weight)
for m in (1, 3, 7):
    x = torch.randn(m, k, dtype=torch.bfloat16)
    actual = op.linear_w8a8(x, packed, n, k).float()
    expected = x.float() @ dequantized.T
    relative_error = (
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected)
    )
    assert float(relative_error) < 0.02, (m, float(relative_error))

thread_inputs = [
    torch.randn(8, k, dtype=torch.bfloat16),
    torch.randn(8, k, dtype=torch.bfloat16),
]
references = [op.linear_w8a8(x, packed, n, k).clone() for x in thread_inputs]

def run(index):
    return op.linear_w8a8(thread_inputs[index], packed, n, k).clone()

with ThreadPoolExecutor(max_workers=2) as pool:
    for _ in range(100):
        outputs = list(pool.map(run, (0, 1)))
        for actual, expected in zip(outputs, references):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
'''
        self.run_in_fresh_cache(code)


if __name__ == "__main__":
    unittest.main()
