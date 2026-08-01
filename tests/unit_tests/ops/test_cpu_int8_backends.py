# Copyright (c) 2026 BAAI. All rights reserved.

import os
import platform
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
        with (
            tempfile.TemporaryDirectory(prefix="fl-triton-cache-") as triton_cache,
            tempfile.TemporaryDirectory(prefix="fl-inductor-cache-") as inductor_cache,
        ):
            env["TRITON_CACHE_DIR"] = triton_cache
            env["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
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

    def test_importing_both_tle_modules_does_not_poison_selected_backend(self):
        kai_dir = os.environ.get("FL_KAI_W4A8_DIR")
        kleidiai_root = os.environ.get("KLEIDIAI_ROOT")
        if not kai_dir or not kleidiai_root:
            self.skipTest("FL_KAI_W4A8_DIR and KLEIDIAI_ROOT are required")

        code = r'''
import torch
from vllm_fl.ops import cpu_int4_tleraw as w4
from vllm_fl.ops import cpu_int8_tleraw as w8

torch.manual_seed(42)
n, k = 64, 128
weight = torch.randn(n, k, dtype=torch.bfloat16)
native, scales = w4.quant_native_qs4c32(weight)
rhs = w4._pack_rhs(native, scales, n, k)
unsigned = torch.empty((n, k), dtype=torch.uint8)
unsigned[:, 0::2] = native & 0xF
unsigned[:, 1::2] = native >> 4
dequant = (unsigned.float() - 8) * scales.float().repeat_interleave(w4.BL, dim=1)
x = torch.randn(1, k, dtype=torch.bfloat16)
actual = w4.linear_w4a8(x, rhs, n, k).float()
expected = x.float() @ dequant.T
relative_error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
assert float(relative_error) < 0.015, float(relative_error)

try:
    w8._register_tle_w8a8()
except RuntimeError as exc:
    assert "already active as w4a8" in str(exc), str(exc)
else:
    raise AssertionError("activating a second packed TLE ABI should fail")
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
