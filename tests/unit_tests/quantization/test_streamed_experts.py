"""Numerical and rejection coverage for the ARM packed-checkpoint loader."""

import os
import platform
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from vllm_fl.patches.minicpmv47_cpu_model import install_cpu_model_hooks
from vllm_fl.quantization.streamed_experts import StreamedExperts


@unittest.skipUnless(
    platform.machine().lower() in ("aarch64", "arm64"), "ARM KAI packer required"
)
class TestStreamedExperts(unittest.TestCase):
    def setUp(self):
        from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import (
            convert_to_w4a8_int8_moe_format,
        )

        generator = torch.Generator().manual_seed(726)
        self.weights = {}
        raw = {}
        for expert in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shape = (64, 32) if proj == "down_proj" else (32, 64)
                weight = torch.randint(
                    -8, 8, shape, generator=generator, dtype=torch.int8
                )
                scale = (
                    torch.rand((shape[0], shape[1] // 32), generator=generator) + 0.1
                ).bfloat16()
                raw[expert, proj] = (weight, scale)
                prefix = f"model.language_model.layers.0.mlp.experts.{expert}.{proj}."
                self.weights[prefix + "weight_packed"] = pack_to_int32(weight, 4)
                self.weights[prefix + "weight_scale"] = scale
        w13 = torch.stack(
            [
                torch.cat([raw[e, p][0] for p in ("gate_proj", "up_proj")])
                for e in range(2)
            ]
        )
        s13 = torch.stack(
            [
                torch.cat([raw[e, p][1] for p in ("gate_proj", "up_proj")])
                for e in range(2)
            ]
        )
        w2 = torch.stack([raw[e, "down_proj"][0] for e in range(2)])
        s2 = torch.stack([raw[e, "down_proj"][1] for e in range(2)])
        self.reference = convert_to_w4a8_int8_moe_format(w13, w2, s13, s2, 32)
        self.raw = dict(
            w13_weight=w13, w2_weight=w2, w13_weight_scale=s13, w2_weight_scale=s2
        )

    def build(self, checkpoint=None):
        routed = torch.nn.Module()
        for name, tensor in self.raw.items():
            routed.register_parameter(
                name, torch.nn.Parameter(torch.empty_like(tensor), requires_grad=False)
            )
        for name in ("w13_weight_packed", "w2_weight_packed"):
            routed.register_parameter(
                name, torch.nn.Parameter(torch.empty(0), requires_grad=False)
            )
        method = type("CompressedTensorsW4A8Int8MoEMethod", (), {})()
        method.group_size = 32
        method.has_bias = False
        method.moe = None
        method.experts_cls = None
        method.process_weights_after_loading = lambda layer: None
        method.get_fused_moe_quant_config = lambda layer: None
        routed.quant_method = method
        routed._expert_routing_tables = lambda: None
        model = NS(
            config=NS(_name_or_path=checkpoint),
            language_model=NS(
                model=NS(layers=[NS(mlp=NS(experts=NS(routed_experts=routed)))])
            ),
        )
        return StreamedExperts(model), routed

    def test_exact_packing_and_completed_postprocess(self):
        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8.make_w4a8_int8_moe_kernel",
            return_value=object(),
        ):
            for reverse in (False, True):
                with self.subTest(reverse=reverse):
                    stream, layer = self.build()
                    self.assertEqual(
                        list(
                            stream.weights(
                                sorted(self.weights.items(), reverse=reverse)
                            )
                        ),
                        [],
                    )
                    self.assertTrue(
                        torch.equal(self.reference[0], layer.w13_weight_packed)
                    )
                    self.assertTrue(
                        torch.equal(self.reference[1], layer.w2_weight_packed)
                    )
                    layer.quant_method.process_weights_after_loading(layer)
                    self.assertEqual(stream.pending_bytes, 0)
                    self.assertEqual(stream.report["experts"], 2)
                    self.assertEqual(layer.w13_weight.numel(), 0)

    def test_incomplete_and_duplicate_rejected(self):
        stream, layer = self.build()
        with self.assertRaises(RuntimeError):
            layer.quant_method.process_weights_after_loading(layer)
        with self.assertRaises(ValueError):
            list(stream.weights(list(self.weights.items())[:1]))
        stream, _ = self.build()
        first = next(iter(self.weights.items()))
        with self.assertRaises(ValueError):
            list(stream.weights([first, first]))

    def test_bad_shape_rejected(self):
        stream, _ = self.build()
        name, tensor = next(iter(self.weights.items()))
        with self.assertRaises(ValueError):
            list(stream.weights([(name, tensor[:1])]))

    def test_mmap_packing(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"MINICPM_EXPERT_MMAP_DIR": directory}),
            patch(
                "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8.make_w4a8_int8_moe_kernel",
                return_value=object(),
            ),
        ):
            stream, layer = self.build()
            list(stream.weights(sorted(self.weights.items())))
            self.assertTrue(torch.equal(self.reference[0], layer.w13_weight_packed))
            self.assertTrue(torch.equal(self.reference[1], layer.w2_weight_packed))
            self.assertGreater(stream.report["mmap_bytes"], 0)

    def test_cache_reuse_and_changed_checkpoint_rejected(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}")
            save_file(self.weights, str(checkpoint / "model.safetensors"))
            with (
                patch.dict(
                    "os.environ",
                    {
                        "MINICPM_EXPERT_MMAP_DIR": str(root / "packed"),
                        "MINICPM_EXPERT_CACHE_REUSE": "1",
                    },
                ),
                patch(
                    "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8.make_w4a8_int8_moe_kernel",
                    return_value=object(),
                ),
            ):
                stream, _ = self.build(str(checkpoint))
                list(stream.weights(sorted(self.weights.items())))
                cached, layer = self.build(str(checkpoint))
                list(cached.weights(sorted(self.weights.items())))
                self.assertTrue(cached.report["cache_hit"])
                self.assertEqual(cached.report["peak_pending_bytes"], 0)
                self.assertTrue(torch.equal(self.reference[0], layer.w13_weight_packed))
                self.assertTrue(torch.equal(self.reference[1], layer.w2_weight_packed))
                # Content verification must not rely only on size and mtime.
                path = root / "packed/layer-0-w13_weight_packed.bin"
                data, stat = path.read_bytes(), path.stat()
                changed = bytes([data[0] ^ 1]) + data[1:]
                path.write_bytes(changed)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.build(str(checkpoint))
                path.write_bytes(data)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                (checkpoint / "config.json").write_text('{"changed":true}')
                with self.assertRaises(ValueError):
                    self.build(str(checkpoint))


def model_type():
    class Model:
        def __init__(self):
            self.language_model = NS(model=NS(layers=[]))
            self.received = None

        def load_weights(self, weights):
            self.received = list(weights)
            return {name for name, _ in self.received}

    return Model


@pytest.mark.parametrize("w4_experts", [False, True])
def test_cpu_model_hook_reuses_shared_loader_and_is_idempotent(monkeypatch, w4_experts):
    from vllm_fl.ops import minicpmv47_moe
    from vllm_fl.patches import minicpmv47_cpu_model

    monkeypatch.delenv("MINICPM_STREAM_EXPERT_LOADING", raising=False)
    monkeypatch.setenv("FL_ARM_NATIVE", "1")
    monkeypatch.setattr(minicpmv47_cpu_model, "_has_w4_experts", lambda _: w4_experts)
    installed = []
    monkeypatch.setattr(minicpmv47_moe, "install", lambda: installed.append("bridge"))
    monkeypatch.setattr(
        minicpmv47_moe, "install_direct_model_moe", lambda _: installed.append("direct")
    )
    cls = model_type()
    assert install_cpu_model_hooks(cls)
    current = cls.load_weights
    assert not install_cpu_model_hooks(cls)
    assert cls.load_weights is current
    model = cls()
    assert model.load_weights(iter([("weight", torch.zeros(1))])) == {"weight"}
    assert len(model.received) == 1
    assert installed == (["bridge", "direct"] if w4_experts else [])


def test_cpu_model_hook_rejects_streaming_on_dense_model(monkeypatch):
    monkeypatch.setenv("MINICPM_STREAM_EXPERT_LOADING", "1")
    cls = model_type()
    install_cpu_model_hooks(cls)
    model = cls()
    with pytest.raises(ValueError, match="W4A8 experts"):
        model.load_weights(iter([]))
    assert model.received is None


def test_cpu_model_hook_keeps_recurrent_state_fp32():
    cls = model_type()
    install_cpu_model_hooks(cls)
    config = NS(
        model_config=NS(dtype=torch.bfloat16),
        cache_config=NS(mamba_cache_dtype="auto"),
    )
    assert cls.get_mamba_state_dtype_from_config(config) == (
        torch.bfloat16,
        torch.float32,
    )
