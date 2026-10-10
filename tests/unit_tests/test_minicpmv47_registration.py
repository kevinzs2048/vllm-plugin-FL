# Copyright (c) 2026 BAAI. All rights reserved.
"""Checkpoint-derived dense/MoE selection and processor registry contracts."""
import unittest

from vllm_fl.models.minicpmv4_7.configuration import MiniCPMV4_7Config
from vllm_fl.models.minicpmv4_7.registration import register_model


class TestMiniCPMV47Registration(unittest.TestCase):
    def test_dense_architecture_comes_from_text_config(self):
        config = MiniCPMV4_7Config(
            architectures=['MiniCPMV4_7ForConditionalGeneration'],
            text_config=dict(model_type='qwen3_5_text'))
        self.assertEqual(config.architectures, ['MiniCPMV4_7DenseForConditionalGeneration'])
        self.assertFalse(getattr(config.text_config, 'num_experts', 0))

    def test_moe_architecture_preserved(self):
        config = MiniCPMV4_7Config(
            architectures=['MiniCPMV4_7ForConditionalGeneration'],
            text_config=dict(model_type='qwen3_5_moe_text', num_experts=256))
        self.assertEqual(config.architectures, ['MiniCPMV4_7ForConditionalGeneration'])
        self.assertEqual(config.text_config.num_experts, 256)

    def test_registration_config_round_trip_and_processor(self):
        from transformers import AutoConfig, AutoProcessor
        from vllm import ModelRegistry
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

        register_model()
        self.assertFalse(register_model())
        self.assertIs(_CONFIG_REGISTRY['minicpmv4_7'], MiniCPMV4_7Config)
        config = AutoConfig.for_model('minicpmv4_7',
            architectures=['MiniCPMV4_7ForConditionalGeneration'],
            text_config=dict(model_type='qwen3_5_text'))
        restored = MiniCPMV4_7Config.from_dict(config.to_dict())
        self.assertEqual(restored.architectures, config.architectures)
        self.assertEqual(PROCESSOR_MAPPING[MiniCPMV4_7Config].__name__, 'MiniCPMV4_7Processor')
        supported = ModelRegistry.get_supported_archs()
        self.assertIn('MiniCPMV4_7ForConditionalGeneration', supported)
        self.assertIn('MiniCPMV4_7DenseForConditionalGeneration', supported)
