# Copyright (c) 2026 BAAI. All rights reserved.
"""vLLM architecture selection from checkpoint metadata, independent of env."""
from .hf.configuration_minicpmv4_7 import MiniCPMV4_7Config as SourceConfig


class MiniCPMV4_7Config(SourceConfig):
    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Both HF checkpoints carry the same conditional architecture. vLLM
        # inspects classes before constructing them, so expose the dense class
        # explicitly rather than selecting it through a machine-specific env.
        if not getattr(self.text_config, "num_experts", 0):
            self.architectures = [
                "MiniCPMV4_7DenseForConditionalGeneration"
                if name == "MiniCPMV4_7ForConditionalGeneration" else name
                for name in (self.architectures or [])
            ]
