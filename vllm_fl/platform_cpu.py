# Copyright (c) 2026 BAAI. All rights reserved.
"""Explicit single-worker CPU deployment; the default platform stays stock."""
from vllm.platforms.cpu import CpuPlatform


class CpuPlatformFL(CpuPlatform):
    @classmethod
    def check_and_update_config(cls, vllm_config):
        config = vllm_config.parallel_config
        requested = config.distributed_executor_backend == 'uni'
        if requested and config.world_size != 1:
            raise ValueError('FL single-worker CPU executor requires world_size=1')
        super().check_and_update_config(vllm_config)
        if requested:
            config.distributed_executor_backend = 'uni'

    @classmethod
    def update_block_size_for_backend(cls, vllm_config):
        if vllm_config.model_config and vllm_config.model_config.is_hybrid:
            backend = cls._find_non_ssm_backend(vllm_config)
            if backend is not None:
                cls._align_hybrid_block_size(vllm_config,backend)
