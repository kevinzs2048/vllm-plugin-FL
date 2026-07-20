# Copyright (c) 2025 BAAI. All rights reserved.
"""ARM CPU platform for the FL TLE-raw W4A8 integration.

vllm-plugin-FL has no CPU vendor/backend (its PlatformFL/WorkerFL are GPU-shaped, adapted
from vLLM v0.20.2 CUDA). On ARM CPU we register this subclass so the plugin provides a CPU
backend while inheriting vLLM's native CpuPlatform (torch.compile, CPUWorker). It also fixes
a vLLM-0.20.2 CPU default that would otherwise prevent model compilation:

  Inductor is off by default: 0.20.2 only wires the inductor backend when
     compilation_config.mode == VLLM_COMPILE, but the default is None -> no fusion. We set it
     to VLLM_COMPILE (unless the user forced eager) so decode gets inductor-fused rms/silu/rope.
The native CPU platform requires its MP executor to configure OpenMP correctly. An explicit
FL_CPU_UNIPROC=1 escape hatch keeps the measured in-process path available for controlled,
single-worker deployments that configure thread affinity and allocator preload themselves.

The KleidiAI int4 op is installed via register_model() (runs in whichever process loads the
model). Net effect: plugin-driven int4 on 0.20.2 matches the native/manual int4 performance.
"""
import os

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

logger = init_logger(__name__)


class CpuPlatformFL(CpuPlatform):
    """Native vLLM CPU platform with compile enabled for ARM TLE-raw W4A8."""

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        from vllm.config import CompilationMode
        from vllm_fl.patches.dynamo_metrics import (
            patch_dynamo_metrics_serialization,
        )

        patch_dynamo_metrics_serialization()

        cc = vllm_config.compilation_config
        eager = getattr(vllm_config.model_config, "enforce_eager", False)
        # vLLM CPU only wires Inductor when mode == VLLM_COMPILE.
        if not eager and cc.mode is None:
            cc.mode = CompilationMode.VLLM_COMPILE
            logger.info("[vllm_fl] CPU compile mode -> VLLM_COMPILE (inductor fusion on)")

        pc = vllm_config.parallel_config
        uniproc_requested = (
            os.environ.get("FL_CPU_UNIPROC", "0") == "1"
            and pc.world_size == 1
        )
        if (
            uniproc_requested
            and os.environ.get("FL_CPU_OMP_ACTIVE_WAIT", "1") != "0"
        ):
            os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
            logger.info(
                "[vllm_fl] uniproc latency mode: OMP_WAIT_POLICY=%s",
                os.environ["OMP_WAIT_POLICY"],
            )

        super().check_and_update_config(vllm_config)

        # Opt-in only: vLLM's CPU platform normally requires MP to configure OMP.
        if (
            uniproc_requested
            and pc.distributed_executor_backend == "mp"
        ):
            pc.distributed_executor_backend = "uni"
            logger.warning(
                "[vllm_fl] FL_CPU_UNIPROC=1: using unsupported-by-vLLM "
                "in-process CPU executor; caller must configure OpenMP"
            )

        logger.info("[vllm_fl] FL ARM CPU platform active (native-backed)")
