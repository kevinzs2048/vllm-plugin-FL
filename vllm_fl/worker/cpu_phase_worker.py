# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in vLLM 0.24 CPU worker with request-aware phase thread counts."""

import ctypes
import os
import sys
from pathlib import Path

import torch

from vllm.v1.worker.cpu_worker import CPUWorker

from vllm_fl.worker.cpu_thread_policy import ThreadPolicy, execution_phase


def _torch_openmp():
    directory = Path(torch.__file__).parent / "lib"
    paths = (
        [directory / "libomp.dylib"]
        if sys.platform == "darwin"
        else sorted(directory.glob("libgomp*.so*"))
        + sorted(directory.glob("libomp*.so*"))
    )
    if not paths or not paths[0].is_file():
        raise RuntimeError(
            "Phase threads require the OpenMP runtime shipped with Torch"
        )
    library = ctypes.CDLL(str(paths[0]))
    library.omp_set_num_threads.argtypes = [ctypes.c_int]
    library.omp_set_num_threads.restype = None
    library.omp_get_max_threads.argtypes = []
    library.omp_get_max_threads.restype = ctypes.c_int
    return library


class CPUPhaseWorker(CPUWorker):
    """Select explicitly with ``--worker-cls``; no global worker patching.

    Use FL_CPU_PREFILL_THREADS and FL_CPU_DECODE_THREADS. Historical
    MINICPM_* aliases remain accepted. Mixed requests use prefill threads.
    Requires the vLLM 0.24 V1 runner, unbound threads, TP=1 and PP=1.
    """

    def __init__(self, *args, **kwargs):
        self.thread_policy = ThreadPolicy.from_env()
        self.thread_calls = {}
        self.observed_threads = {}
        super().__init__(*args, **kwargs)
        if self.thread_policy is None:
            return
        if os.environ.get("VLLM_CPU_OMP_THREADS_BIND") != "nobind":
            raise ValueError("Phase threads require VLLM_CPU_OMP_THREADS_BIND=nobind")
        if self.use_v2_model_runner:
            raise ValueError("Phase threads require the vLLM V1 model runner")
        if (
            self.parallel_config.tensor_parallel_size != 1
            or self.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Phase threads require TP=1 and PP=1")
        # Match Torch's runtime directly; 1B W8A8 does not need to load G32,
        # KleidiAI or any model-specific extension just to set its thread count.
        self._phase_openmp = _torch_openmp()

    def execute_model(self, scheduler_output):
        if self.thread_policy is not None:
            phase = execution_phase(scheduler_output, self.model_runner.requests)
            if phase != "idle":
                count = self.thread_policy.threads_for(phase)
                # CPUWorker.init_device replaces the Python export with a
                # warning-only stub, including for nobind. Use Torch's own
                # unchanged binding for this explicit unbound worker; do not
                # restore or patch the process-global safeguard.
                torch_threads = torch.get_num_threads()
                if torch_threads != count:
                    torch._C.set_num_threads(count)
                    torch_threads = torch.get_num_threads()
                openmp_threads = self._phase_openmp.omp_get_max_threads()
                if openmp_threads != count:
                    self._phase_openmp.omp_set_num_threads(count)
                    openmp_threads = self._phase_openmp.omp_get_max_threads()
                actual = dict(
                    torch_threads=torch_threads, openmp_max_threads=openmp_threads
                )
                if (
                    actual["torch_threads"] != count
                    or actual["openmp_max_threads"] != count
                ):
                    raise RuntimeError(
                        f"CPU phase thread setting failed: {phase}, {count}, {actual}"
                    )
                self.observed_threads[phase] = actual
                self.thread_calls[phase] = self.thread_calls.get(phase, 0) + 1
        return super().execute_model(scheduler_output)
