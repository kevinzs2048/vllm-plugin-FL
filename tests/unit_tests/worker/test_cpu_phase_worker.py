# Copyright (c) 2026 BAAI. All rights reserved.
"""Request-aware phase selection, including chunked/mixed/prefix-cached batches."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from vllm_fl.worker.cpu_thread_policy import (
    BlocktimePolicy,
    ThreadPolicy,
    execution_phase,
)


def schedule(new=(), cached=(), computed=(), tokens=None):
    tokens = tokens or {}
    return NS(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=NS(
            req_ids=list(cached), num_computed_tokens=list(computed)
        ),
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
    )


def fresh(req_id, prompt, done=0):
    return NS(
        req_id=req_id,
        prompt_token_ids=list(range(prompt)),
        prompt_embeds=None,
        num_computed_tokens=done,
    )


states = {
    "a": NS(num_prompt_tokens=512, num_computed_tokens=511),
    "b": NS(num_prompt_tokens=64, num_computed_tokens=64),
}
cases = [
    ("initial_prompt", schedule([fresh("a", 512)], tokens={"a": 512}), {}, "prefill"),
    ("one_token_prompt", schedule([fresh("a", 1)], tokens={"a": 1}), {}, "prefill"),
    (
        "one_token_tail_of_chunked_prefill",
        schedule(cached=["a"], computed=[511], tokens={"a": 1}),
        states,
        "prefill",
    ),
    (
        "single_decode",
        schedule(cached=["a"], computed=[512], tokens={"a": 1}),
        states,
        "decode",
    ),
    (
        "batched_decode",
        schedule(cached=["a", "b"], computed=[513, 65], tokens={"a": 1, "b": 1}),
        states,
        "decode",
    ),
    (
        "multiple_decode_tokens",
        schedule(cached=["a"], computed=[512], tokens={"a": 4}),
        states,
        "decode",
    ),
    (
        "mixed_batch",
        schedule(
            [fresh("a", 512)], cached=["b"], computed=[65], tokens={"a": 128, "b": 1}
        ),
        states,
        "mixed",
    ),
    (
        "cached_prefix_with_remaining_prompt",
        schedule([fresh("a", 512, 480)], tokens={"a": 32}),
        {},
        "prefill",
    ),
    (
        "fully_cached_prompt",
        schedule([fresh("a", 512, 512)], tokens={"a": 1}),
        {},
        "decode",
    ),
    (
        "resumed_chunk",
        schedule(cached=["a"], computed=[0], tokens={"a": 128}),
        states,
        "prefill",
    ),
    ("idle", schedule(), {}, "idle"),
    (
        "prompt_embeddings",
        schedule(
            [
                NS(
                    req_id="a",
                    prompt_token_ids=None,
                    prompt_embeds=NS(shape=(32, 2048)),
                    num_computed_tokens=0,
                )
            ],
            tokens={"a": 32},
        ),
        {},
        "prefill",
    ),
]


class TestCPUPhaseThreads(unittest.TestCase):
    def test_real_torch_openmp_counts_follow_request_phase(self):
        import torch

        from vllm.v1.worker.cpu_worker import CPUWorker

        from vllm_fl.worker.cpu_phase_worker import CPUPhaseWorker, _torch_openmp

        worker = object.__new__(CPUPhaseWorker)
        worker.thread_policy = ThreadPolicy(8, 6)
        worker.blocktime_policy = None
        worker.thread_calls = {}
        worker.observed_threads = {}
        worker.model_runner = NS(requests=states)
        worker._phase_openmp = _torch_openmp()
        old_torch = torch.get_num_threads()
        old_openmp = worker._phase_openmp.omp_get_max_threads()
        try:
            with (
                patch.object(CPUWorker, "execute_model", return_value="result"),
                patch.object(torch, "set_num_threads") as disabled_setter,
            ):
                self.assertEqual(
                    worker.execute_model(
                        schedule([fresh("a", 512)], tokens={"a": 128})
                    ),
                    "result",
                )
                self.assertEqual(torch.get_num_threads(), 8)
                self.assertEqual(worker._phase_openmp.omp_get_max_threads(), 8)
                worker.execute_model(
                    schedule(cached=["a"], computed=[512], tokens={"a": 1})
                )
                self.assertEqual(torch.get_num_threads(), 6)
                self.assertEqual(worker._phase_openmp.omp_get_max_threads(), 6)
                worker.execute_model(schedule())
                self.assertEqual(worker.thread_calls, {"prefill": 1, "decode": 1})
                disabled_setter.assert_not_called()
        finally:
            torch.set_num_threads(old_torch)
            worker._phase_openmp.omp_set_num_threads(old_openmp)

    def test_blocktime_policy_boundaries_and_configuration(self):
        self.assertIsNone(BlocktimePolicy.from_env({}))
        policy = BlocktimePolicy.from_env(
            {
                "FL_CPU_SHORT_BLOCKTIME_TOKENS": "512",
                "FL_CPU_SHORT_BLOCKTIME_MS": "1",
                "FL_CPU_BLOCKTIME_MS": "0",
            }
        )
        self.assertEqual(policy.blocktime_for("prefill", 512), 1)
        self.assertEqual(policy.blocktime_for("prefill", 513), 0)
        self.assertEqual(policy.blocktime_for("prefill", 0), 0)
        self.assertEqual(policy.blocktime_for("decode", 1), 0)
        self.assertEqual(policy.blocktime_for("mixed", 32), 0)
        with self.assertRaises(ValueError):
            BlocktimePolicy.from_env({"FL_CPU_SHORT_BLOCKTIME_TOKENS": "512"})
        with self.assertRaisesRegex(ValueError, "requires KMP_BLOCKTIME in ms"):
            BlocktimePolicy.from_env(
                {
                    "FL_CPU_SHORT_BLOCKTIME_TOKENS": "512",
                    "FL_CPU_SHORT_BLOCKTIME_MS": "1",
                    "FL_CPU_BLOCKTIME_MS": "0",
                    "KMP_BLOCKTIME": "200us",
                }
            )
        for values in [(0, 1, 0), (512, -1, 0), (512, 1, -1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                BlocktimePolicy(*values)

    def test_real_libomp_blocktime_follows_request_size(self):
        from vllm.v1.worker.cpu_worker import CPUWorker

        from vllm_fl.worker.cpu_phase_worker import CPUPhaseWorker, _torch_openmp

        runtime = _torch_openmp()
        if not hasattr(runtime, "kmp_get_blocktime"):
            self.skipTest("LLVM/Intel libomp required")
        worker = object.__new__(CPUPhaseWorker)
        worker.thread_policy = ThreadPolicy(4, 4)
        worker.blocktime_policy = BlocktimePolicy(512, 1, 0)
        worker.thread_calls = {}
        worker.observed_threads = {}
        worker.model_runner = NS(requests=states)
        worker._phase_openmp = _torch_openmp(blocktime=True)
        import torch

        old_threads = torch.get_num_threads()
        old_openmp = runtime.omp_get_max_threads()
        old_blocktime = worker._phase_openmp.kmp_get_blocktime()
        try:
            with patch.object(CPUWorker, "execute_model", return_value="result"):
                worker.execute_model(schedule([fresh("a", 512)], tokens={"a": 512}))
                self.assertEqual(worker._phase_openmp.kmp_get_blocktime(), 1)
                worker.execute_model(schedule([fresh("a", 1024)], tokens={"a": 1024}))
                self.assertEqual(worker._phase_openmp.kmp_get_blocktime(), 0)
        finally:
            torch._C.set_num_threads(old_threads)
            runtime.omp_set_num_threads(old_openmp)
            worker._phase_openmp.kmp_set_blocktime(old_blocktime)

    def test_execution_phase(self):
        for name, output, requests, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(execution_phase(output, requests), expected)

    def test_unknown_request_rejected(self):
        with self.assertRaisesRegex(ValueError, "No state"):
            execution_phase(schedule(tokens={"missing": 1}), {})

    def test_invalid_threads_rejected(self):
        for pair in [(0, 8), (18, 0), (65, 8), (18, 65)]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                ThreadPolicy(*pair)

    def test_policy_configuration(self):
        self.assertIsNone(ThreadPolicy.from_env({}))
        self.assertIsNone(ThreadPolicy.from_env({"MINICPM_PREFILL_THREADS": "0"}))
        policy = ThreadPolicy.from_env(
            {"MINICPM_PREFILL_THREADS": "18", "MINICPM_DECODE_THREADS": "8"}
        )
        self.assertEqual(policy, ThreadPolicy(18, 8))
        self.assertEqual(policy.threads_for("mixed"), 18)
        self.assertEqual(policy.threads_for("decode"), 8)
        policy = ThreadPolicy.from_env(
            {"FL_CPU_PREFILL_THREADS": "8", "FL_CPU_DECODE_THREADS": "6"}
        )
        self.assertEqual(policy, ThreadPolicy(8, 6))

    def test_optional_small_prefill_boundary_and_phase(self):
        policy = ThreadPolicy.from_env(
            {
                "FL_CPU_PREFILL_THREADS": "18",
                "FL_CPU_DECODE_THREADS": "8",
                "FL_CPU_SMALL_PREFILL_TOKENS": "32",
                "FL_CPU_SMALL_PREFILL_THREADS": "4",
            }
        )
        self.assertEqual(policy.threads_for("prefill", 32), 4)
        self.assertEqual(policy.threads_for("prefill", 33), 18)
        self.assertEqual(policy.threads_for("prefill", 0), 18)
        self.assertEqual(policy.threads_for("prefill"), 18)
        self.assertEqual(policy.threads_for("mixed", 8), 18)
        self.assertEqual(policy.threads_for("decode", 1), 8)
        for tokens, threads in [(32, 0), (0, 4), (-1, 4), (32, 65)]:
            with (
                self.subTest(tokens=tokens, threads=threads),
                self.assertRaises(ValueError),
            ):
                ThreadPolicy(18, 8, tokens, threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
