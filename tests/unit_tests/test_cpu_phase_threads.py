# Copyright (c) 2026 BAAI. All rights reserved.
"""Request-aware phase selection, including chunked/mixed/prefix-cached batches."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from vllm_fl.worker.cpu_thread_policy import ThreadPolicy, execution_phase

def schedule(new=(), cached=(), computed=(), tokens=None):
    tokens = tokens or {}
    return NS(scheduled_new_reqs=list(new),
              scheduled_cached_reqs=NS(req_ids=list(cached), num_computed_tokens=list(computed)),
              num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values()))


def fresh(req_id, prompt, done=0):
    return NS(req_id=req_id, prompt_token_ids=list(range(prompt)),
              prompt_embeds=None, num_computed_tokens=done)


states = {'a': NS(num_prompt_tokens=512, num_computed_tokens=511),
          'b': NS(num_prompt_tokens=64, num_computed_tokens=64)}
cases = [
    ('initial_prompt', schedule([fresh('a',512)], tokens={'a':512}), {}, 'prefill'),
    ('one_token_prompt', schedule([fresh('a',1)], tokens={'a':1}), {}, 'prefill'),
    ('one_token_tail_of_chunked_prefill', schedule(cached=['a'], computed=[511], tokens={'a':1}), states, 'prefill'),
    ('single_decode', schedule(cached=['a'], computed=[512], tokens={'a':1}), states, 'decode'),
    ('batched_decode', schedule(cached=['a','b'], computed=[513,65], tokens={'a':1,'b':1}), states, 'decode'),
    ('multiple_decode_tokens', schedule(cached=['a'], computed=[512], tokens={'a':4}), states, 'decode'),
    ('mixed_batch', schedule([fresh('a',512)], cached=['b'], computed=[65], tokens={'a':128,'b':1}), states, 'mixed'),
    ('cached_prefix_with_remaining_prompt', schedule([fresh('a',512,480)], tokens={'a':32}), {}, 'prefill'),
    ('fully_cached_prompt', schedule([fresh('a',512,512)], tokens={'a':1}), {}, 'decode'),
    ('resumed_chunk', schedule(cached=['a'], computed=[0], tokens={'a':128}), states, 'prefill'),
    ('idle', schedule(), {}, 'idle'),
    ('prompt_embeddings', schedule([NS(req_id='a',prompt_token_ids=None,prompt_embeds=NS(shape=(32,2048)),num_computed_tokens=0)], tokens={'a':32}), {}, 'prefill'),
]
class TestCPUPhaseThreads(unittest.TestCase):
    def test_real_torch_openmp_counts_follow_request_phase(self):
        import torch
        from vllm.v1.worker.cpu_worker import CPUWorker
        from vllm_fl.worker.cpu_phase_worker import CPUPhaseWorker, _torch_openmp

        worker = object.__new__(CPUPhaseWorker)
        worker.thread_policy = ThreadPolicy(8, 6)
        worker.thread_calls = {}
        worker.observed_threads = {}
        worker.model_runner = NS(requests=states)
        worker._phase_openmp = _torch_openmp()
        old_torch = torch.get_num_threads()
        old_openmp = worker._phase_openmp.omp_get_max_threads()
        try:
            with patch.object(CPUWorker, 'execute_model', return_value='result'), patch.object(torch, 'set_num_threads') as disabled_setter:
                self.assertEqual(worker.execute_model(schedule([fresh('a',512)], tokens={'a':128})), 'result')
                self.assertEqual(torch.get_num_threads(), 8)
                self.assertEqual(worker._phase_openmp.omp_get_max_threads(), 8)
                worker.execute_model(schedule(cached=['a'], computed=[512], tokens={'a':1}))
                self.assertEqual(torch.get_num_threads(), 6)
                self.assertEqual(worker._phase_openmp.omp_get_max_threads(), 6)
                worker.execute_model(schedule())
                self.assertEqual(worker.thread_calls, {'prefill':1, 'decode':1})
                disabled_setter.assert_not_called()
        finally:
            torch.set_num_threads(old_torch)
            worker._phase_openmp.omp_set_num_threads(old_openmp)

    def test_execution_phase(self):
        for name, output, requests, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(execution_phase(output, requests), expected)

    def test_unknown_request_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No state'):
            execution_phase(schedule(tokens={'missing':1}), {})

    def test_invalid_threads_rejected(self):
        for pair in [(0,8),(18,0),(65,8),(18,65)]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                ThreadPolicy(*pair)

    def test_policy_configuration(self):
        self.assertIsNone(ThreadPolicy.from_env({}))
        self.assertIsNone(ThreadPolicy.from_env({'MINICPM_PREFILL_THREADS':'0'}))
        policy = ThreadPolicy.from_env({'MINICPM_PREFILL_THREADS':'18','MINICPM_DECODE_THREADS':'8'})
        self.assertEqual(policy, ThreadPolicy(18,8))
        self.assertEqual(policy.threads_for('mixed'), 18)
        self.assertEqual(policy.threads_for('decode'), 8)
        policy = ThreadPolicy.from_env({'FL_CPU_PREFILL_THREADS':'8','FL_CPU_DECODE_THREADS':'6'})
        self.assertEqual(policy, ThreadPolicy(8,6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
