"""Seeded CPU sampling must survive discarded chunks without consuming RNG draws."""
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_fl.patches.arm_minicpm_runtime import _install_cpu_sampling_rng


class TestCPUChunkedSampling(unittest.TestCase):
    def test_discarded_chunk_and_mixed_decode(self):
        gpu_sample = GPUModelRunner._sample
        gpu_bookkeeping = GPUModelRunner._bookkeeping_sync
        _install_cpu_sampling_rng()
        runner = object.__new__(CPUModelRunner)
        generators = {i: torch.Generator().manual_seed(1234+i) for i in range(2)}
        baseline = {i: gen.get_state().clone() for i, gen in generators.items()}
        runner.input_batch = NS(
            num_reqs=2, generators=generators,
            sampling_metadata=NS(generators=generators),
            update_async_output_token_ids=lambda: None,
            req_ids=['prefill', 'decode'], req_id_to_index={'prefill':0,'decode':1},
            token_ids_cpu=np.zeros((2, 8), dtype=np.int64),
            is_token_ids=np.zeros((2, 8), dtype=bool),
            num_tokens_no_spec=np.array([0,0]),
        )
        runner.discard_request_mask = NS(np=np.array([True,False]))
        runner.use_async_scheduling = False
        runner.routed_experts_initialized = False
        runner.max_model_len = 8
        runner.requests = {key:NS(output_token_ids=[]) for key in runner.input_batch.req_ids}
        runner._to_list = lambda tensor: tensor.tolist()
        runner._get_prompt_logprobs_dict = lambda *args: {}

        def sampler(logits, sampling_metadata):
            tokens = [torch.multinomial(row.softmax(-1), 1, generator=generators[i])
                      for i, row in enumerate(logits)]
            return NS(sampled_token_ids=torch.stack(tokens), logprobs_tensors=None)

        runner.sampler = sampler
        logits = torch.tensor([[1.,2.,3.,4.], [4.,3.,2.,1.]])
        expected_gen = torch.Generator().set_state(baseline[1])
        for _ in range(2):
            output = runner._sample(logits, None)
            expected = torch.multinomial(logits[1].softmax(-1), 1, generator=expected_gen).item()
            result = runner._bookkeeping_sync(NS(num_scheduled_tokens={}), output,
                                             logits, torch.zeros((2,1)), 2)
            self.assertEqual(result[2], [[], [expected]])
            self.assertTrue(torch.equal(generators[0].get_state(), baseline[0]))
            self.assertTrue(torch.equal(generators[1].get_state(), expected_gen.get_state()))
            self.assertEqual(set(generators), {0,1})
        runner.discard_request_mask.np[:] = False
        output = runner._sample(logits, None)
        fresh = torch.Generator().set_state(baseline[0])
        expected = torch.multinomial(logits[0].softmax(-1), 1, generator=fresh).item()
        self.assertEqual(output.sampled_token_ids[0].item(), expected)
        self.assertTrue(torch.equal(generators[0].get_state(), fresh.get_state()))
        self.assertIs(GPUModelRunner._sample, gpu_sample)
        self.assertIs(GPUModelRunner._bookkeeping_sync, gpu_bookkeeping)


if __name__ == '__main__':
    unittest.main(verbosity=2)
