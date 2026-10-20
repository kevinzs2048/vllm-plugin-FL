# Copyright (c) 2026 BAAI. All rights reserved.
"""Request-aware CPU phase selection, independent of model operators."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadPolicy:
    prefill: int
    decode: int

    def __post_init__(self):
        if not 1 <= self.prefill <= 64 or not 1 <= self.decode <= 64:
            raise ValueError("Prefill/decode thread counts must be in [1, 64]")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        prefill = int(
            env.get("FL_CPU_PREFILL_THREADS", env.get("MINICPM_PREFILL_THREADS", "0"))
        )
        if prefill == 0:
            return None
        return cls(
            prefill,
            int(
                env.get("FL_CPU_DECODE_THREADS", env.get("MINICPM_DECODE_THREADS", "8"))
            ),
        )

    def threads_for(self, phase):
        if phase in ("prefill", "mixed"):
            return self.prefill
        if phase == "decode":
            return self.decode
        raise ValueError(f"No thread policy for phase {phase!r}")


def _prompt_length(request):
    if hasattr(request, "num_prompt_tokens"):
        return int(request.num_prompt_tokens)
    if request.prompt_token_ids is not None:
        return len(request.prompt_token_ids)
    if request.prompt_embeds is not None:
        return int(request.prompt_embeds.shape[0])
    raise ValueError("Scheduled request has neither prompt tokens nor embeddings")


def execution_phase(scheduler_output, requests):
    """Use prompt progress, including fresh cached-prefix/resumed metadata.

    A mixed batch runs with prefill threads: one forward pass has one CPU
    thread setting. The token count alone never identifies the phase.
    """
    if scheduler_output.total_num_scheduled_tokens == 0:
        return "idle"
    new = {r.req_id: r for r in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    computed = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
    prefill = decode = False
    for req_id, count in scheduler_output.num_scheduled_tokens.items():
        if count <= 0:
            continue
        request = new.get(req_id)
        if request is not None:
            done = int(request.num_computed_tokens)
        else:
            request = requests.get(req_id)
            if request is None:
                raise ValueError(f"No state for scheduled request {req_id!r}")
            done = int(computed.get(req_id, request.num_computed_tokens))
        if done < _prompt_length(request):
            prefill = True
        else:
            decode = True
    if prefill and decode:
        return "mixed"
    if prefill:
        return "prefill"
    if decode:
        return "decode"
    raise ValueError("Nonempty scheduler output has no scheduled request tokens")
