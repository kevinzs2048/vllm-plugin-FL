# Copyright (c) 2026 BAAI. All rights reserved.
"""Request-aware CPU phase selection, independent of model operators."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadPolicy:
    prefill: int
    decode: int
    small_prefill_tokens: int = 0
    small_prefill_threads: int = 0

    def __post_init__(self):
        if not 1 <= self.prefill <= 64 or not 1 <= self.decode <= 64:
            raise ValueError("Prefill/decode thread counts must be in [1, 64]")
        if (
            (self.small_prefill_tokens == 0) != (self.small_prefill_threads == 0)
            or self.small_prefill_tokens < 0
            or not 0 <= self.small_prefill_threads <= 64
        ):
            raise ValueError(
                "Small-prefill token threshold and threads must both be positive or both zero"
            )

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
            int(env.get("FL_CPU_SMALL_PREFILL_TOKENS", "0")),
            int(env.get("FL_CPU_SMALL_PREFILL_THREADS", "0")),
        )

    def threads_for(self, phase, tokens=None):
        if (
            phase == "prefill"
            and self.small_prefill_tokens
            and tokens is not None
            and 0 < tokens <= self.small_prefill_tokens
        ):
            return self.small_prefill_threads
        if phase in ("prefill", "mixed"):
            return self.prefill
        if phase == "decode":
            return self.decode
        raise ValueError(f"No thread policy for phase {phase!r}")


@dataclass(frozen=True)
class BlocktimePolicy:
    """Opt-in libomp spin window for short prefill; milliseconds, zero allowed."""

    max_tokens: int
    short_ms: int
    default_ms: int

    def __post_init__(self):
        if (
            type(self.max_tokens) is not int
            or self.max_tokens <= 0
            or any(
                type(x) is not int or not 0 <= x < 2147483
                for x in [self.short_ms, self.default_ms]
            )
        ):
            raise ValueError("Invalid OpenMP blocktime policy")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        keys = (
            "FL_CPU_SHORT_BLOCKTIME_TOKENS",
            "FL_CPU_SHORT_BLOCKTIME_MS",
            "FL_CPU_BLOCKTIME_MS",
        )
        if not any(key in env for key in keys):
            return None
        if not all(key in env for key in keys):
            raise ValueError("Specify all three OpenMP blocktime policy options")
        # libomp APIs follow the units selected by KMP_BLOCKTIME.
        if env.get("KMP_BLOCKTIME", "").strip().lower().endswith("us"):
            raise ValueError(
                "Millisecond phase policy requires KMP_BLOCKTIME in ms; use 0 or 0ms"
            )
        return cls(*(int(env[key]) for key in keys))

    def blocktime_for(self, phase, tokens):
        if phase == "prefill" and tokens is not None and 0 < tokens <= self.max_tokens:
            return self.short_ms
        return self.default_ms


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
