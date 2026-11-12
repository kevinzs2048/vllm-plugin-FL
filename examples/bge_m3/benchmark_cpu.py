# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU dense BGE-M3 benchmark with saved token fixtures and per-shape results.

The FlagOS profile selects the Arm I8MM path. It does not select an ISA from
CPU model names, remove model guards, or patch stock vLLM functions.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepare-inputs", type=Path)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512]
    )
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--backend", choices=["native-bf16", "flagos-w8"], default="native-bf16"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument("--cpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--scales", type=Path)
    parser.add_argument("--raw-library", type=Path)
    parser.add_argument("--norm-library", type=Path)
    parser.add_argument("--attention-library", type=Path)
    parser.add_argument("--short-compile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model.is_dir() or any(n < 2 or n > 8192 for n in args.lengths):
        parser.error("Use an existing BGE checkpoint and input lengths in [2,8192]")
    if args.prepare_inputs:
        if args.prepare_inputs.exists():
            parser.error(
                "Input fixture already exists; reuse it instead of overwriting"
            )
        return args
    if not args.inputs or not args.inputs.is_file() or not args.output:
        parser.error("Inference requires --inputs and a new --output directory")
    if args.output.exists() or not 1 <= args.threads <= 64:
        parser.error(
            "Output must not exist; this phase-worker profile supports 1..64 threads"
        )
    if not 0 < args.cpu_memory_utilization <= 1:
        parser.error("CPU memory utilization must be in (0,1]")
    if args.warmups < 1 or args.iterations < 2 or args.batch_size < 1:
        parser.error("Use positive batch/warmups and at least two measured iterations")
    if max(args.lengths) > args.max_model_len:
        parser.error("Input exceeds --max-model-len")
    if args.batch_size * max(args.lengths) > args.max_batched_tokens:
        parser.error(
            "A full batch must fit --max-batched-tokens; split long-input and batch tests"
        )
    if args.backend == "flagos-w8":
        if platform.system() != "Linux" or platform.machine().lower() != "aarch64":
            parser.error("This server profile is Linux aarch64 I8MM only")
        for name in ["scales", "raw_library", "norm_library"]:
            value = getattr(args, name)
            if value is None or not value.is_file():
                parser.error(f"FlagOS W8 requires existing --{name.replace('_', '-')}")
        if args.short_compile and (
            not args.attention_library or not args.attention_library.is_file()
        ):
            parser.error("--short-compile requires a matching --attention-library")
    return args


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed = (
        "FlagOS supports local document retrieval. CPU inference converts text into vectors. "
        * 2000
    )
    tokens = tokenizer.encode(
        seed, add_special_tokens=False, truncation=True, max_length=8192
    )
    rows = []
    for length in args.lengths:
        ids = [tokenizer.cls_token_id] + tokens[: length - 2] + [tokenizer.sep_token_id]
        assert len(ids) == length
        rows.append(dict(length=length, token_ids=ids))
    args.prepare_inputs.parent.mkdir(parents=True, exist_ok=True)
    with args.prepare_inputs.open("x") as file:
        file.write("".join(json.dumps(row) + "\n" for row in rows))
    print(args.prepare_inputs)


def configure(args):
    # Start this example in a fresh process, without a platform launcher that
    # rewrites these values. Keep the caller's wait/binding/cache policy.
    for key in list(os.environ):
        if key.startswith(("FL_BGE_", "FL_CPU_", "MINICPM_", "FLAGGEMS_ARM_")):
            del os.environ[key]
    os.environ.update(
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        VLLM_CPU_OMP_THREADS_BIND="nobind",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_PLUGINS="fl" if args.backend == "flagos-w8" else "",
    )
    if args.backend == "flagos-w8":
        os.environ.update(
            FLAGGEMS_VENDOR="arm",
            TRITON_CPU_BACKEND="1",
            TRITON_BACKENDS_IN_TREE="1",
            FL_CPU_UNIPROC="1",
            FL_CPU_PREFILL_THREADS=str(args.threads),
            FL_CPU_DECODE_THREADS=str(args.threads),
            FL_BGE_M3_ARM="1",
            FL_BGE_QUANT="w8a8-sq-fit",
            FL_BGE_QUANT_BACKEND="i8mm",
            FL_BGE_SQ_FIT=str(args.scales.resolve()),
            FL_BGE_SME_THREADS=str(args.threads),
            FL_BGE_LINEAR="native",
            FL_BGE_ATTENTION="native",
            FL_BGE_NORM="native",
            FL_BGE_NORM_THREADS=str(args.threads),
            FL_BGE_GELU_THREADS=str(args.threads),
            FL_BGE_RESIDUAL_NORM="raw",
            FL_BGE_GELU="native",
            FL_BGE_FUSED_NORM_MAX_M="8192",
            FL_BGE_GELU_MAX_M="128",
            FL_BGE_I8MM_KERNEL="fused",
            FL_BGE_I8MM_ORDER="auto",
            FL_BGE_W8_ADD_NORM="1",
            FL_BGE_W8_ADD_NORM_MAX_M="128",
            FL_BGE_W8_FFN_RAW="1",
            FLAGGEMS_ARM_I8MM_TENSOR_RAW="1",
            FLAGGEMS_ARM_W8G32_I8MM_LIBRARY=str(args.raw_library.resolve()),
            FLAGGEMS_ARM_ELEMENTWISE_LIBRARY=str(args.norm_library.resolve()),
            FL_BGE_SHORT_ENCODER_COMPILE="1" if args.short_compile else "0",
        )
        if args.short_compile:
            os.environ.update(
                FL_BGE_SHORT_ENCODER_FASTPATH="1",
                FL_BGE_SHORT_ENCODER_SPECIALIZE="1",
                FL_BGE_ATTENTION_BRIDGE_LIBRARY=str(args.attention_library.resolve()),
            )


def run(args):
    configure(args)
    import numpy as np
    import torch
    from vllm import LLM
    from vllm.config import PoolerConfig

    if args.backend == "flagos-w8":
        from flag_gems.runtime.backend._arm.fused.w8a8_group_i8mm import _require_i8mm

        _require_i8mm()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    args.output.joinpath("arguments.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
    )
    architecture = (
        "FlagOSBgeM3QuantModel"
        if args.backend == "flagos-w8"
        else "BgeM3EmbeddingModel"
    )
    kwargs = {}
    if args.backend == "flagos-w8":
        kwargs["worker_cls"] = "vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker"
    started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        hf_overrides={"architectures": [architecture], "head_dtype": "model"},
        runner="pooling",
        dtype="bfloat16",
        quantization=None,
        # vLLM 0.24 uses this historical name for its CPU memory budget too.
        gpu_memory_utilization=args.cpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.batch_size,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        distributed_executor_backend="uni",
        pooler_config=PoolerConfig(
            task="embed", seq_pooling_type="CLS", use_activation=True
        ),
        **kwargs,
    )
    setup_seconds = time.perf_counter() - started
    fixtures = {
        row["length"]: row["token_ids"]
        for row in map(json.loads, args.inputs.read_text().splitlines())
    }
    for length in args.lengths:
        ids = fixtures[length]
        if len(ids) != length:
            raise ValueError("Fixture length mismatch")
        prompts = [{"prompt_token_ids": list(ids)} for _ in range(args.batch_size)]
        for _ in range(args.warmups):
            llm.embed(prompts, use_tqdm=False)
        samples = []
        cpu_start = time.process_time()
        for _ in range(args.iterations):
            begin = time.perf_counter()
            values = llm.embed(prompts, use_tqdm=False)
            vectors = np.asarray(
                [value.outputs.embedding for value in values], dtype=np.float32
            )
            samples.append(time.perf_counter() - begin)
        cpu_seconds = time.process_time() - cpu_start
        assert vectors.shape == (args.batch_size, 1024)
        assert np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=5e-3)
        np.save(args.output / f"embeddings-{length}.npy", vectors)
        result = dict(
            backend=args.backend,
            length=length,
            batch_size=args.batch_size,
            threads=args.threads,
            vector_norms=np.linalg.norm(vectors, axis=1).tolist(),
            torch_threads_after_request=torch.get_num_threads(),
            warmups_excluded=args.warmups,
            setup_seconds_excluded=setup_seconds,
            iterations=args.iterations,
            samples_seconds=samples,
            mean_ms=float(np.mean(samples) * 1000),
            p50_ms=float(np.percentile(samples, 50) * 1000),
            p90_ms=float(np.percentile(samples, 90) * 1000),
            p99_ms=float(np.percentile(samples, 99) * 1000),
            documents_per_second=args.iterations * args.batch_size / sum(samples),
            input_tokens_per_second=args.iterations
            * args.batch_size
            * length
            / sum(samples),
            process_cpu_seconds=cpu_seconds,
            cpu_core_equivalent=cpu_seconds / sum(samples),
            peak_main_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            scope="Pretokenized embedding API to host vectors; excludes loading/JIT/warmup/tokenization/HTTP/search. CPU calls are synchronous.",
        )
        args.output.joinpath(f"result-{length}.json").write_text(
            json.dumps(result, indent=2)
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "samples_seconds"}),
            flush=True,
        )


if __name__ == "__main__":
    args = arguments()
    if args.prepare_inputs:
        prepare(args)
    else:
        run(args)
