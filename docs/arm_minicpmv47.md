# MiniCPM V 4.7 on ARM CPU

This opt-in integration targets vLLM **0.24.0** with MiniCPM V 4.7 35B-A3B
W4A8 G32 and 1B W8A8 checkpoints. Model/config/processor registration, weight
loading and vLLM cache/thread ABI bindings live in this plugin. FlagGems owns
all native arithmetic and ordinary CPU Triton kernels. No separate model
adapter or runtime C++ source compilation is required; vLLM sources are
unchanged.

## Dependencies and enablement

Install the corresponding FlagTree CPU (`triton_v3.7.x`, version 3.7.2),
FlagGems Python package and optional ARM native libraries. See
`FlagGems/cpp/arm/README.md` for the CMake/KleidiAI build and CPU TLE ABI.
Use the same Torch build during compilation and inference; its exported KAI
expert-packing ABI is required. Install Python wheels first, then install the
native libraries into that package prefix with CMake.

```sh
export FLAGGEMS_VENDOR=arm
export GEMS_VENDOR=arm
export VLLM_FL_PLATFORM=arm
export VLLM_TARGET_DEVICE=cpu
export FL_ARM_NATIVE=1
export FL_ARM_OPERATOR_BACKEND=native
export FL_ARM_GEMMA_BACKEND=triton
export FL_INT8_LMHEAD=1
export VLLM_CPU_OMP_THREADS_BIND=nobind
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V2_MODEL_RUNNER=0
```

`FL_ARM_OPERATOR_BACKEND=raw` selects cached CPU TLE entries for the remaining
native operators. It preserves their C++/Neon/KAI arithmetic. Ordinary Triton
Gemma normalization is selected independently by `FL_ARM_GEMMA_BACKEND=triton`;
`inherit` (the default) uses the selected native/Raw Gemma implementation.
The ordinary specialization supports BF16 hidden sizes 128/1024/2048 with
epsilon 1e-6; other eligible sizes/epsilon values keep the inherited path.
Unsupported dtypes/layouts keep vLLM's original implementation.

Use `worker_cls="vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker"` and
`distributed_executor_backend="uni"`. The tested configuration uses
`enforce_eager=True`, tensor parallel size 1, max sequences 1, FP32 mamba state,
and disabled prefix caching. Text, image input and OCR were checked on M5 Pro
and Cix P1. Video and other execution configurations are not covered by these
checks. Vision and embeddings remain BF16; `FL_INT8_LMHEAD=1` performs bounded
per-channel W8 preparation of a BF16 output head at load time.

## Thread and matrix policy

Set `FL_CPU_PREFILL_THREADS` and `FL_CPU_DECODE_THREADS` independently. The
worker identifies prefill from actual request/cache metadata and applies the
new count only when it changes. It updates Torch's real C binding and the
shared OpenMP runtime, because vLLM replaces the Python setter after binding.
The policy does not modify global defaults of other workers.

Tested profiles (machine policy, not universal defaults):

| Host | Model | Prefill | Decode |
| --- | --- | ---: | ---: |
| M5 Pro | 35B-A3B | 18 | 8 |
| M5 Pro | 1B | 8 | 8 |
| Cix P1 | 35B-A3B | 8 | 6 |
| Cix P1 | 1B | 8 | 4 |

Cix P1 has no SME2: keep `FL_ARM_SME_FRACTION=0`. On an SME2 host, an explicit
nonzero fraction prepares an additional dense prefill layout; decode retains
its Neon layout. Unsupported packed G32 projections are unpacked to signed
INT8 before stock processing. The activation contract follows vLLM's KAI
Dynamic4bit kernel, including its dynamic asymmetric A8 execution.

Expert packing/cache limits, affinity and OpenMP wait settings belong to the
machine launcher. Cache reuse checks the checkpoint identity and shard
metadata; it does not rehash checkpoint contents on every load. Benchmark
reports and quantization-generation scripts are maintained separately from
these source repositories.
