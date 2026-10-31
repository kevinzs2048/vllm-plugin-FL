# MiniCPM-V 4.7 on Arm CPU

This branch layers optional CPU runtime and quantized-operator integration on
community model adaptation PR #567. The two dependency commits retain their
original author and cherry-pick attribution. They are separate from the ARM
change set; drop those dependency commits when the model adaptation is merged.
No separate HF processor, vision implementation, or model registry is added.

The engine target is vLLM 0.24.0 CPU. Install compatible FlagGems and FlagTree-CPU
builds with the CPU native-call support described below. Ordinary Triton kernels
and CPU TLE Raw native leaves are distinct implementations. Raw arithmetic
may remain in compiled C++/Neon; it does not imply that native compilation or
KleidiAI dependencies disappear.

## Optional execution

- `FL_ARM_NATIVE=1`: enable the Arm quantized lifecycle and GDN runtime hooks.
- `FL_ARM_OPERATOR_BACKEND=raw`: use cached CPU TLE Raw entries (the default);
  `native` explicitly selects the comparison route.
- `FL_ARM_GEMMA_BACKEND=triton`: select prepared ordinary Triton Gemma math.
- `FL_INT8_LMHEAD=1`: prepare the stored floating-point LM Head for W8 execution.
- `FL_CPU_UNIPROC=1`: opt into the CPU single-worker platform; use TP=PP=1.

For distinct phase threads, select
`--worker-cls vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker`, set
`VLLM_CPU_OMP_THREADS_BIND=nobind`, `FL_CPU_PREFILL_THREADS`, and
`FL_CPU_DECODE_THREADS`. Mixed batches use the prefill setting; phase selection
uses request progress and does not equate a one-token prompt with decode.
Without these explicit settings, the normal CPU platform/thread policy stays
in effect. The worker supports vLLM 0.24's V1 runner, TP=PP=1.

The same worker can optionally select fewer threads for short embedding/prefill
requests. Set both `FL_CPU_SMALL_PREFILL_TOKENS` and
`FL_CPU_SMALL_PREFILL_THREADS`; omitting both preserves the existing policy.
The threshold applies to total scheduled tokens in a prefill-only batch. Mixed
batches and decode retain their normal settings.

On LLVM/Intel libomp, an optional short-prefill waiting policy requires all three
variables: `FL_CPU_SHORT_BLOCKTIME_TOKENS`, `FL_CPU_SHORT_BLOCKTIME_MS`, and
`FL_CPU_BLOCKTIME_MS`. Values are milliseconds; `KMP_BLOCKTIME` must not select
microsecond units. For example, a threshold of 32 with waits of 1 and 0 requests
1 ms for short prefill and 0 ms otherwise. The worker loads Torch's own OpenMP
runtime, checks the applied values, and avoids setting an unchanged value.
GNU libgomp supports phase threads but not this blocktime feature. These are
explicit tuning controls, not model-independent performance defaults; retain
the measured configuration and CPU-time cost for the particular workload.

## Quantization contracts

G32 means 32 consecutive K-dimension weights per quantization group.
The optional G32 and historical W8 native kernels use dynamic asymmetric A8.
The community symmetric W8 adapter has a different activation contract; this
branch preserves its selection and does not silently replace it.

FlagGems has mutually exclusive standalone W8/GDN and KAI G32/MoE build modes.
The standalone W8 mode needs no KleidiAI. The 35B G32/MoE mode still needs the
specified KAI library and the compatible Torch packed-expert ABI. Do not mix
packed weights from the two implementations or load both libraries in one
process. See FlagGems `src/flag_gems/runtime/backend/_arm/tle_raw/README.md`
and `tools/build_arm_cpu.py`.

For packed W4 experts, the shared model loader can unpack experts or optionally
stream them with `MINICPM_STREAM_EXPERT_LOADING=1`. Optional mmap caching uses
`MINICPM_EXPERT_MMAP_DIR`; its layout/checkpoint identity and packed-file SHA256
are validated before reuse. Hashing runs during startup, outside inference timing.
Checkpoint shard identity uses file metadata; keep both checkpoint and cache
immutable while in use. Dense models reject expert streaming. Recurrent GDN state remains FP32.
The direct MoE path retains explicit M=1, shape and TP/EP eligibility guards.

GDN prefill selection is owned by FlagGems. The plugin preserves its existing
ordinary Triton convolution default through
`FLAGGEMS_GDN_CONV_PREFILL_TRITON=1`; set it to `0` for the native/Raw convolution.
The older `MINICPM_GDN_CONV_PREFILL_IMPL` option is translated at initialization.

Tests follow the existing directories: phase threads in `tests/unit_tests/worker`,
checkpoint/loading contracts in `tests/unit_tests/quantization`, and runtime
norm/RNG bindings in `tests/unit_tests/patches`. No benchmark or installation
probe scripts are included in this plugin contribution.

Initialize weight packing and prepared kernel entries before graph capture.
FakeTensor support on the FlagGems inference operations is not a claim that
all vLLM compilation, video, speculative or prefix-cache configurations pass.
The final integrated source still needs fresh end-to-end validation on both
M5 and CIX P1 before release/performance publication. Unit and bridge results
are recorded separately from historical full-model measurements.

Quantization-generation scripts, weights, evaluation data, profiling fixtures,
and DSpark experiments are outside this contribution.
