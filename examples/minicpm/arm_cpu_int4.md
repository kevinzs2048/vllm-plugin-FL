# ARM CPU INT4

The FL plugin consumes offline, channelwise W4A8 checkpoints through a
FlagTree TLE-raw operator on AArch64 CPUs with dot-product, i8mm, and BF16
extensions. It does not quantize BF16 model weights at load time. Checkpoint
weights must be signed INT4 values stored in `torch.int8`, with one scale per
output channel and dynamic per-token INT8 activations.

Runtime linear computation enters `create_cpu_kleidiai_w4a8_linear`.
FlagTree selects KleidiAI dot-product GEMV for decode (`M == 1`) and BF16-output
i8mm GEMM for prefill (`M > 1`). LHS packing is split over M and matmul is
split over N, matching the native KleidiAI channelwise scheduling.

The generated kernel specialization includes a cache identity derived from the
FlagTree runtime source and compiler identity, so stale Triton kernels are not
reused after the native implementation changes.

## Native source ownership and build

The vLLM plugin does not contain KleidiAI source, C wrappers, object files, or
shared libraries. `flagtree-cpu` carries the pinned minimal KleidiAI subset,
license, revision, integration wrappers, and deterministic build rule. On first
use it compiles a content-addressed runtime under
`~/.cache/triton/kleidiai/`; later processes reuse that cache.

No `FL_KAI_W4A8_DIR`, `KLEIDIAI_ROOT`, or prebuilt `.so` is required. A C
compiler is required because Triton CPU already compiles host kernels at
runtime. `TRITON_KLEIDIAI_CACHE_DIR` may override the native cache directory.

## Run

Quantize the model offline with a vLLM compressed-tensors channelwise W4A8
recipe, then enable the plugin with `VLLM_PLUGINS=fl`. Compile mode is on by
default. The relevant controls are:

- `FL_CPU_INT4=1`: require INT4; missing or invalid FlagTree support is a hard error.
- `FL_CPU_INT4=0`: retain the ARM CPU platform but use BF16 linears.
- `FL_CPU_INT4_BACKEND=tleraw`: the only supported INT4 runtime backend.
- `FL_CPU_UNIPROC=1`: opt into the faster in-process executor for a controlled
  single-worker deployment. The caller must configure OpenMP threads, affinity,
  and allocator preload. Otherwise vLLM's supported MP executor is retained.
- `FL_CPU_OMP_ACTIVE_WAIT=0`: disable the uniproc latency default
  `OMP_WAIT_POLICY=ACTIVE`. Active wait improved this machine's PP512/TG128 by
  about 0.8%, but keeps OpenMP workers busy between short kernels and therefore
  increases idle CPU usage and power.

For predictable libgomp initialization, production launch scripts should also
export `OMP_WAIT_POLICY=ACTIVE` before starting Python. The platform hook sets
the same value on a best-effort basis before model execution.

Example:

```bash
VLLM_PLUGINS=fl FL_CPU_UNIPROC=1 OMP_NUM_THREADS=8 OMP_WAIT_POLICY=ACTIVE \
  vllm serve /path/to/model --dtype bfloat16 --trust-remote-code
```

The plugin only selects checkpoints whose effective group size equals K.
Groupwise `group_size=32` checkpoints deliberately fall through to another
vLLM kernel. If `lm_head` should also use W4A8, it must be an explicit target
in the offline checkpoint and provide `lm_head.weight_scale`; there is no
runtime `FL_INT4_LMHEAD` conversion. Evaluate output quality before doing this:
the standard recipe ignores `lm_head` because channelwise INT4 can have
materially larger quantization error there.
