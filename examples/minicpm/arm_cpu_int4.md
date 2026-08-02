# ARM CPU INT4

The FL plugin uses a FlagTree TLE-raw W4A8 operator on AArch64 CPUs with
dot-product, i8mm, and BF16 extensions. Runtime
linear computation enters `create_cpu_kleidiai_w4a8_linear`; FlagTree's
source-built runtime selects KleidiAI dot-product GEMV for decode (`M == 1`) and i8mm GEMM
for prefill (`M > 1`). Python does not dispatch on `M`, because vLLM CPU's
`DYNAMO_TRACE_ONCE` graph is reused after shape guards are removed.

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

Enable the plugin with `VLLM_PLUGINS=fl`. Compile mode is on by default. INT4 is
selected automatically when the installed FlagTree CPU package provides the
runtime sources; otherwise the clean-install default remains BF16. The relevant controls are:

- `FL_CPU_INT4=1`: require INT4; missing or invalid FlagTree support is a hard error.
- `FL_CPU_INT4=0`: retain the ARM CPU platform but use BF16 linears.
- `FL_CPU_INT4_BACKEND=tleraw`: the only supported INT4 runtime backend.
- `FL_INT4_LMHEAD=1`: include a compatible language-model head; off by default.
- `FL_CPU_INT4_STRICT=0`: allow an eligible linear that fails packing to fall
  back to BF16; strict failure is the default.
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
