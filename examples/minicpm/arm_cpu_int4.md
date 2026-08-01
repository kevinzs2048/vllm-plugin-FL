# ARM CPU INT4

The FL plugin uses a FlagTree TLE-raw W4A8 operator on AArch64 CPUs with
dot-product, i8mm, and BF16 extensions. Runtime
linear computation enters `create_cpu_gemm_q4_0_v2_smmla_bf16`; its compiled C
backing selects KleidiAI dot-product GEMV for decode (`M == 1`) and i8mm GEMM
for prefill (`M > 1`). Python does not dispatch on `M`, because vLLM CPU's
`DYNAMO_TRACE_ONCE` graph is reused after shape guards are removed.

`TLE_CACHE_ABI` in `cpu_int4_tleraw.py` is part of the generated kernel
specialization. Triton CPU does not include registered external C source or
linked object contents in its cache key, so this value must be bumped whenever
the native wrapper or KleidiAI object changes.

## Build the KleidiAI microkernels

KleidiAI W4A8 code is not vendored here: the wheel ships only the two W4A8 C
sources this plugin owns. The unified ARM wheel is nevertheless
platform-specific because its W8A8 fallback library is packaged separately.
Build the W4A8 microkernels once from a FlagTree checkout:

```bash
bash python/scripts/build_kai_w4a8_assets.sh                      # clones KleidiAI
bash python/scripts/build_kai_w4a8_assets.sh --kleidiai-path DIR   # or reuse a clone
```

The script prints the two variables to export:

```bash
export FL_KAI_W4A8_DIR=...   # holds libkai_w4a8_ukernels.o
export KLEIDIAI_ROOT=...     # KleidiAI source root (headers)
```

`FL_KAI_W4A8_DIR` supplies the relocatable object appended to every compiled
Triton kernel's link line. `KLEIDIAI_ROOT` is needed only the first time the
plugin runs: it compiles `cpu_int4_pack.c` against the KleidiAI headers into a
small shared library, cached under `~/.cache/flagos-kai-w4a8/`, and loads it
through `ctypes` to pack weights at model-load time. That library is not on the
inference path. Set `FL_KAI_W4A8_PACK_SO` to supply a prebuilt one instead.

KleidiAI is Apache-2.0, Copyright Arm Limited; obtaining it is the deployer's
responsibility.

## Run

Enable the plugin with `VLLM_PLUGINS=fl`. ARM CPU INT4 and compile mode are on
by default. The relevant controls are:

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
