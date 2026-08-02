# ARM CPU INT8

The FL plugin provides two W8A8 backends and one torch-native W8A16 fallback
on AArch64 CPUs with dot-product, i8mm, and BF16 extensions:

- `tleraw` (default): FlagTree TLE-raw operator backed by KleidiAI. Decode
  (`M == 1`) uses dot-product GEMV and prefill (`M > 1`) uses i8mm GEMM.
- `kleidiai`: calls the same FlagTree-built KleidiAI runtime through a ctypes custom
  op. This is useful when TLE compilation is unavailable.
- `torchpack`: uses `torch._weight_int8pack_mm`; activations remain BF16, so
  this mode is W8A16 rather than W8A8.

Weights are quantized online from the BF16 checkpoint and packed once during
model loading. The inference path does not materialize BF16 weights.

## Native source ownership and build

The vLLM plugin contains no native binary or copied KleidiAI source.
`flagtree-cpu` carries the pinned source subset, Apache-2.0 license, exact
upstream revision, integration wrapper, and build rule. On first use FlagTree
builds a content-addressed shared library under
`~/.cache/triton/kleidiai/`; subsequent workers reuse it.

No external KleidiAI checkout or prebuilt `.so` is required. A C compiler is
required because Triton CPU already compiles host kernels at runtime.
`TRITON_KLEIDIAI_CACHE_DIR` may override the native cache directory.

## Run

```bash
VLLM_PLUGINS=fl FL_CPU_INT8=1 FL_CPU_INT8_BACKEND=tleraw \
  OMP_NUM_THREADS=8 vllm serve /path/to/model \
  --dtype bfloat16 --trust-remote-code
```

Controls:

- `FL_CPU_INT8_BACKEND=tleraw|kleidiai|torchpack`: select the implementation;
  `tleraw` is the default.
- `FL_INT8_LMHEAD=1`: include a compatible language-model head; off by default.
- `FL_CPU_INT8_STRICT=0`: fall back to BF16 if packing an eligible linear
  fails; strict failure is the default.

W4A8 and W8A8 have separate compiler ops and runtime symbols. Both modules can
be imported and executed in one process without sharing a packed-weight ABI.
