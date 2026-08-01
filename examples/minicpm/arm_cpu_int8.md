# ARM CPU INT8

The FL plugin provides two W8A8 backends and one torch-native W8A16 fallback
on AArch64 CPUs with dot-product, i8mm, and BF16 extensions:

- `tleraw` (default): FlagTree TLE-raw operator backed by KleidiAI. Decode
  (`M == 1`) uses dot-product GEMV and prefill (`M > 1`) uses i8mm GEMM.
- `kleidiai`: calls the same packaged KleidiAI kernels through a ctypes custom
  op. This is useful when TLE compilation is unavailable.
- `torchpack`: uses `torch._weight_int8pack_mm`; activations remain BF16, so
  this mode is W8A16 rather than W8A8.

Weights are quantized online from the BF16 checkpoint and packed once during
model loading. The inference path does not materialize BF16 weights.

## Build the packaged W8A8 library

The AArch64 wheel contains `vllm_fl/ops/libkai_w8a8.so`. Rebuild it whenever
`cpu_int8_kai_wrapper.c` or the selected KleidiAI revision changes:

```bash
bash tools/build_arm_int8_assets.sh /path/to/kleidiai
```

To build into a staging directory without modifying the package tree:

```bash
bash tools/build_arm_int8_assets.sh /path/to/kleidiai /tmp/fl-w8-assets
```

The script prints the KleidiAI Git revision. Record that revision when the
library is refreshed. The current packaged library was built from
`a2abc4d3fb4669dcd30bcad8cd6fd4c232f64924`.

`libkai_w8a8.so` is the only packaged native W8A8 asset. The generated TLE
kernel resolves its KleidiAI compute symbols from this process-global shared
library; no separate relocatable ukernel object is required.

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

W4A8 and W8A8 TLE currently lower through the same compiler symbol with
different packed-weight layouts. Both modules can be imported together, but a
single process may activate only one of them. Restart the worker to switch
between W4A8 and W8A8.
