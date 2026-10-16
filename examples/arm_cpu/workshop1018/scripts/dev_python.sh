#!/bin/bash
set -euo pipefail
RT=/opt/flagos-runtime
export PYTHONHOME=/work/envs/dev
export PYTHONPATH="$PYTHONHOME/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$RT/venv/lib/python3.11/site-packages/torch/lib:$RT/cpython/lib:$RT/lib"
export LD_PRELOAD="$RT/lib/libtcmalloc_minimal.so.4"
export PATH="$PYTHONHOME/bin:/llvm/bin:/usr/bin:/bin"
export OMP_PROC_BIND=false
export TRITON_BACKENDS_IN_TREE=1 TRITON_CPU_BACKEND=1
export FLAGOS_CACHE_DIR=/cache/dev TRITON_CACHE_DIR=/cache/dev/triton
export TMPDIR=/cache/dev/tmp VLLM_CACHE_ROOT=/cache/dev/vllm
export TORCHINDUCTOR_CACHE_DIR=/cache/dev/inductor
export VLLM_RPC_BASE_PATH="$(mktemp -d /tmp/flagos-workshop.XXXXXX)"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
unset VLLM_CPU_KVCACHE_SPACE
source "$RT/share/flagos-minicpmv47/profiles/1b.env"
source /work/scripts/triton-tle.env
exec "$PYTHONHOME/bin/python3.11" "$@"
