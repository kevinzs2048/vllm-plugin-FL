#!/bin/bash
set -euo pipefail
MODE="${1:?stock|minimal|flagos|flagos-dev}"
REPORT="${2:?new output JSON}"
RT=/opt/flagos-runtime
if [[ "$MODE" == flagos || "$MODE" == flagos-dev ]]; then
  source /work/scripts/triton-tle.env
fi
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=1000000 KMP_BLOCKTIME=5
export MIMALLOC_PURGE_DELAY=-1 TCMALLOC_RELEASE_RATE=0
export LD_PRELOAD="$RT/lib/libtcmalloc_minimal.so.4"
export LD_LIBRARY_PATH="$RT/venv/lib/python3.11/site-packages/torch/lib:$RT/lib:$RT/cpython/lib"
export VLLM_CPU_OMP_THREADS_BIND=nobind VLLM_ENABLE_V1_MULTIPROCESSING=0
unset VLLM_CPU_KVCACHE_SPACE
shift 2
if [[ "$MODE" == flagos-dev || "$MODE" == flagos ]]; then
  if [ -f /work/envs/dev/PREBUILT-INSTALLED.json ]; then
    export BUILD_VARIANT=prebuilt
  else
    export BUILD_VARIANT=source-built
  fi
  exec bash /work/scripts/dev_python.sh /work/scripts/bench.py \
    --mode flagos --output "$REPORT" "$@"
else
  export PYTHONHOME=/work/envs/native
  export PYTHONPATH="$PYTHONHOME/lib/python3.11/site-packages"
  export VLLM_PLUGINS='' VLLM_TARGET_DEVICE=cpu FL_INT8_LMHEAD=0
  export MODEL_PLUGIN_SOURCE=/work/source/vllm-plugin-FL/vllm_fl
  exec "$PYTHONHOME/bin/python3.11" /work/scripts/bench.py \
    --mode "$MODE" --output "$REPORT" "$@"
fi
