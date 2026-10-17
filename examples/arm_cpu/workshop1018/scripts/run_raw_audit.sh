#!/bin/bash
set -euo pipefail
REPORT="${1:?Use a new /work/evidence/*.json filename}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=8
export OMP_WAIT_POLICY=PASSIVE
if [ -f /work/envs/dev/PREBUILT-INSTALLED.json ]; then
  export BUILD_VARIANT=prebuilt
else
  export BUILD_VARIANT=source-built
fi
exec bash /work/scripts/dev_python.sh /work/scripts/bench_raw.py \
  --mode flagos --output "$REPORT" --repeats 0 --route-audit
