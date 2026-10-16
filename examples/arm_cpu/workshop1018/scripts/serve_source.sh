#!/bin/bash
set -euo pipefail
export VLLM_ENABLE_V1_MULTIPROCESSING=0
exec bash /work/scripts/dev_python.sh -m vllm.entrypoints.cli.main serve /model \
  --served-model-name minicpm-v47-1b --trust-remote-code --dtype bfloat16 \
  --max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 1 \
  --worker-cls vllm_fl.worker.cpu_phase_worker.CPUPhaseWorker \
  --distributed-executor-backend uni --enforce-eager --no-enable-prefix-caching \
  --kv-cache-memory-bytes 268435456 --mamba-ssm-cache-dtype float32 \
  --generation-config vllm --limit-mm-per-prompt '{"image":1,"video":0}' \
  --host 127.0.0.1 --port 8000
