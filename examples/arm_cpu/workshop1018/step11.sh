#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 11 "$@"

if ((INSIDE)); then
 _dev
 _ready
 [[ ! -e "$E/stock.json" ]] || _fail "Existing $E/stock.json; preserve it."
 export VLLM_PLUGINS=""
 if _run bash /work/scripts/run_bench.sh stock "$E/stock.json" --repeats 0; then :; else
  printf 'Checking whether stock failure is the expected unsupported architecture...\n' | tee -a "$LOG"
 fi
 _run bash /work/scripts/dev_python.sh /work/scripts/check_stock.py "$E/stock.json"
 _pass
else
 _host
 [[ -t 0 && -t 1 ]] || _fail 'Step11 needs a normal interactive SSH terminal.'
 _owned_container "$HTTP_CONTAINER"
 _owned_container "$DEV_CONTAINER"
 _run sudo -n docker stop -t 30 "$HTTP_CONTAINER"
 _run sudo -n docker exec "$DEV_CONTAINER" bash /work/step11.sh --inside --run "$RUN_ID"
 _pass
 printf 'Stock check finished. Next: bash step12.sh in the development container.\n'
 _enter_dev
fi
