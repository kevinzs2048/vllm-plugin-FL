#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 04 "$@"

_host
for repo in FlagGems flagtree-cpu vllm-plugin-FL; do
 [[ -e "$ROOT/source/$repo/.git" ]] || _fail "Missing source/$repo. Ask the organizer to preload the fixed sources."
done
_run python3 "$ROOT/scripts/verify_online_inputs.py"
_run ls -lh "$ROOT/prebuilt/wheels"
printf 'MODEL_DIR=%s\n' "$MODEL_DIR" | tee -a "$LOG"
_pass
