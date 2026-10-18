#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 15 "$@"

_host
_run python3 "$ROOT/_steps/wait_port.py"
_run python3 "$ROOT/scripts/test_workshop_boundaries.py" --output "$ROOT/evidence/boundaries-$RUN_ID.json"
printf 'All evidence retained under %s/evidence/. This step does not delete containers or results.\n' "$ROOT" | tee -a "$LOG"
_pass
