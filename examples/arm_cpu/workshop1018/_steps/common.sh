#!/usr/bin/env bash
set -Eeuo pipefail
STEP_ID="$1"; shift
_HELPER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$_HELPER_DIR/.." && pwd)
REQUESTED_RUN=""
INSIDE=0
while (($#)); do
  case "$1" in
    --run) (($#>=2)) || { echo 'ERROR: --run requires run01, run02, ...' >&2; exit 2; }; REQUESTED_RUN="$2"; shift 2 ;;
    --inside) [[ "$STEP_ID" == 11 ]] || { echo 'ERROR: internal option' >&2; exit 2; }; INSIDE=1; shift ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [[ -f /.dockerenv ]]; then
 CONTROL_PYTHON=(/opt/flagos-runtime/bin/flagos-python --profile 1b)
 [[ -x "${CONTROL_PYTHON[0]}" ]] || { echo "ERROR: use the fixed workshop container" >&2; exit 1; }
else
 CONTROL_PYTHON=(python3)
fi
"${CONTROL_PYTHON[@]}" "$ROOT/_steps/verify_steps.py" "$ROOT"
RUN_ID=$("${CONTROL_PYTHON[@]}" "$ROOT/_steps/select_run.py" "$ROOT" "$REQUESTED_RUN")
export WORK_ROOT="$ROOT"
if [[ -f /.dockerenv ]]; then
  export MODEL_DIR=/model
  WHERE=container
else
  WHERE=host
  source "$ROOT/scripts/activate_host.env"
fi
DEV_CONTAINER=flagos-workshop1018
HTTP_CONTAINER=minicpm-1b-workshop1018
[[ "$RUN_ID" == run01 ]] || HTTP_CONTAINER="$HTTP_CONTAINER-$RUN_ID"
export CONTAINER="$HTTP_CONTAINER"
export CACHE_DIR="$ROOT/cache/http-$RUN_ID"
export RUN_DIR="$ROOT/evidence/http-$RUN_ID"
E="$ROOT/evidence/workshop-$RUN_ID"
LOG_DIR="$ROOT/evidence/steps-$RUN_ID"
mkdir -p "$LOG_DIR"
LOG=$(mktemp "$LOG_DIR/step${STEP_ID}-${WHERE}-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
_failed() {
 local rc="$1" line="$2"
 trap - ERR
 printf '\nERROR Step%s exit=%s line=%s\nLog: %s\n' "$STEP_ID" "$rc" "$line" "$LOG" >&2
 printf 'Original results retained. Fix the error before continuing.\n' >&2
 exit "$rc"
}
trap '_failed $? $LINENO' ERR
_fail() { echo "ERROR: $*" >&2; return 1; }
_run() {
 { printf '+ '; printf '%q ' "$@"; printf '\n'; } | tee -a "$LOG"
 "$@" 2>&1 | tee -a "$LOG"
}
_host() {
 [[ "$WHERE" == host ]] || _fail "Step$STEP_ID must run on the P1 host. Type exit, then cd ~/workshop1018."
}
_dev() {
 [[ "$WHERE" == container && "$ROOT" == /work ]] || _fail "Step$STEP_ID must run in the development container at /work. Enter with Step05 or Step11."
 [[ -f /opt/flagos-runtime/share/flagos-minicpmv47/profiles/1b.env && -f /model/config.json ]] || _fail "Not the workshop development container."
 "${CONTROL_PYTHON[@]}" -c 'from pathlib import Path; a=Path("/proc/1/cmdline").read_bytes().split(b"\0"); assert a[:2]==[b"sleep",b"infinity"], "Use the development container, not the HTTP service container"'
}
_ready() {
 [[ -f "$ROOT/envs/dev/PREBUILT-INSTALLED.json" ]] || _fail 'Finish Step06 first.'
}
_source_validated() {
 "${CONTROL_PYTHON[@]}" - "$E" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for n in ('numeric','routes','model'):
 d=json.loads((p/(n+'.json')).read_text())
 assert d.get('complete') and d.get('passed'),n
PY
}
_owned_container() {
 "${CONTROL_PYTHON[@]}" "$ROOT/_steps/inspect_container.py" "$ROOT" "$1"
}
_new_response() {
 [[ ! -e "$1" && ! -e "${1%.json}-request.json" ]] || _fail "Existing response/request: $1. Preserve it and use the recovery instructions."
}
_enter_dev() {
 [[ -t 0 && -t 1 ]] || _fail 'Interactive terminal required. Use a normal SSH terminal, not a pipe.'
 printf '\nEntering development container. Commands: bash step06.sh / step07.sh, or step12.sh ...\n'
 printf 'Container directory: /work. Type exit only when the PPT requests returning to the host.\n\n'
 sudo -n docker exec -it -w /work "$DEV_CONTAINER" /bin/bash
 printf '\nReturned to the P1 host. Directory: %s\n' "$ROOT"
}
_pass() { printf '\nSTEP%s_PASS run=%s\nLog: %s\n' "$STEP_ID" "$RUN_ID" "$LOG" | tee -a "$LOG"; }
printf '\nStep%s  context=%s  run=%s\n' "$STEP_ID" "$WHERE" "$RUN_ID" | tee -a "$LOG"
