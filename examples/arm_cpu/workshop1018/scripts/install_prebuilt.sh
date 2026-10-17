#!/bin/bash
set -euo pipefail
test -f /work/prebuilt/manifest.json
test -x /work/envs/dev/bin/python3.11
# Fail before pip modifies anything if the archive is incomplete or ABI differs.
bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py preflight /work
mkdir -p /work/evidence/prebuilt-install
LOG=$(mktemp /work/evidence/prebuilt-install/install-XXXXXXXX.log)
START=$(date +%s)
(
  set -euo pipefail
  bash /work/scripts/dev_python.sh -m pip install --no-index --no-deps \
    --no-compile --target /work/envs/dev/lib/python3.11/site-packages \
    /work/prebuilt/wheels/*.whl
  bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py installed /work
) 2>&1 | tee "$LOG"
printf 'PREBUILT_INSTALL_PASS elapsed_seconds=%s log=%s\n' "$(( $(date +%s) - START ))" "$LOG"
