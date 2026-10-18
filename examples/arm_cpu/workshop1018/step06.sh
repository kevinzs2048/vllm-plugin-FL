#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 06 "$@"

_dev
[[ ! -e "$ROOT/envs" ]] || _fail 'envs already exists. Preserve it; see the PPT recovery appendix before retrying Step06.'
_run /opt/flagos-runtime/bin/flagos-python --profile 1b /work/scripts/prepare_env.py
_run bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py preflight /work
wheels=("$ROOT"/prebuilt/wheels/*.whl)
[[ ${#wheels[@]} == 3 && -f "${wheels[0]}" ]] || _fail 'Expected exactly 3 workshop wheels.'
_run bash /work/scripts/dev_python.sh -m pip install --no-index --no-deps --no-compile --target /work/envs/dev/lib/python3.11/site-packages "${wheels[@]}"
_run bash /work/scripts/dev_python.sh /work/scripts/prebuilt_tools.py installed /work
_pass
