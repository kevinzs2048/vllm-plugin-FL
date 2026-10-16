#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WORK_ROOT="${WORK_ROOT:-$(cd "$HERE/.." && pwd)}"
source "$HERE/release.env"
python3 "$HERE/check_platform.py"
python3 "$HERE/verify_online_inputs.py"
test "$(sudo -n docker image inspect "$CIX_IMAGE_REF" --format '{{.Id}}')" = "$CIX_IMAGE_ID"
bash "$HERE/check_prebuilt.sh"
echo PRELOADED_CHECK_PASS
