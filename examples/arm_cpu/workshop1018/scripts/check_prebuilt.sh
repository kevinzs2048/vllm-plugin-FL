#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
python3 "$HERE/prebuilt_tools.py" verify "$ROOT"
echo PREBUILT_CHECK_PASS
