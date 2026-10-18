#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 14 "$@"

_dev
_ready
_run bash /work/scripts/dev_python.sh /work/scripts/compare_bench.py "$E/minimal.json" "$E/flagos.json" --output "$E/comparison.json"
_run "${CONTROL_PYTHON[@]}" /work/_steps/show_comparison.py "$E/comparison.json"
_pass
