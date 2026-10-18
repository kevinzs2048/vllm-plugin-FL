#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 12 "$@"

_dev
_ready
_source_validated
_run bash /work/scripts/run_bench.sh minimal "$E/minimal.json" --repeats 5
_pass
