#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 13 "$@"

_dev
_ready
_source_validated
[[ -f "$E/minimal.json" ]] || _fail 'Run Step12 first.'
_run bash /work/scripts/run_bench.sh flagos-dev "$E/flagos.json" --repeats 5
_pass
