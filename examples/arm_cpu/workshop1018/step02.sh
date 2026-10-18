#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 02 "$@"

_host
_run bash "$ROOT/scripts/check_preloaded.sh"
_run bash "$ROOT/scripts/check_prebuilt.sh"
_pass
