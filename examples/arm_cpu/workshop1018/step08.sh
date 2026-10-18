#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 08 "$@"

_host
_ready
_source_validated
_run bash "$ROOT/scripts/start_service.sh"
_pass
