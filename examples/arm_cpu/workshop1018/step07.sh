#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 07 "$@"

_dev
_ready
_run bash /work/scripts/validate_source.sh "$E"
_pass
