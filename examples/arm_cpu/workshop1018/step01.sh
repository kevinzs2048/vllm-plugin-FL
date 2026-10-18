#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 01 "$@"

_host
_run uname -m
_run free -h
_run df -h "$ROOT"
_run lscpu -e=CPU,CORE,ONLINE,MAXMHZ
_run python3 "$ROOT/scripts/check_platform.py"
_pass
