#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 09 "$@"

_host
_new_response "$RUN_DIR/text-response.json"
_owned_container "$HTTP_CONTAINER"
_run curl --noproxy '*' --fail --max-time 5 http://127.0.0.1:8000/v1/models
_run python3 "$ROOT/scripts/http_request.py" --output "$RUN_DIR/text-response.json"
_pass
