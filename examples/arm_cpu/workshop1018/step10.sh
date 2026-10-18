#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 10 "$@"

_host
_new_response "$RUN_DIR/image-response.json"
_owned_container "$HTTP_CONTAINER"
_run python3 "$ROOT/scripts/http_request.py" --image "$ROOT/scripts/example.png" --text '这张图片的主要颜色是什么？' --output "$RUN_DIR/image-response.json"
sudo -n docker logs "$HTTP_CONTAINER" > "$RUN_DIR/server-full.log" 2>&1
printf 'Saved full server log: %s\n' "$RUN_DIR/server-full.log" | tee -a "$LOG"
_pass
