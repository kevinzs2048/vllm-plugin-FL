#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 05 "$@"

_host
[[ -t 0 && -t 1 ]] || _fail 'Step05 needs a normal interactive SSH terminal.'
if sudo -n docker container inspect "$DEV_CONTAINER" >/dev/null 2>&1; then
 _owned_container "$DEV_CONTAINER"
 [[ "$(sudo -n docker inspect "$DEV_CONTAINER" --format '{{.State.Running}}')" == true ]] || _fail 'The existing development container is stopped. Inspect it before restarting.'
 printf 'Re-entering the existing matching development container; no files are replaced.\n'
else
 _run env CONTAINER="$DEV_CONTAINER" bash "$ROOT/scripts/init_container.sh"
fi
_owned_container "$DEV_CONTAINER"
_pass
_enter_dev
