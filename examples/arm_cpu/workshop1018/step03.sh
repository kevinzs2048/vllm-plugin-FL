#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/_steps/common.sh" 03 "$@"

_host
_run sudo -n docker image inspect "$CIX_IMAGE_REF" --format 'id={{.Id}} repo-digests={{json .RepoDigests}}'
[[ "$(sudo -n docker image inspect "$CIX_IMAGE_REF" --format '{{.Id}}')" == "$CIX_IMAGE_ID" ]] || _fail 'Image identity mismatch.'
_pass
