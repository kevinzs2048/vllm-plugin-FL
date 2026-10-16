#!/bin/bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/release.env"
WORK_ROOT="${WORK_ROOT:-$(cd "$HERE/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/models/MiniCPM-V-4.7-1B-W8A8-FlagOS}"
CONTAINER="${CONTAINER:-flagos-workshop1018}"
test "$(sudo -n docker image inspect "$CIX_IMAGE_REF" --format '{{.Id}}')" = "$CIX_IMAGE_ID"
if sudo -n docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "ERROR: container $CONTAINER already exists; inspect it and exec into it, do not recreate over an existing container" >&2
  exit 1
fi
mkdir -p "$WORK_ROOT/cache" "$WORK_ROOT/evidence"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
SDK_ARGS=()
if [ -x "$WORK_ROOT/assets/llvm/bin/clang++" ]; then
  SDK_ARGS=(-v "$WORK_ROOT/assets/llvm:/llvm:ro")
fi
sudo docker run --detach --name "$CONTAINER" --network host \
 --cpuset-cpus 0,1,6,7,8,9,10,11 --entrypoint /bin/bash \
 -e FLAGOS_PROFILE=1b -e OMP_PROC_BIND=false \
 -v "$WORK_ROOT:/work" -v "$MODEL_DIR:/model:ro" \
 -v "$WORK_ROOT/cache:/cache" \
 "${SDK_ARGS[@]}" \
 "$CIX_IMAGE_REF" -c 'exec sleep infinity'
