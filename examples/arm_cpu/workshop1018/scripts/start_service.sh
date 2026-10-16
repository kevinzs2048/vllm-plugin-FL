#!/bin/bash
set -euo pipefail
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_ROOT/release.env"
source "$SCRIPT_ROOT/triton-tle.env"
MODEL_DIR="${MODEL_DIR:?Set the downloaded MiniCPM-V-4.7-1B W8A8 directory}"
WORK_ROOT="${WORK_ROOT:-$(cd "$SCRIPT_ROOT/.." && pwd)}"
CACHE_DIR="${CACHE_DIR:-$WORK_ROOT/cache/serve}"
RUN_DIR="${RUN_DIR:-$WORK_ROOT/evidence/http-$(date +%Y%m%d-%H%M%S)}"
CONTAINER="${CONTAINER:-minicpm-1b-workshop1018}"
# Check conflicts before creating artifacts or claiming readiness for another service.
python3 - <<'CHECK_PORT'
import socket
s=socket.socket()
try:
 s.bind(('0.0.0.0',8000))
except OSError as e:
 raise SystemExit('ERROR: host port 8000 is occupied; stop the existing service first: '+str(e))
finally:
 s.close()
CHECK_PORT
if sudo -n docker container inspect "$CONTAINER" >/dev/null 2>&1; then
 echo "ERROR: container $CONTAINER already exists; choose a new name or inspect/remove that workshop container" >&2
 exit 1
fi
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
mkdir -p "$CACHE_DIR"
mkdir "$RUN_DIR" # new directory only; preserve any previous run unchanged
test "$(sudo docker image inspect "$CIX_IMAGE_REF" --format '{{.Id}}')" = "$CIX_IMAGE_ID"
test -x "$WORK_ROOT/envs/dev/bin/python3.11"
sudo docker run --detach --name "$CONTAINER" --network host --entrypoint /bin/bash \
  --cpuset-cpus 0,1,6,7,8,9,10,11 \
  -e FLAGOS_PROFILE=1b -e OMP_PROC_BIND=false \
  -e FL_ARM_OPERATOR_BACKEND=raw -e FLAGGEMS_ARM_GDN_BACKEND=raw \
  -e FL_ARM_GEMMA_BACKEND=triton \
  -e FLAGGEMS_GDN_CONV_PREFILL_TRITON=1 -e MINICPM_GDN_CONV_PREFILL_IMPL=triton \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -v "$WORK_ROOT:/work" -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/cache" \
  "$CIX_IMAGE_REF" /work/scripts/serve_source.sh
ready=0
for attempt in $(seq 1 120); do
  if curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:8000/v1/models \
    > "$RUN_DIR/models.json"; then ready=1; break; fi
  sleep 5
done
sudo docker logs "$CONTAINER" > "$RUN_DIR/server.log" 2>&1
if [ "$ready" -ne 1 ]; then
 echo "ERROR: service not ready; logs retained at $RUN_DIR/server.log; inspect docker logs $CONTAINER before retrying" >&2
 exit 1
fi
sudo docker inspect "$CONTAINER" > "$RUN_DIR/container.json"
sudo docker top "$CONTAINER" -eo pid,args > "$RUN_DIR/processes.txt"
sudo docker inspect "$CONTAINER" --format '{{.State.Running}}' | grep -qx true
printf 'READY container=%s evidence=%s\n' "$CONTAINER" "$RUN_DIR"
