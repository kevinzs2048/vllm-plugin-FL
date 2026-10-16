#!/bin/bash
set -euo pipefail
E="${1:?Usage: validate_source.sh /work/evidence/new-validation-directory}"
[[ "$E" == /work/evidence/* ]] || { echo 'ERROR: output must be inside /work/evidence' >&2; exit 1; }
mkdir "$E"  # A repeated invocation must never overwrite previous evidence.
step() {
 local name="$1"; shift
 printf '%q ' "$@" > "$E/$name.command.txt"; printf '\n' >> "$E/$name.command.txt"
 "$@" > "$E/$name.log" 2>&1
 printf '%s PASS %s\n' "$(date -Is)" "$name" | tee -a "$E/progress.log"
}
step compiler bash /work/scripts/dev_python.sh /work/scripts/smoke_tree.py
step libraries bash /work/scripts/dev_python.sh /work/scripts/inspect_nokai.py
step numeric bash /work/scripts/dev_python.sh /work/scripts/test_w8_standalone.py --triton --output "$E/numeric.json"
step routes bash /work/scripts/run_raw_audit.sh "$E/routes.json"
step model env FLAGOS_FONT_PATH=/work/assets/fonts/DejaVuSans.ttf MINICPM_KVCACHE_BYTES=268435456 VLLM_ENABLE_V1_MULTIPROCESSING=0 bash /work/scripts/dev_python.sh /opt/flagos-runtime/share/flagos-minicpmv47/validation/model_smoke.py --model /model --worker phase --vision --repeats 1 --package-site /work/envs/dev/lib/python3.11/site-packages --output "$E/model.json"
bash /work/scripts/dev_python.sh - "$E" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for n in ('numeric','routes','model'):
 d=json.loads((p/(n+'.json')).read_text())
 assert d['complete'] and d['passed'],n
print('SOURCE_VALIDATION_PASS',p)
PY
