"""Verify the participant entrypoints and their exact r2 helper contract."""
from pathlib import Path
import hashlib,json,sys
r=Path(sys.argv[1]).resolve()
m=json.loads((r/'STEPS.lock.json').read_text())
assert m['revision']=='workshop-step-scripts-20261018-r1'
assert hashlib.sha256((r/'ONLINE-INPUTS.lock.json').read_bytes()).hexdigest()==m['online_lock_sha256'],'Wrong underlying helper package'
for n,h in m['files'].items():
 p=(r/n).resolve()
 assert p.is_relative_to(r) and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,'Missing/modified step file: '+n
