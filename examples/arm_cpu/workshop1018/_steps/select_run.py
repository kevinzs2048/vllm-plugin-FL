"""Persist the selected evidence run across host/container shell transitions."""
from pathlib import Path
import os,re,sys,tempfile
r=Path(sys.argv[1]); requested=sys.argv[2]
s=r/'.workshop';p=s/'active-run.txt'
old=p.read_text().strip() if p.exists() else 'run01'
run=requested or old
if not re.fullmatch(r'run[0-9]{2,4}',run):raise SystemExit('ERROR: use --run run02, run03, ...')
s.mkdir(exist_ok=True)
if not p.exists() or requested:
 with tempfile.NamedTemporaryFile('w',dir=s,prefix='active-',delete=False) as f:
  f.write(run+'\n');f.flush();os.fsync(f.fileno());tmp=Path(f.name)
 tmp.replace(p)
print(run)
