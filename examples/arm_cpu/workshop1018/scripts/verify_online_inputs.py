"""Verify directly downloaded online inputs without an LLVM SDK or full archive."""
from pathlib import Path
import hashlib,json,subprocess
r=Path(__file__).resolve().parent.parent
lock=json.loads((r/'ONLINE-INPUTS.lock.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
for n,digest in lock['helper_files'].items():
 p=r/n;assert p.is_file() and sha(p)==digest, 'Missing/modified Workshop helper: '+n
m=r/'models/MiniCPM-V-4.7-1B-W8A8-FlagOS'
for n,digest in lock['model_runtime_files'].items():
 p=m/n;assert p.is_file() and sha(p)==digest, 'Missing/mismatched checkpoint file: '+n
idx=json.loads((m/'model.safetensors.index.json').read_text())
for n in set(idx['weight_map'].values()):
 assert (m/n).resolve().is_relative_to(m.resolve()) and (m/n).is_file(),n
for name,commit in lock['sources'].items():
 p=r/'source'/name
 if not p.exists():
  print('SOURCE_NOT_DOWNLOADED',name,'(optional for FlagOS inference; plugin source required for minimal baseline)');continue
 assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==commit,name
 assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip(), 'Dirty source '+name
 print('SOURCE_PIN_PASS',name,commit)
print('ONLINE_INPUTS_CHECK_PASS',len(lock['model_runtime_files']),'model files; LLVM SDK not required for prebuilt route')
