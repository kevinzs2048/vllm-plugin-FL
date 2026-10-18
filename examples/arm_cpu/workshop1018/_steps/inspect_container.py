"""Refuse stopping or entering an unrelated container with a matching name."""
from pathlib import Path
import json,subprocess,sys
r=Path(sys.argv[1]).resolve();name=sys.argv[2]
x=json.loads(subprocess.check_output(['sudo','-n','docker','inspect',name],text=True))[0]
assert x['Image']=='sha256:72d655d6bf308074aa53374ae32e7f1596d0493b81c1960df0209994bc54816b','Wrong image: '+name
mounts={m['Destination']:m for m in x['Mounts']}
assert Path(mounts['/work']['Source']).resolve()==r,'Unrelated /work mount: '+name
assert Path(mounts['/model']['Source']).resolve()==r/'models/MiniCPM-V-4.7-1B-W8A8-FlagOS','Wrong model: '+name
assert mounts['/model']['RW'] is False,'Model must be read-only'
print('WORKSHOP_CONTAINER_PASS',name,x['State']['Status'])
