"""Create isolated CPython prefixes using locked release dependency assets."""
from pathlib import Path
import shutil
import os

base = Path('/opt/flagos-runtime')
packages = base/'venv/lib/python3.11/site-packages'
for name in ['native', 'dev']:
    target = Path('/work/envs')/name
    if target.exists():
        # Replace only the empty venv left by our failed ensurepip preflight.
        if name == 'dev' and (target/'pyvenv.cfg').exists() and not (target/'lib/python3.11/encodings').exists():
            shutil.rmtree(target)
        else:
            raise FileExistsError(target)
    shutil.copytree(base/'cpython', target, symlinks=True)
    site = target/'lib/python3.11/site-packages'
    site.mkdir(parents=True,exist_ok=True)
    for item in packages.iterdir():
        if item.name.startswith(('flag_gems', 'vllm_fl', 'vllm_plugin_fl', 'triton', '_triton')):
            continue
        if item.suffix == '.pth':
            continue
        destination = site/item.name
        if not destination.exists():
            destination.symlink_to(item, target_is_directory=item.is_dir())
    for script in ['cmake','ninja']:
        origin = base/'venv/bin'/script
        if origin.exists():
            (target/'bin'/script).symlink_to(origin)
    print('CREATED',target,flush=True)
