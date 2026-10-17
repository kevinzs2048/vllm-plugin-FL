"""Verify this workshop's supplemental binary bundle and installation contract."""
from pathlib import Path
import hashlib, importlib.metadata as md, json, platform, sys, time, zipfile, csv, io, base64

mode, arg = sys.argv[1:]
root = Path(arg).resolve()
manifest = json.loads((root/'prebuilt/manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = (root/name).resolve()
    assert path.is_relative_to(root) and path.is_file(), name
    assert path.stat().st_size == expected['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
for wheel in (root/'prebuilt/wheels').glob('*.whl'):
    with zipfile.ZipFile(wheel) as z:
        record = next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
        for name, digest, size in csv.reader(io.StringIO(z.read(record).decode())):
            if digest:
                data = z.read(name)
                actual = 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                assert actual == digest and len(data) == int(size), (wheel.name, name)
print('PREBUILT_FILES_PASS', manifest['bundle_id'], flush=True)
if mode == 'verify':
    sys.exit(0)
assert mode in ('preflight', 'installed'), mode
assert root == Path('/work'), root
assert platform.machine() == manifest['architecture'], platform.machine()
assert platform.python_version() == manifest['python'], platform.python_version()
for package in ('torch', 'vllm'):
    assert md.version(package) == manifest[package], (package, md.version(package))
site = root/'envs/dev/lib/python3.11/site-packages'
assert Path(sys.prefix) == root/'envs/dev', sys.prefix
receipt = root/'envs/dev/PREBUILT-INSTALLED.json'
if mode == 'preflight':
    assert not receipt.exists(), 'Already installed; refuse overwriting existing prefix'
    for prefix in ('triton', '_triton', 'flag_gems', 'vllm_fl', 'vllm_plugin_fl'):
        assert not list(site.glob(prefix+'*')), 'Nonempty component prefix: '+prefix
    print('PREBUILT_ABI_PASS', flush=True)
else:
    import torch, triton, flag_gems, vllm_fl
    for module in (triton, flag_gems, vllm_fl):
        assert Path(module.__file__).resolve().is_relative_to(site), module.__file__
    lib = site/'flag_gems/runtime/backend/_arm/lib'
    torch.ops.load_library(str(lib/'libflag_gems_arm_native.so'))
    assert (lib/'flag_gems_arm_silu.bc').is_file()
    report = {'complete': True, 'bundle_id': manifest['bundle_id'], 'installed_at': time.time(),
              'versions': {p: md.version(p) for p in ['torch', 'vllm', 'triton', 'flag_gems', 'vllm-plugin-fl']},
              'manifest_sha256': hashlib.sha256((root/'prebuilt/manifest.json').read_bytes()).hexdigest()}
    with receipt.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print('PREBUILT_IMPORT_AND_RAW_LOAD_PASS', flush=True)
