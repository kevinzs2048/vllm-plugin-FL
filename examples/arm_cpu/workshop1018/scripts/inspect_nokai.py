"""Inspect the actual loaded FlagGems library and cached Triton Raw entries."""
import json,subprocess
from pathlib import Path
from flag_gems.runtime.backend._arm.fused import native_ops,native_raw
ops=native_ops.arm_native_ops('w8a8',raw=True)
assert ops.implementation()=='standalone-neon'
library=Path(native_ops._library[0])
symbols=subprocess.check_output(['nm','-D',str(library)],text=True)
assert not any('kai_' in line for line in symbols.splitlines())
assert 'flag_gems_arm_w8_compute' in symbols
assert not Path('/kai-source').exists() and not Path('/kai-build').exists()
owners=native_raw._owners
report={'passed':True,'implementation':ops.implementation(),'library':str(library),
        'kai_symbols':[],'kai_mounts_present':False,
        'compiled_raw_kernel':owners[2].name,
        'compiled_dense_entries':[{'index':entry[0],'name':entry[1].name,'address':entry[3]} for entry in owners[4]],
        'scope':'FlagGems standalone W8/GDN operator library; does not assert PyTorch contains no KAI code'}
print(json.dumps(report,indent=2))
