"""Fail before creating containers on an incompatible board or full disk."""
from pathlib import Path
import os, platform, shutil, subprocess
root = Path(__file__).resolve().parent.parent
if Path('/.dockerenv').exists():
    raise SystemExit('ERROR: run this check on the P1 host, outside Docker')
if platform.machine() != 'aarch64':
    raise SystemExit('ERROR: this workshop requires Linux aarch64')
if any(c.isspace() for c in str(root)):
    raise SystemExit('ERROR: workshop root must not contain whitespace')
if root.is_symlink() or Path(os.environ.get('WORK_ROOT', root)).resolve() != root:
    raise SystemExit('ERROR: WORK_ROOT does not match the scripts directory')
selected = (0, 1, 6, 7, 8, 9, 10, 11)
allowed = set(os.sched_getaffinity(0))
for cpu in selected:
    p = Path(f'/sys/devices/system/cpu/cpu{cpu}/regs/identification/midr_el1')
    if cpu not in allowed or not p.exists() or ((int(p.read_text().strip(),16) >> 4) & 0xfff) != 0xd81:
        raise SystemExit(f'ERROR: CPU {cpu} is unavailable or is not Cortex-A720; revalidate affinity on this board')
mem = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemTotal:'))) * 1024
if mem < 30 * 1024**3:
    raise SystemExit('ERROR: this build was validated with at least 30 GiB usable host RAM')
# Source-build products need space in addition to the preloaded inputs.
free = shutil.disk_usage(root).free
if not (root/'envs/dev/bin/python3.11').exists() and free < 7*1024**3:
    raise SystemExit('ERROR: need at least 7 GiB free after preloading, before the fresh source build')
subprocess.run(['sudo','-n','docker','version','--format','{{.Server.Version}}'],check=True)
print('PLATFORM_CHECK_PASS', 'cpus='+','.join(map(str,selected)), 'free_GiB='+format(free/1024**3,'.2f'))
