"""Identity and completion metadata for a local KAI weight mmap cache."""

import hashlib
import json
import platform
from pathlib import Path

import torch


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    files = []
    for p in sorted(checkpoint.glob("*.safetensors")):
        stat = p.stat()
        files.append(dict(name=p.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    if not files:
        raise ValueError(f"No local safetensors checkpoint in {checkpoint}")
    config_hash = hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()
    index = checkpoint / "model.safetensors.index.json"
    return dict(
        format_version=1,
        checkpoint=str(checkpoint),
        config_sha256=config_hash,
        index_sha256=(
            hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
        ),
        torch_version=torch.__version__,
        platform=platform.system(),
        architecture=platform.machine(),
        files=files,
    )


def write_manifest(directory, checkpoint, layouts, dtypes=None):
    directory = Path(directory)
    entries = []
    for idx, e, h, inter in layouts:
        for name in ("w13_weight_packed", "w2_weight_packed"):
            path = directory / f"layer-{idx}-{name}.bin"
            stat = path.stat()
            dtype = (dtypes or {}).get((idx, name), torch.uint8)
            element_size = torch.empty(0, dtype=dtype).element_size()
            if stat.st_size % (e * element_size):
                raise ValueError("Malformed expert cache byte count")
            entries.append(
                dict(
                    layer=idx,
                    name=name,
                    shape=[e, stat.st_size // (e * element_size)],
                    dtype=str(dtype),
                    h=h,
                    inter=inter,
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=_file_sha256(path),
                )
            )
    report = dict(
        identity=checkpoint_identity(checkpoint), entries=entries, complete=True
    )
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")
    return report


def read_manifest(directory, checkpoint, layouts):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("complete") is not True or report["identity"] != checkpoint_identity(
        checkpoint
    ):
        raise ValueError(
            "Expert cache belongs to a different checkpoint or Torch layout"
        )
    expected = {
        (idx, name): (e, h, inter)
        for idx, e, h, inter in layouts
        for name in ("w13_weight_packed", "w2_weight_packed")
    }
    entries = {}
    for entry in report["entries"]:
        key = (entry["layer"], entry["name"])
        if key in entries or key not in expected:
            raise ValueError("Unexpected expert cache entry")
        e, h, inter = expected[key]
        if entry["shape"][0] != e or entry["h"] != h or entry["inter"] != inter:
            raise ValueError("Expert cache model dimensions changed")
        dtype_name = entry.get("dtype", "torch.uint8")
        if dtype_name not in ("torch.uint8", "torch.float32"):
            raise ValueError("Unsupported expert cache container dtype")
        element_size = torch.empty(
            0, dtype=getattr(torch, dtype_name.removeprefix("torch."))
        ).element_size()
        if entry["shape"][0] * entry["shape"][1] * element_size != entry["bytes"]:
            raise ValueError("Expert cache shape and byte count disagree")
        file = directory / f"layer-{key[0]}-{key[1]}.bin"
        stat = file.stat()
        if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError("Expert cache file changed; rebuild in a new directory")
        if _file_sha256(file) != entry.get("sha256"):
            raise ValueError("Expert cache SHA256 mismatch; rebuild in a new directory")
        entries[key] = entry
    if set(entries) != set(expected):
        raise ValueError("Incomplete expert cache")
    return entries
