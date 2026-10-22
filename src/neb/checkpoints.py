"""Retrieve paper checkpoints into their portable case directories."""
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile


def run(root, *, download=False, only=None):
    manifest = json.loads((root / "checkpoints/index.json").read_text(encoding="utf-8"))
    entries = manifest["checkpoints"]
    if only:
        names = set(only)
        unknown = names - {item["id"] for item in entries}
        if unknown:
            raise ValueError(f"Unknown checkpoint IDs: {sorted(unknown)}")
        entries = [item for item in entries if item["id"] in names]
    files=[]
    for entry in entries:
        files.append(entry)
        files.extend(dict(aux,id=entry['id']+' / multipliers') for aux in entry.get('auxiliary',[]))
    for item in files:
        path = root / item["path"]
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
                subprocess.run(["gh", "release", "download", manifest["release"],
                                "--repo", manifest["repository"], "--pattern", item["asset"],
                                "--dir", temporary], check=True)
                fetched = Path(temporary) / item["asset"]
                if fetched.stat().st_size != item["bytes"]:
                    raise ValueError(f"Incomplete checkpoint: {item['id']}")
                with zipfile.ZipFile(fetched) as bundle:
                    if bundle.testzip() is not None:
                        raise ValueError(f"Unreadable checkpoint: {item['id']}")
                fetched.replace(path)
        if path.exists():
            if path.stat().st_size != item["bytes"] or not zipfile.is_zipfile(path):
                raise ValueError(f"Checkpoint verification failed: {item['id']}")
            print(f"{item['id']}: verified ({item['bytes'] / 1e6:.2f} MB)")
        else:
            print(f"{item['id']}: {item['bytes'] / 1e6:.2f} MB; use --download")
