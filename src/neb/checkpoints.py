"""Retrieve release assets and verify them against the published SHA256."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(root, *, download=False, only=None):
    manifest = json.loads((root / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    entries = manifest["checkpoints"]
    if only:
        names = set(only)
        unknown = names - {item["id"] for item in entries}
        if unknown:
            raise ValueError(f"Unknown checkpoint IDs: {sorted(unknown)}")
        entries = [item for item in entries if item["id"] in names]
    for item in entries:
        path = root / item["path"]
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gh", "release", "download", manifest["release"],
                            "--repo", manifest["repository"], "--pattern", item["asset"],
                            "--dir", str(path.parent)], check=True)
            fetched = path.parent / item["asset"]
            if fetched != path:
                fetched.rename(path)
        if path.exists():
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Checkpoint verification failed: {item['id']}")
            print(f"{item['id']}: verified ({item['bytes'] / 1e6:.2f} MB)")
        else:
            print(f"{item['id']}: {item['bytes'] / 1e6:.2f} MB; use --download")
