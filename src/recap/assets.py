"""Streaming integrity checks for independently distributed assets."""
import hashlib
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes asset root: {relative}')
    return path

def verify(root, entries):
    failures = []
    for entry in entries:
        p = safe_path(root, entry['path'])
        if not p.is_file(): failures.append(f'missing: {entry["path"]}')
        elif p.stat().st_size != entry['bytes']: failures.append(f'size: {entry["path"]}')
        elif sha256(p) != entry['sha256']: failures.append(f'sha256: {entry["path"]}')
    return failures
