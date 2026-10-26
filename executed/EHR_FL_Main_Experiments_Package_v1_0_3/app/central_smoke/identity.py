"""Fail-closed identity, membership and acceptance checks (no network access)."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def verify_manifest(root: Path, manifest: Path, expected_digest: str | None = None) -> list[dict]:
    root = root.resolve()
    if expected_digest and sha256_file(manifest) != expected_digest:
        raise ValueError('Identity manifest differs from the pinned release')
    seen, rows = set(), []
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, rel = line.split('  ', 1)
        pp = PurePosixPath(rel)
        if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Invalid checksum')
        if pp.is_absolute() or '..' in pp.parts or '\\' in rel or rel in seen:
            raise ValueError('Unsafe or repeated identity path')
        seen.add(rel)
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f'Identity mismatch: {rel}')
        rows.append({'path': rel, 'sha256': digest})
    if not rows:
        raise ValueError('Empty checksum manifest')
    return rows

