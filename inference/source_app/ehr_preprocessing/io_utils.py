"""Read-only source access, deterministic metadata and artefact hashing."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Iterator
import zipfile


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def id_set_hash(ids) -> str:
    return hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode('utf-8')).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


class BenchmarkSource:
    """Read the original eleven CSVs, either in one directory or one archive.

    ZIPs are streamed, never extracted. Only original CSV names are supported.
    A previous approved source manifest, not the input archive's metadata, is used
    to check identity. No source CSV is ever opened for writing.
    """
    def __init__(self, path: Path, manifest: Path):
        self.path = Path(path).expanduser().resolve()
        with Path(manifest).open(encoding='utf-8-sig', newline='') as f:
            self.manifest = list(csv.DictReader(f))
        self.expected = {Path(r['path']).name: r for r in self.manifest}
        if len(self.expected) != 11:
            raise ValueError('Expected the approved eleven-file source manifest')
        self.archive = None
        self.members = {}
        if self.path.is_dir():
            choices = [self.path / 'data', self.path]
            valid = [d for d in choices if all((d / n).is_file() for n in self.expected)]
            if len(valid) != 1:
                raise ValueError('Source directory must contain the eleven CSVs, or a data/ folder containing them')
            self.data_dir = valid[0]
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                self.close()
                raise ValueError('ZIP contains duplicate member names')
            prefixes = []
            for name in names:
                p = PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or '\\' in name:
                    self.close()
                    raise ValueError('Unsafe ZIP path')
                if p.name == 'events.csv':
                    prefix = name[:-len('events.csv')]
                    if all(prefix + n in names for n in self.expected):
                        prefixes.append(prefix)
            if len(prefixes) != 1:
                self.close()
                raise ValueError('Archive must contain one unambiguous set of eleven source CSVs')
            self.members = {n: prefixes[0] + n for n in self.expected}
            total = sum(self.archive.getinfo(n).file_size for n in self.members.values())
            if total > 500 * 1024 * 1024:
                self.close()
                raise ValueError('Source size exceeds the approved snapshot safety limit')
        else:
            raise FileNotFoundError('Expected an existing benchmark release directory or ZIP')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.archive is not None:
            self.archive.close()

    @contextlib.contextmanager
    def binary(self, filename: str) -> Iterator:
        if filename not in self.expected:
            raise ValueError('Unrecognised source filename')
        f = self.archive.open(self.members[filename]) if self.archive else (self.data_dir / filename).open('rb')
        try:
            yield f
        finally:
            f.close()

    @contextlib.contextmanager
    def text(self, filename: str) -> Iterator:
        with self.binary(filename) as f:
            wrapper = io.TextIOWrapper(f, encoding='utf-8-sig', newline='')
            try:
                yield wrapper
            finally:
                wrapper.detach()

    def verify(self) -> list[dict]:
        verified = []
        for r in self.manifest:
            name = Path(r['path']).name
            h, size = hashlib.sha256(), 0
            with self.binary(name) as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    size += len(b)
                    h.update(b)
            if size != int(r['bytes']) or h.hexdigest() != r['sha256']:
                raise ValueError(f'STOP: source identity mismatch for {name}')
            verified.append({'file': name, 'bytes': size, 'sha256': h.hexdigest(), 'status': 'PASS'})
        return verified


def artefact_manifest(root: Path) -> list[dict]:
    excluded = {'artefact_manifest.csv', 'SHA256SUMS.txt'}
    return [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name not in excluded]


def seal_artefacts(root: Path) -> list[dict]:
    rows = artefact_manifest(root)
    write_csv(root / 'artefact_manifest.csv', rows, ['path', 'bytes', 'sha256'])
    (root / 'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in rows), encoding='utf-8')
    return rows


def verify_artefacts(root: Path) -> list[dict]:
    root = Path(root).resolve()
    with (root / 'artefact_manifest.csv').open(encoding='utf-8', newline='') as f:
        expected = list(csv.DictReader(f))
    if not expected or len({r['path'] for r in expected}) != len(expected):
        raise ValueError('Empty or duplicated artefact manifest')
    for r in expected:
        p = (root / r['path']).resolve()
        if not p.is_relative_to(root):
            raise ValueError('Unsafe manifest path')
        if not p.is_file() or p.stat().st_size != int(r['bytes']) or sha256_file(p) != r['sha256']:
            raise ValueError(f"STOP: artefact mismatch {r['path']}")
    actual = {r['path'] for r in artefact_manifest(root)}
    if actual != {r['path'] for r in expected}:
        raise ValueError('Unexpected or absent artefact files')
    return expected
