"""Strict artefact I/O and explicit preparation/training/test boundaries.

These are reproducibility workflow guards on a trusted local machine, not access
control against an administrator, signatures, or remote attestation.
"""
from __future__ import annotations
import csv, hashlib, json, os, shutil
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ('C1', 'C2', 'C3')
SEEDS = (11, 29, 47, 71, 97)
TESTS = ('test_held_out', 'test_known')


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def write(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json_bytes(value)
    if exclusive:
        with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path: Path, rows, fields: list[str] | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows: raise ValueError('Empty CSV needs explicit columns')
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def safe_child(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative); root = Path(root).resolve()
    if p.is_absolute() or '..' in p.parts or '\\' in relative or not p.parts:
        raise ValueError('Unsafe artefact path')
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Symlink/escape not accepted for sealed artefacts')
    return target


def file_record(path: Path, root: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink(): raise ValueError('Expected regular artefact')
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def inventory(root: Path) -> list[dict]:
    return [file_record(p, root) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')]


def verify_rows(root: Path, rows: list[dict]) -> None:
    seen = set()
    if not rows: raise ValueError('Empty sealed manifest')
    for r in rows:
        if r['path'] in seen: raise ValueError('Duplicate manifest member')
        seen.add(r['path']); p = safe_child(root, r['path'])
        if not p.is_file() or p.stat().st_size != r['bytes'] or sha(p) != r['sha256']:
            raise ValueError('Sealed artefact changed: ' + r['path'])


def package_integrity() -> str:
    from central_smoke.identity import verify_manifest
    rows=verify_manifest(ROOT, ROOT / 'PACKAGE_SHA256SUMS.txt')
    expected={x['path'] for x in rows}|{'PACKAGE_SHA256SUMS.txt'}
    # Generated reports and bytecode are not executable release additions.
    extras=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')
            and not (p.parent==ROOT and 'Report_' in p.name and p.suffix=='.zip')
            and p.relative_to(ROOT).as_posix() not in expected]
    if extras: raise ValueError('Unlisted release artefacts: '+repr(extras[:5]))
    return sha(ROOT / 'PACKAGE_SHA256SUMS.txt')


def verify_campaign(campaign: Path) -> dict:
    campaign = Path(campaign).resolve(); m = read(campaign / 'campaign_manifest.json')
    if m['package_manifest_sha256'] != package_integrity(): raise ValueError('Different runnable source release')
    if m['decision_sha256'] != sha(ROOT / 'protocol/decisions/config/main_protocol_decisions.json'):
        raise ValueError('Different adopted decisions')
    verify_rows(campaign, m['files'])
    return m


def require_authorisation(path: Path, action: str, campaign: Path, *, expected_models_sha: str | None = None) -> dict:
    """A reviewed run-specific receipt is needed; example receipts are deliberately false."""
    v = read(path)
    if v.get('approved') is not True or v.get('action') != action:
        raise PermissionError('No explicit approval for ' + action)
    if v.get('campaign_manifest_sha256') != sha(Path(campaign) / 'campaign_manifest.json'):
        raise PermissionError('Approval does not identify this campaign')
    if v.get('package_manifest_sha256') != package_integrity():
        raise PermissionError('Approval does not identify this source release')
    if action == 'TRAIN' and v.get('test_data_authorised') is not False:
        raise PermissionError('Training authorisation must not authorise tests')
    if action == 'EVALUATE_FINAL_TESTS':
        if expected_models_sha is None or v.get('selected_models_manifest_sha256') != expected_models_sha:
            raise PermissionError('Test release must lock trained models and calibration first')
    return v
