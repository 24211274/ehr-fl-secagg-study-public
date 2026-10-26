"""Verify/read original JSON bytes transported as a supported Python resource.

Flower 1.22's FAB builder only includes .py, .toml and .md. Original JSON
files are retained in the release for audit, but are not runtime dependencies
of the installed Flower App. This module never searches the current directory,
fills missing values, changes the approved schema, or reads data records.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from ._embedded_resources import RESOURCE_BYTES, RESOURCE_SHA256

NAMES = ('fixed_config.json', 'encoding_schema.json')


def resource_bytes(name: str) -> bytes:
    if name not in NAMES:
        raise KeyError('Unrecognised bundled resource: ' + str(name))
    payload = RESOURCE_BYTES[name]
    if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != RESOURCE_SHA256[name]:
        raise ValueError('Bundled resource digest mismatch: ' + name)
    return payload


def resource_sha256(name: str) -> str:
    return hashlib.sha256(resource_bytes(name)).hexdigest()


def read_document(name: str) -> dict:
    doc = json.loads(resource_bytes(name).decode('utf-8'))
    if not isinstance(doc, dict):
        raise ValueError('Bundled document must be a JSON object: ' + name)
    # Every call returns a fresh object, so caller mutations cannot alter the rules.
    return doc


def verify_source_resources(app_directory: Path) -> dict:
    """Host-only preflight: prove the carried bytes equal the unchanged originals."""
    directory = Path(app_directory) / 'ehr_fedavg'
    records = []
    for name in NAMES:
        raw = (directory / name).read_bytes()
        if raw != resource_bytes(name):
            raise ValueError('Original JSON and bundled representation differ: ' + name)
        records.append({'name': name, 'bytes': len(raw), 'sha256': resource_sha256(name)})
    conf = read_document('fixed_config.json')
    schema = read_document('encoding_schema.json')
    if conf['schema_sha256'] != resource_sha256('encoding_schema.json'):
        raise ValueError('Configuration/schema digest link differs')
    if conf['input_dim'] != 119 or not schema.get('approved'):
        raise ValueError('Only the approved 119-dimensional schema is accepted')
    return {'status': 'PASS', 'resources': records, 'original_bytes_preserved': True}
