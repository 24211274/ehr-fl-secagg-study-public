"""Read-only inventory of the original environment; derive a minimal pinned closure."""
from __future__ import annotations
import hashlib,importlib.metadata as md,json,platform,re,sys
from pathlib import Path
from .build import manifest,sha

def canon(n):return re.sub(r'[-_.]+','-',n).lower()
def snapshot():
    m=manifest()
    if sys.version_info[:2]!=(3,12) or sys.prefix==sys.base_prefix:
        raise ValueError('Run the host installer with the existing project Python 3.12 venv')
    dist=md.distribution('flwr')
    if dist.version!=m['stock_version']:raise ValueError('Original environment is not unmodified Flower 1.22.0')
    helper=Path(dist.locate_file(m['helper_path'])).resolve()
    if not helper.is_relative_to(Path(sys.prefix).resolve()) or sha(helper.read_bytes())!=m['sources']['flower_v1_22_0_secaggplus_utils.py']:
        raise ValueError('Original installed helper differs from reviewed source')
    versions={canon(d.metadata['Name']):d.version for d in md.distributions()}
    source_root=helper.parents[2]
    hashes={str(p.relative_to(source_root)):sha(p.read_bytes()) for p in sorted(source_root.rglob('*')) if p.is_file() and p.suffix not in ('.pyc','.pyo')}
    # pip's vendored packaging avoids installing a new dependency in the baseline.
    from pip._vendor.packaging.requirements import Requirement
    selected={};pending=[('flwr',frozenset())];visited=set()
    while pending:
        name,extras=pending.pop();name=canon(name)
        key=(name,extras)
        if key in visited:continue
        visited.add(key);d=md.distribution(name);selected[name]=d.version
        for line in d.requires or []:
            req=Requirement(line)
            if req.marker and not any(req.marker.evaluate({'extra':extra}) for extra in ({''}|set(extras))):continue
            if req.url:raise ValueError('Direct-URL dependency requires separate review')
            child=md.distribution(req.name)
            if child.version not in req.specifier:raise ValueError('Baseline has incompatible dependency: '+req.name)
            pending.append((req.name,frozenset(req.extras)))
    if selected.get('numpy')!='2.3.5':raise ValueError('Expected reviewed NumPy 2.3.5')
    if any(n in selected for n in ['torch','tensorflow','ray']):raise ValueError('Validation closure unexpectedly includes an ML/simulation backend')
    return {'python':platform.python_version(),'packages':versions,'flower_runtime_hashes':hashes,
            'minimal_dependency_closure':dict(sorted(selected.items())),
            'flower_helper_sha256':sha(helper.read_bytes()),'original_environment_modified':False}
