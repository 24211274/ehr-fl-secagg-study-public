"""Strict read-only manifest checks and shareable failure reports."""
from __future__ import annotations
import json,zipfile,re,os
from pathlib import Path,PurePosixPath
from central_smoke.identity import sha256_file as sha,write_json,verify_manifest

def read(p):return json.loads(Path(p).read_text())
def check_report(root,expected=None):
    root=Path(root).resolve();m=root/'report_files.json'
    if expected and sha(m)!=expected:raise ValueError('Predecessor report manifest not the accepted revision')
    rows=read(m)['files'];seen=set()
    if not rows:raise ValueError('Empty report manifest')
    for r in rows:
        rel=PurePosixPath(r['path']);p=root/r['path']
        if rel.is_absolute() or '..' in rel.parts or '\\' in r['path'] or r['path'] in seen or not p.resolve().is_relative_to(root) or p.is_symlink():raise ValueError('Unsafe/duplicate predecessor member')
        seen.add(r['path'])
        if not p.is_file() or sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('Predecessor file changed: '+r['path'])
    extras={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}-seen-{'report_files.json'}
    if extras:raise ValueError('Unlisted predecessor report files')
    return {'status':'PASS','files_checked':len(rows),'manifest_sha256':sha(m)}

def package_report(reports,target):
    reports=Path(reports);target=Path(target)
    files=[{'path':str(p.relative_to(reports)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(reports.rglob('*')) if p.is_file() and p!=reports/'report_files.json']
    write_json(reports/'report_files.json',{'files':files,'excluded':['checkpoints','client arrays','per-session evaluation rows','protocol secrets/shares/masks','Flower stores and full process logs','wheel files and environments']})
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(reports.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(reports)))
    # Verify actual payload bytes after closing archive; no unsafe extract.
    with zipfile.ZipFile(target) as z:
        import hashlib
        if len(z.namelist())!=len(files)+1:raise ValueError('Report archive file mismatch')
        for r in files:
            raw=z.read(r['path'])
            if len(raw)!=r['bytes'] or hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Report archive integrity failure')
    return target

def copy_exclusive(source,target):
    target=Path(target)
    with Path(source).open('rb') as src,target.open('xb') as dst:
        import shutil
        shutil.copyfileobj(src,dst)
    if sha(source)!=sha(target):raise ValueError('Report copy checksum failure')
