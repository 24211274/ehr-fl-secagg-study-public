"""Deterministic wheel-level backport. Never write into an installed package.

The public PyPI wheel is SHA-256 pinned. Only the mask helper is changed under
flwr/; version/provenance and RECORD are updated under .dist-info. No build hooks
or unrelated upstream AGENTS.md changes are executed.
"""
from __future__ import annotations
import ast,base64,csv,hashlib,io,json,re,stat,zipfile
from email.parser import BytesParser
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def manifest()->dict:return json.loads((ROOT/'CHANGE_MANIFEST.json').read_text())
def function_source(data:bytes)->str:
    text=data.decode('utf-8')
    matches=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='pseudo_rand_gen']
    if len(matches)!=1:raise ValueError('Exactly one named PRG function is required')
    return ast.get_source_segment(text,matches[0])+'\n'

def check_sources()->dict:
    m=manifest()
    for name,digest in m['sources'].items():
        if sha((ROOT/'sources'/name).read_bytes())!=digest:raise ValueError('Pinned source mismatch: '+name)
    old=(ROOT/'sources/flower_v1_22_0_secaggplus_utils.py').read_bytes()
    pr=(ROOT/'sources/UNMERGED_PR_7570_e87804ad_secaggplus_utils.py').read_bytes()
    new=(ROOT/'sources/backport_secaggplus_utils.py').read_bytes()
    expected=old.decode().replace(function_source(old),function_source(pr)).replace('import numpy as np','import hashlib\nimport hmac\nimport struct\n\nimport numpy as np',1).encode()
    notice='# MODIFIED FOR PROJECT EVALUATION (not an official Flower release).\n# Backport of PR #7570, commit e87804ad00c46bff6dc3f058b73de81529e50bc0.\n# See accompanying NOTICE.md and wheel provenance for attribution and limitations.\n'
    expected=expected.replace(b'import hashlib\n',(notice+'\nimport hashlib\n').encode(),1)
    if expected!=new:raise ValueError('Backport differs from the minimal approved transformation')
    if function_source(pr)!=function_source(new):raise ValueError('PR function changed in backport')
    if sha(function_source(new).encode())!=m['new_function_sha256']:raise ValueError('Function hash mismatch')
    return m

def read_wheel(path:Path,expected_sha:str)->dict[str,bytes]:
    raw=path.read_bytes()
    if sha(raw)!=expected_sha:raise ValueError('Input wheel SHA-256 differs from pinned value')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries={}
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if (info.filename in entries or p.is_absolute() or '..' in p.parts or '\\' in info.filename
                or stat.S_ISLNK(info.external_attr>>16)):
                raise ValueError('Unsafe/duplicate wheel member')
            if info.is_dir():continue
            if info.file_size>40_000_000:raise ValueError('Unexpected member size')
            entries[info.filename]=z.read(info)
    validate_record(entries)
    return entries

def validate_record(entries:dict[str,bytes])->None:
    records=[n for n in entries if n.endswith('.dist-info/RECORD')]
    if len(records)!=1:raise ValueError('Expected one RECORD')
    rec=records[0];rows=list(csv.reader(io.StringIO(entries[rec].decode())))
    seen=set()
    for row in rows:
        if len(row)!=3:raise ValueError('Malformed RECORD')
        name,hash_field,size=row
        if name in seen or name not in entries:raise ValueError('Duplicate/missing RECORD member')
        seen.add(name)
        if name==rec:
            if hash_field or size:raise ValueError('RECORD must not hash itself')
            continue
        want='sha256='+base64.urlsafe_b64encode(hashlib.sha256(entries[name]).digest()).decode().rstrip('=')
        if hash_field!=want or size!=str(len(entries[name])):raise ValueError('RECORD integrity mismatch: '+name)
    if seen!=set(entries):raise ValueError('Unlisted wheel member')

def make_record(entries:dict[str,bytes],rec:str)->bytes:
    buf=io.StringIO(newline='');writer=csv.writer(buf,lineterminator='\n')
    for n in sorted(set(entries)-{rec}):
        writer.writerow([n,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(entries[n]).digest()).decode().rstrip('='),len(entries[n])])
    writer.writerow([rec,'','']);return buf.getvalue().encode()

def write_wheel(entries:dict[str,bytes],dest:Path)->None:
    if dest.exists():raise FileExistsError('Refusing to overwrite existing wheel')
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(entries):
            i=zipfile.ZipInfo(name,(2025,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o100644<<16
            z.writestr(i,entries[name])

def transform_entries(entries:dict[str,bytes],m:dict)->tuple[dict[str,bytes],dict]:
    # Kept separate to permit fixture-wheel testing without accepting fixtures in production.
    hp=m['helper_path'];old=(ROOT/'sources/flower_v1_22_0_secaggplus_utils.py').read_bytes()
    if entries.get(hp)!=old:raise ValueError('Stock wheel helper is not the reviewed source')
    old_info='flwr-1.22.0.dist-info';new_info=f"flwr-{m['candidate_version']}.dist-info"
    if f'{old_info}/METADATA' not in entries:raise ValueError('Unexpected stock dist-info')
    metadata=entries[f'{old_info}/METADATA'];parsed=BytesParser().parsebytes(metadata)
    if parsed['Name'].lower()!='flwr' or parsed['Version']!='1.22.0':raise ValueError('Wrong stock wheel identity')
    w=entries.get(f'{old_info}/WHEEL',b'')
    if b'Tag: py3-none-any' not in w or b'Root-Is-Purelib: true' not in w:raise ValueError('Unexpected stock wheel tags')
    if any(n.endswith(('/RECORD.jws','/RECORD.p7s')) for n in entries):raise ValueError('Signed wheel needs separate review')
    out={n.replace(old_info+'/',new_info+'/',1) if n.startswith(old_info+'/') else n:b for n,b in entries.items()}
    out[hp]=(ROOT/'sources/backport_secaggplus_utils.py').read_bytes()
    changed_meta,count=re.subn(rb'(?m)^Version: 1\.22\.0\r?$',('Version: '+m['candidate_version']).encode(),metadata)
    if count!=1:raise ValueError('Ambiguous version metadata')
    out[f'{new_info}/METADATA']=changed_meta
    provenance={'local_build':True,'official_flower_release':False,'pr_merged':False,'base_version':'1.22.0',
                'candidate_version':m['candidate_version'],'upstream_pr':7570,'commit':m['candidate_commit'],
                'helper_sha256':sha(out[hp]),'function_sha256':m['new_function_sha256'],
                'purpose':'supervisor-scope-approved evaluation only','privacy_certified':False}
    out[f'{new_info}/MAI_PRG_PROVENANCE.json']=(json.dumps(provenance,sort_keys=True,indent=2)+'\n').encode()
    rec=f'{new_info}/RECORD';out[rec]=make_record(out,rec);validate_record(out)
    changed_flwr=[n for n in entries if n.startswith('flwr/') and out.get(n)!=entries[n]]
    if changed_flwr!=[hp]:raise ValueError('Unexpected Flower runtime file change')
    if {n for n in out if n.startswith('flwr/')}!={n for n in entries if n.startswith('flwr/')}:
        raise ValueError('Unexpected runtime file addition/removal')
    return out,{'changed_flower_runtime_files':changed_flwr,'unchanged_flower_runtime_files':sum(n.startswith('flwr/') for n in entries)-1,
                'dependency_metadata_unchanged':BytesParser().parsebytes(out[f'{new_info}/METADATA']).get_all('Requires-Dist')==parsed.get_all('Requires-Dist'),
                'stock_wheel_sha256':m['official_wheel_sha256'],'provenance':provenance}

def build(stock:Path,out_dir:Path)->tuple[Path,dict]:
    m=check_sources();entries=read_wheel(stock,m['official_wheel_sha256']);out,report=transform_entries(entries,m)
    path=out_dir/f"flwr-{m['candidate_version']}-py3-none-any.whl"
    write_wheel(out,path);read_wheel(path,sha(path.read_bytes()))
    report.update(candidate_wheel=path.name,candidate_wheel_sha256=sha(path.read_bytes()),status='PASS')
    return path,report
