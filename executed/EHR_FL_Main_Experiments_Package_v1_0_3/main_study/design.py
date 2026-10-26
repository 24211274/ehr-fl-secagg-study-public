"""Main conditions and training-only schedules, inherited without outcome-driven tuning."""
from __future__ import annotations
from collections import Counter
import copy, hashlib, itertools, math, os, shutil
from pathlib import Path
import numpy as np
from .common import ROOT, CLIENTS, SEEDS, read, read_csv, csv_write, sha, write, file_record, json_bytes
from pilot_core.core import derived_seed
from central_smoke.sampling import sample_indices, jsd

DRAW_FIELDS = ['condition','outer_epoch','client','client_draw_position','session_id','department','sampling_seed','hospital_id']


def decisions(): return read(ROOT/'protocol/decisions/config/main_protocol_decisions.json')


def plan() -> list[dict]:
    rows = read_csv(ROOT/'protocol/decisions/config/main_run_order.csv')
    expected = {(m,c,s) for m in 'ABC' for c in ('reference','noniid') for s in SEEDS}
    if len(rows)!=30 or {(r['configuration'],r['condition'],int(r['training_seed'])) for r in rows}!=expected:
        raise ValueError('Main order does not cover exact adopted matrix')
    if [int(r['execution_order']) for r in rows]!=list(range(1,31)) or any(int(r['epochs_or_rounds'])!=30 for r in rows):
        raise ValueError('Main order/budget drift')
    return rows


def ordered_blocks():
    return list(dict.fromkeys(r['block_id'] for r in plan()))


def department_rows() -> list[dict]:
    """Read only training manifests that were audited before this release; no raw tests."""
    rows=[]
    for c in CLIENTS:
        x=read_csv(ROOT/f'reference/fullsize_manifests/{c}_five_round_draws.csv')
        y=[r for r in x if r['condition']=='reference' and int(r['outer_epoch'])==1]
        expected=decisions()['data']['training_sessions_per_client'][c]
        if len(y)!=expected or len({r['session_id'] for r in y})!=expected: raise ValueError('Original train modes incomplete')
        rows.extend({k:r[k] for k in ['client','session_id','department','hospital_id']} for r in y)
    return sorted(rows,key=lambda r:(r['client'],r['session_id']))


def make_draws(seed: int, condition: str, rounds: int, departments: list[dict] | None = None) -> list[dict]:
    if type(seed)!=int or condition not in ('reference','noniid') or not 1<=rounds<=30:
        raise ValueError('Invalid schedule identity')
    ds=department_rows() if departments is None else departments
    dominant=decisions()['sampling_inherited_candidate']['dominant_departments']
    out=[]
    for epoch in range(1,rounds+1):
        for c in CLIENTS:
            frame=sorted([r for r in ds if r['client']==c],key=lambda r:r['session_id'])
            if not frame or len({r['session_id'] for r in frame})!=len(frame):raise ValueError('Bad train membership')
            sd=derived_seed(seed,condition,c,epoch)
            idx=sample_indices([r['department'] for r in frame],dominant[c],condition,sd)
            for pos,j in enumerate(idx):
                r=frame[int(j)]
                out.append({'condition':condition,'outer_epoch':epoch,'client':c,
                            'client_draw_position':pos,'session_id':r['session_id'],
                            'department':r['department'],'sampling_seed':sd,'hospital_id':r['hospital_id']})
    return out


def pooled_round(draws: list[dict], seed: int, condition: str, epoch: int) -> list[dict]:
    rows=sorted([r for r in draws if r['condition']==condition and int(r['outer_epoch'])==epoch],
                key=lambda r:(r['client'],int(r['client_draw_position'])))
    if not rows: raise ValueError('Empty pooled epoch')
    for c in CLIENTS:
        part=[r for r in rows if r['client']==c]
        if [int(r['client_draw_position']) for r in part]!=list(range(len(part))):raise ValueError('Duplicate/gap draw position')
    seed=derived_seed(seed,condition,'pooled',epoch)
    return [rows[int(i)] for i in np.random.default_rng(seed).permutation(len(rows))]


def distribution_audit(ds: list[dict], draws: list[dict]) -> dict:
    """Expected JSD plus first-five-round observed gate, as inherited (not all-round retuning)."""
    cats=sorted({r['department'] for r in ds});vectors={'reference':[],'expected':[],'observed_first_five':[]};mass={}
    doms=decisions()['sampling_inherited_candidate']['dominant_departments']
    for c in CLIENTS:
        f=[r for r in ds if r['client']==c];n=len(f);co=Counter(r['department'] for r in f)
        nd=sum(v for k,v in co.items() if k in doms[c])
        if nd in (0,n):raise ValueError('Non-IID group absent')
        vectors['reference'].append([co[x]/n for x in cats])
        vectors['expected'].append([co[x]*(.6/nd if x in doms[c] else .4/(n-nd)) for x in cats])
        ob=[r for r in draws if r['client']==c and r['condition']=='noniid' and int(r['outer_epoch'])<=5]
        if not ob:raise ValueError('No first-five draws for distribution gate')
        counts=Counter(r['department'] for r in ob)
        vectors['observed_first_five'].append([counts[x]/len(ob) for x in cats])
        mass[c]=sum(v for k,v in counts.items() if k in doms[c])/len(ob)
    j={k:float(np.mean([jsd(v[i],v[j]) for i,j in itertools.combinations(range(3),2)])) for k,v in vectors.items()}
    checks={k: j[k]>=.10 and j[k]-j['reference']>=.05 for k in ('expected','observed_first_five')}
    checks['dominant_mass_tolerance']=all(abs(x-.6)<=.05 for x in mass.values())
    if not all(checks.values()):raise ValueError('Predeclared Non-IID gate failed; no alternative mapping/seed search')
    return {'status':'PASS','mean_pairwise_JSD_bits':j,'dominant_masses':mass,'gate_rounds':[1,2,3,4,5],
            'clinical_validation_claim':False,'mapping_changed':False}


def cell_config(seed:int,condition:str,rounds:int,*,engineering:bool=False) -> dict:
    d=read(ROOT/'reference/fullsize_B.json')
    d.update(version='1.0.0-main-runtime',stage='ENGINEERING_PREPARATION' if engineering else 'MAIN_CONTROLLED_COMPARISON',
             training_seed=seed,selection_seed=seed,conditions=[condition],rounds=rounds,
             expected_total_updates=rounds*171,condition_watchdog_seconds=21600,
             membership_sha256=sha(ROOT/'reference/full_membership.csv'),schedule_sha256=None)
    # Explicitly operational limit only: six times five-round budget, not early stopping.
    d['new_engineering_decisions']=['30-round adopted budgets; matched seed/client draws; native code gate before training.',
        'Main fit-time total includes all rounds; round-one diagnostic retained, no exclusion from main total.',
        '3600 seconds per central outer epoch and 21600 seconds per whole cell; transport timeouts unchanged.',
        'No best-seed selection, no test access from training, no baseline/research-question additions.']
    return d


def bind_app(appdir:Path, conf:dict, *, configuration:str):
    """Carry literal config/schema bytes in .py to survive the already audited FAB filter."""
    appdir=Path(appdir);q=appdir/'ehr_fedavg';write(q/'fixed_config.json',conf)
    payload={n:(q/n).read_bytes() for n in ('fixed_config.json','encoding_schema.json')}
    s='"""Generated frozen job configuration; no data, learned weights or secrets."""\n'
    s+='RESOURCE_BYTES = '+repr(payload)+'\nRESOURCE_SHA256 = '+repr({n:hashlib.sha256(b).hexdigest() for n,b in payload.items()})+'\n'
    (q/'_embedded_resources.py').write_text(s)
    cs=read(ROOT/'reference/fullsize_C.json')
    cs.update(version='1.0.0-main-runtime',stage=conf['stage'],rounds=conf['rounds'],conditions=conf['conditions'],
              base_fedavg_config_sha256=hashlib.sha256(payload['fixed_config.json']).hexdigest())
    cs['new_engineering_decisions']=conf['new_engineering_decisions']
    raw=json_bytes(cs);ch=hashlib.sha256(raw).hexdigest()
    (appdir/'ehr_secagg/settings.py').write_text('"""Pinned main C configuration."""\nimport json,hashlib\n_RAW = '+repr(raw)+'\nSHA256 = '+repr(ch)+'''\ndef scfg():
    if hashlib.sha256(_RAW).hexdigest()!=SHA256:raise ValueError('C config changed')
    return json.loads(_RAW)
''')
    return ch


def copy_or_link(source:Path,destination:Path):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    try:os.link(source,destination)
    except OSError:shutil.copyfile(source,destination)
    if sha(source)!=sha(destination):raise ValueError('Staging checksum differs')


def stage_cell(campaign:Path,job:Path,conf:dict,draws_file:Path,expected_initial:str,configuration:str):
    """Only development artefacts. Shared arrays are private copies inside this new campaign."""
    import uuid
    job=Path(job);job.mkdir(parents=True,exist_ok=False)
    base=Path(campaign)/'frozen/development';members=read_csv(ROOT/'reference/full_membership.csv')
    manifest=read_csv(base/'session_manifest.csv');draws=read_csv(draws_file);files={}
    for c in CLIENTS:
        d=job/'client_data'/c;d.mkdir(parents=True)
        for part in ('training','validation'):
            for f in sorted((base/c/part).iterdir()):
                if f.is_file():copy_or_link(f,d/c/part/f.name)
        for n in ('encoding_schema.json','relative_time_scaler.json','preprocessing_summary.json'):copy_or_link(base/n,d/n)
        for n,rows in [('membership.csv',members),('session_manifest.csv',manifest),('draws.csv',draws)]:
            csv_write(d/n,[r for r in rows if r['client']==c],list(rows[0]))
        files[c]=[file_record(f,d) for f in sorted(d.rglob('*')) if f.is_file()]
    appdir=job/'apps'/conf['conditions'][0]
    shutil.copytree(ROOT/'app',appdir,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sec_hash=bind_app(appdir,conf,configuration=configuration)
    runtime={'fixed_config_sha256':sha(appdir/'ehr_fedavg/fixed_config.json'),
             'expected_initial_digest':expected_initial,'run_token':uuid.uuid4().hex,'client_files':files}
    if configuration=='C':runtime.update(security_config_sha256=sec_hash,toy_oracle_passed=True)
    write(job/'runtime.json',runtime,exclusive=True)
    write(job/'job.json',{'configuration':configuration,'condition':conf['conditions'][0],'stage':conf['stage'],
                        'training_seed':conf['training_seed'],'rounds':conf['rounds'],
                        'initial_state_digest':expected_initial,'draws_sha256':sha(draws_file),
                        'staging':'campaign-private arrays hard-linked where supported, never linked to original source',
                        'test_data_allowed':False},exclusive=True)
    # Existing generator sets num-server-rounds=6; bind the explicit runtime value here.
    from ehr_fedavg.host import write_project
    write_project(appdir/'pyproject.toml',job,conf['conditions'][0],conf['ports']['control'])
    text=(appdir/'pyproject.toml').read_text().replace('num-server-rounds = 6',f"num-server-rounds = {conf['rounds']}")
    if configuration=='C':text=text.replace('ehr_fedavg.server_app:app','ehr_secagg.server_app:app').replace('ehr_fedavg.client_app:app','ehr_secagg.client_app:app')
    (appdir/'pyproject.toml').write_text(text)
    import tomllib
    assert tomllib.loads(text)['tool']['flwr']['app']['config']['num-server-rounds']==conf['rounds']
    write(job/'reports/client_staging.json',{'files':files,'logical_only_not_os_isolation':True})
    return job
