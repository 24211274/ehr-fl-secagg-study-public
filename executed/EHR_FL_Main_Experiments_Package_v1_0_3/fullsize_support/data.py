"""Full-size training-only manifests, replay, local staging and immutable identity."""
from __future__ import annotations
import csv, json, math, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from central_smoke.identity import sha256_file as sha,write_json,verify_manifest
from ehr_preprocessing.io_utils import verify_artefacts
from pilot_core.core import derived_seed
from ehr_fedavg.task import cfg


def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def write_csv(p,rows,fields):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def batch_sizes(n,b=64):
    if type(n)!=int or n<1 or type(b)!=int or b<1:raise ValueError('Positive integer batch geometry required')
    return [min(b,n-i) for i in range(0,n,b)]

def audit_manifests(package, prepared=None):
    """Reconcile every occurrence with original files and (optionally) encoded training modes.

    Department is per-session modal department with lexical tie-breaking, as in
    the original training-only manifest builder. Nothing is fit/refit here.
    """
    package=Path(package);conf=cfg()
    if sha(package/'config/full_membership.csv')!=conf['membership_sha256'] or sha(package/'config/full_client_draws_five_rounds.csv')!=conf['schedule_sha256']:
        raise ValueError('Frozen full-size membership/schedule bytes changed')
    members=read_csv(package/'config/full_membership.csv');draws=read_csv(package/'config/full_client_draws_five_rounds.csv')
    expected={(c,p):conf[k][c] for c in conf['clients'] for p,k in [('training','training_sessions_per_client'),('validation','validation_sessions_per_client')]}
    if Counter((r['client'],r['partition']) for r in members)!=expected or len({r['session_id'] for r in members})!=len(members):raise ValueError('Unexpected, duplicate, foreign or nondevelopment member')
    identities={(r['client'],r['session_id']):r for r in members};sessions={c:{r['session_id'] for r in members if r['client']==c and r['partition']=='training'} for c in conf['clients']}
    original=[]
    for c in conf['clients']:
        for r in read_csv(package/f'reference/fullsize_manifests/{c}_five_round_draws.csv'):
            r['client_draw_position']=r.pop('draw_position');original.append(r)
    key=lambda r:(conf['conditions'].index(r['condition']),int(r['outer_epoch']),r['client'],int(r['client_draw_position']))
    if sorted(original,key=key)!=draws:raise ValueError('Joined schedule differs from original per-client occurrences')
    groups=defaultdict(list)
    for r in draws:
        k=(r['client'],r['session_id'])
        if k not in identities or identities[k]['partition']!='training' or identities[k]['hospital_id']!=r['hospital_id']:raise ValueError('Draw outside original training/client origin')
        groups[(r['condition'],int(r['outer_epoch']),r['client'])].append(r)
    ec={(co,i,c) for co in conf['conditions'] for i in range(1,conf['rounds']+1) for c in conf['clients']}
    if set(groups)!=ec:raise ValueError('Wrong set of round/client/condition schedules')
    dept={};audit=[]
    for c in conf['clients']:
        initial=groups[('reference',1,c)]
        dept[c]={r['session_id']:r['department'] for r in initial}
        if set(dept[c])!=sessions[c]:raise ValueError('Reference schedule does not cover full training membership')
        ids=sorted(sessions[c]);ds=[dept[c][s] for s in ids];dom=np.array([d in conf['dominant_departments'][c] for d in ds],bool)
        if not dom.any() or dom.all():raise ValueError('Missing dominant/residual training department group')
        p=np.where(dom,.6/int(dom.sum()),.4/int((~dom).sum()))
        for co in conf['conditions']:
            for i in range(1,conf['rounds']+1):
                rows=groups[(co,i,c)];n=conf['training_sessions_per_client'][c]
                if len(rows)!=n or [int(r['client_draw_position']) for r in rows]!=list(range(n)):raise ValueError('Wrong count/position sequence')
                if any(dept[c][r['session_id']]!=r['department'] for r in rows):raise ValueError('Per-session department changed')
                seed=derived_seed(conf['selection_seed'],co,c,i)
                if any(int(r['sampling_seed'])!=seed for r in rows):raise ValueError('Wrong sampling seed')
                rng=np.random.default_rng(seed);idx=rng.permutation(n) if co=='reference' else rng.choice(n,n,replace=True,p=p)
                if [ids[int(j)] for j in idx]!=[r['session_id'] for r in rows]:raise ValueError('Exact frozen draw replay failed')
                audit.append({'client':c,'condition':co,'round':i,'draws':n,'distinct_draws':len({r['session_id'] for r in rows}),'batches':len(batch_sizes(n)),'last_batch':batch_sizes(n)[-1]})
    encoded_checked=0
    if prepared is not None:
        prepared=Path(prepared);man=read_csv(prepared/'session_manifest.csv')
        if [{k:r[k] for k in ['session_id','hospital_id','client','partition']} for r in man]!=members:raise ValueError('Prepared arrays membership differs from frozen fullsize members')
        schema=json.loads((prepared/'encoding_schema.json').read_text());field=schema['categorical_fields'][2]
        if field['name']!='department':raise ValueError('Department index changed')
        for c in conf['clients']:
            cats=np.load(prepared/c/'training/categorical_targets.npy',mmap_mode='r',allow_pickle=False)
            valid=np.load(prepared/c/'training/categorical_valid.npy',mmap_mode='r',allow_pickle=False)
            for r in man:
                if r['client']!=c or r['partition']!='training':continue
                a=int(r['event_offset']);b=a+int(r['length'])
                if not valid[a:b,2].all():raise ValueError('Invalid department in training mode audit')
                values=[field['categories'][int(x)] for x in cats[a:b,2]]
                if any(d.startswith('__') for d in values):raise ValueError('Unknown department in training-only schedule audit')
                counts=Counter(values);mode=sorted(counts,key=lambda d:(-counts[d],d))[0]
                if mode!=dept[c][r['session_id']]:raise ValueError('Encoded training department mode disagrees with approved original schedule')
                encoded_checked+=1
    return {'status':'PASS','sessions':len(members),'training_sessions':sum(conf['training_sessions_per_client'].values()),'validation_sessions':sum(conf['validation_sessions_per_client'].values()),'schedule_occurrences':len(draws),'cells':audit,'original_five_round_manifests_exact':True,'all_schedule_occurrences_replayed':True,'encoded_training_department_modes_checked':encoded_checked,'validation_resampled':False,'test_data_opened':False,'scaler_refitted':False}


def verify_prepared(package,pre_run):
    conf=cfg();pre=Path(pre_run);base=pre/'package';data=base/'prepared_trainval'
    status=json.loads((pre/'results/reports/stage_status.json').read_text())
    if status.get('status')!='PREPROCESSING_GATE_PASS_NOT_FULL_PILOT':raise ValueError('Preprocessing gate missing')
    release=verify_manifest(base,base/'PACKAGE_SHA256SUMS.txt',conf['base_release_manifest_sha256'])
    arrays=verify_artefacts(data)
    policy=json.loads((Path(package)/'config/fullsize_policy.json').read_text())
    if sha(data/'session_manifest.csv')!=policy['preprocessing_session_manifest_sha256'] or sha(data/'relative_time_scaler.json')!=conf['scaler_sha256']:raise ValueError('Prepared scaler or membership changed')
    audit=audit_manifests(package,data);audit.update(release_files_checked=len(release),prepared_files_checked=len(arrays))
    return audit


def stage_data(package,pre_run,out):
    package=Path(package);data=Path(pre_run)/'package/prepared_trainval';out=Path(out);conf=cfg()
    members=read_csv(package/'config/full_membership.csv');draws=read_csv(package/'config/full_client_draws_five_rounds.csv');man=read_csv(data/'session_manifest.csv');files={}
    for c in conf['clients']:
        d=out/'client_data'/c;d.mkdir(parents=True,exist_ok=False)
        for part in ['training','validation']:shutil.copytree(data/c/part,d/c/part)
        for name in ['encoding_schema.json','relative_time_scaler.json','preprocessing_summary.json']:shutil.copyfile(data/name,d/name)
        for name,rows in [('membership.csv',members),('draws.csv',draws),('session_manifest.csv',man)]:
            write_csv(d/name,[r for r in rows if r['client']==c],list(rows[0]))
        files[c]=[{'path':str(p.relative_to(d)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(d.rglob('*')) if p.is_file()]
    return files


def verify_staged(out,files):
    for c,rows in files.items():
        for r in rows:
            p=Path(out)/'client_data'/c/r['path']
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Staged client arrays changed')
    return True
