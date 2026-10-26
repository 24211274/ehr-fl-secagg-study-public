"""Replay the preselected timing workload without fitting any preprocessing."""
from __future__ import annotations
import csv,hashlib,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from central_smoke.identity import sha256_file as sha,verify_manifest
from central_smoke.sampling import read_csv,csv_bytes,DRAW_FIELDS,validate_membership,validate_departments,derived_seed
from ehr_preprocessing.io_utils import verify_artefacts
from ehr_fedavg.task import cfg


def verify_timing_inputs(package,pre_run=None):
    root=Path(package);conf=cfg()
    member=root/'config/timing_membership.csv';schedule=root/'config/timing_draws_six_rounds.csv'
    if sha(member)!=conf['membership_sha256'] or sha(schedule)!=conf['schedule_sha256']:
        raise ValueError('Timing input checksum mismatch')
    if member.read_bytes()!=(root/'reference/pilot_membership.csv').read_bytes():
        raise ValueError('Original preselected smoke membership changed')
    members=read_csv(member);dept=read_csv(root/'reference/smoke_training_departments.csv')
    validate_membership(members,conf);validate_departments(dept,members)
    expected=[]
    for r in range(1,7):
        for c in conf['clients']:
            frame=sorted([d for d in dept if d['client']==c],key=lambda x:x['session_id'])
            seed=derived_seed(conf['selection_seed'],'reference',c,r)
            for i,j in enumerate(np.random.default_rng(seed).permutation(256)):
                d=frame[int(j)]
                expected.append({'condition':'reference','outer_epoch':r,'client':c,'client_draw_position':i,
                                 'session_id':d['session_id'],'department':d['department'],'sampling_seed':seed})
    if csv_bytes(expected,DRAW_FIELDS)!=schedule.read_bytes():raise ValueError('Timing schedule replay mismatch')
    proof={'status':'PASS','training_sessions':768,'validation_sessions':384,'clients':3,
           'rounds':6,'reference_draw_occurrences_per_run':4608,'same_schedule_each_repetition':True,
           'membership_matches_original_smoke_bytes':True,'all_six_permutations_reproduced':True,
           'scaler_refitted':False,'test_data_opened':False,'prepared_arrays_verified':False}
    if pre_run is not None:
        pre=Path(pre_run);base=pre/'package';data=base/'prepared_trainval'
        if json.loads((pre/'results/reports/stage_status.json').read_text()).get('status')!='PREPROCESSING_GATE_PASS_NOT_FULL_PILOT':
            raise ValueError('Preprocessing gate required')
        release=verify_manifest(base,base/'PACKAGE_SHA256SUMS.txt',conf['base_release_manifest_sha256'])
        artefacts=verify_artefacts(data)
        policy=json.loads((root/'config/timing_policy.json').read_text())
        if sha(data/'session_manifest.csv')!=policy['preprocessing_session_manifest_sha256'] or sha(data/'relative_time_scaler.json')!=conf['scaler_sha256']:
            raise ValueError('Prepared membership/scaler differs')
        manifest=read_csv(data/'session_manifest.csv');lookup={r['session_id']:r for r in manifest}
        for x in members:
            y=lookup.get(x['session_id'])
            if y is None or any(x[k]!=y[k] for k in ('client','partition','hospital_id')):
                raise ValueError('Subset membership not in original assigned training/validation')
        # Prove the original ranked selection, not simply its count/hash.
        for c in conf['clients']:
            for p,n in [('training',256),('validation',128)]:
                available=[r for r in manifest if r['client']==c and r['partition']==p]
                ranked=sorted(available,key=lambda x:hashlib.sha256(f"{conf['selection_seed']}|smoke|{c}|{x['session_id']}".encode()).hexdigest())
                if {r['session_id'] for r in ranked[:n]}!={r['session_id'] for r in members if r['client']==c and r['partition']==p}:
                    raise ValueError('Preselected ranked subset differs')
        proof.update(prepared_arrays_verified=True,prepared_files_checked=len(artefacts),release_files_checked=len(release))
    return proof


def stage_timing_data(package,pre_run,out):
    root=Path(package);data=Path(pre_run)/'package/prepared_trainval';out=Path(out)
    members=read_csv(root/'config/timing_membership.csv');draws=read_csv(root/'config/timing_draws_six_rounds.csv')
    manifest=read_csv(data/'session_manifest.csv');files={}
    def write(p,rows):p.write_bytes(csv_bytes(rows,list(rows[0])))
    for c in cfg()['clients']:
        d=out/'client_data'/c;d.mkdir(parents=True,exist_ok=False)
        # Copy only this client's previously encoded routine training/validation.
        # Loader selects the fixed timing subset; copying does not re-fit features.
        for part in ['training','validation']:shutil.copytree(data/c/part,d/c/part)
        for n in ['encoding_schema.json','relative_time_scaler.json','preprocessing_summary.json']:
            shutil.copyfile(data/n,d/n)
        for n,rows in [('membership.csv',members),('draws.csv',draws),('session_manifest.csv',manifest)]:
            write(d/n,[r for r in rows if r['client']==c])
        files[c]=[{'path':str(p.relative_to(d)),'sha256':sha(p),'bytes':p.stat().st_size}
                  for p in sorted(d.rglob('*')) if p.is_file()]
    return files
