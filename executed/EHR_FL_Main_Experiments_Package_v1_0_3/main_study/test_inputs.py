"""Final-test-only encoding, callable only after an external run-specific test receipt.

Shares approved feature functions with development preprocessing; never fits a
vocabulary, scaler, threshold or model. No test file is opened at import time.
"""
from __future__ import annotations
import csv,re
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import torch
from .common import CLIENTS, TESTS, ROOT, read, sha
from ehr_preprocessing.pipeline import (Session, approved_schema, encode_category, parse_boolean,
       parse_timestamp, temporal_values, event_features, CLIENTS as HOSPITALS, FORBIDDEN_TENSOR_COLUMNS)


def encode_rows(sid,client,hid,part,rows,schema,scaler):
    if part not in TESTS or not rows:raise ValueError('Explicit nonempty final-test session required')
    names=[f['name'] for f in schema['categorical_fields']]
    if set(names+schema['binary_fields'])&set(FORBIDDEN_TENSOR_COLUMNS):raise ValueError('Forbidden model feature')
    parsed=[]
    for r in rows:
        if r['session_id']!=sid or r['hospital_id']!=hid:raise ValueError('Session/client event mismatch')
        if not re.fullmatch(r'0|[1-9][0-9]*',r['sequence_position']):raise ValueError('Invalid event order')
        parsed.append((int(r['sequence_position']),parse_timestamp(r['timestamp_utc']),r))
    parsed.sort(key=lambda x:(x[0],x[1],x[2]['event_id']))
    if [p[0] for p in parsed]!=list(range(len(parsed))):raise ValueError('Duplicate/gapped event positions')
    cats=np.empty((len(rows),13),dtype='<i8');valid=np.empty_like(cats,dtype=bool);binary=np.empty((len(rows),3),dtype='<f4')
    for i,(_,_,r) in enumerate(parsed):
        for j,f in enumerate(schema['categorical_fields']):
            mapping={x:k for k,x in enumerate(f['categories'])}
            cats[i,j],valid[i,j]=encode_category(r[f['name']],f,mapping)
        for j,name in enumerate(schema['binary_fields']):binary[i,j]=parse_boolean(r[name])
    us,temporal=temporal_values([p[1] for p in parsed],schema['preprocessing_policy_reference']['calendar_timezone'])
    s=Session(sid,client,hid,part,[p[2]['event_id'] for p in parsed],us,cats,valid,binary,temporal)
    x=event_features(s,schema,scaler)
    # Keys mirror SessionDataset without invoking its development-only partition gate.
    return {'session_id':sid,'client':client,'hospital_id':hid,'partition':part,
            'x':torch.from_numpy(x),'categorical':torch.from_numpy(cats),'categorical_valid':torch.from_numpy(valid),
            'binary':torch.from_numpy(x[:,110:113].copy()),'temporal':torch.from_numpy(x[:,113:119].copy())}


def load_final_inputs(source, campaign:Path):
    """Called after authorisation and selection manifest verification in evaluate_final.py."""
    base=Path(campaign)/'frozen/development'
    schema=approved_schema(base/'encoding_schema.json');scaler=read(base/'relative_time_scaler.json')
    expected=read(ROOT/'protocol/decisions/config/main_protocol_decisions.json')['representation']['scaler_sha256']
    if sha(base/'relative_time_scaler.json')!=expected:raise ValueError('Fixed scaler differs')
    hospital_to_client={v:k for k,v in HOSPITALS.items()};selected={};all_ids=set();labels={p:[] for p in TESTS}
    with source.text('labels.csv') as f:
        for r in csv.DictReader(f):
            sid=r['session_id'];part=r['partition']
            if not sid or sid in all_ids or part not in ('training','validation',*TESTS):raise ValueError('Invalid/duplicate partition label')
            all_ids.add(sid)
            if part not in TESTS:continue
            hid=r['hospital_id']
            if hid not in hospital_to_client:raise ValueError('Unknown client')
            y=int(parse_boolean(r['binary_label']));c=hospital_to_client[hid]
            selected[sid]=(c,hid,part)
            labels[part].append({'session_id':sid,'client':c,'partition':part,'label':y,
                                 'scenario':r.get('scenario_id',''),'primary_class':r.get('primary_class','')})
    groups={s:[] for s in selected};seen=set()
    keep=['session_id','hospital_id','event_id','sequence_position','timestamp_utc']+[x['name'] for x in schema['categorical_fields']]+schema['binary_fields']
    with source.text('events.csv') as f:
        for r in csv.DictReader(f):
            sid=r['session_id']
            if sid not in all_ids:raise ValueError('Event has no session label')
            if sid not in selected:continue
            if not r['event_id'] or r['event_id'] in seen:raise ValueError('Duplicate/empty test event id')
            seen.add(r['event_id']);groups[sid].append({k:r[k] for k in keep})
    samples={p:[] for p in TESTS};coverage={}
    for sid in sorted(selected):
        c,hid,p=selected[sid];s=encode_rows(sid,c,hid,p,groups[sid],schema,scaler);samples[p].append(s)
    for p in TESTS:
        if not samples[p] or {s['client'] for s in samples[p]}!=set(CLIENTS):raise ValueError('Empty/missing-client final test')
        coverage[p]={}
        for c in CLIENTS:
            ss=[s for s in samples[p] if s['client']==c];v=np.concatenate([s['categorical_valid'] for s in ss])
            x=np.concatenate([s['categorical'] for s in ss])
            coverage[p][c]={'sessions':len(ss),'events':len(v),'excluded_categorical_targets':int((~v).sum()),
                'events_without_any_valid_categorical_target':int((~v.any(axis=1)).sum()),
                'by_field':{f['name']:{'unknown':int((x[:,j]==f['unknown_index']).sum()),
                    'missing':int((x[:,j]==f['missing_index']).sum())} for j,f in enumerate(schema['categorical_fields'])}}
    return samples,labels,coverage
