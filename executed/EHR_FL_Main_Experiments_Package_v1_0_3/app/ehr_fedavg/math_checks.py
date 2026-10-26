"""Pure checked parameter transport/math. No Flower import or network side effects."""
from __future__ import annotations
import hashlib
import math
from collections import OrderedDict
import numpy as np
import torch


def arrays_from_model(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def spec_from_model(model):
    return [{'name':n,'shape':list(v.shape),'dtype':str(v.detach().cpu().numpy().dtype)} for n,v in model.state_dict().items()]


def validate_arrays(arrays,spec):
    if len(arrays)!=len(spec) or not spec: raise ValueError('Incorrect parameter count')
    for a,s in zip(arrays,spec):
        if not isinstance(a,np.ndarray) or list(a.shape)!=s['shape'] or str(a.dtype)!=s['dtype'] or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid parameter array: '+s['name'])


def load_arrays(model,arrays):
    spec=spec_from_model(model);validate_arrays(arrays,spec)
    model.load_state_dict(OrderedDict((s['name'],torch.from_numpy(a.copy())) for s,a in zip(spec,arrays)),strict=True)


def arrays_digest(arrays,spec):
    validate_arrays(arrays,spec);h=hashlib.sha256()
    for a,s in zip(arrays,spec):
        h.update(s['name'].encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def weighted_reference(updates,weights,spec):
    """Independent float64 weighted mean, not Flower or its aggregation helper."""
    if not updates or len(updates)!=len(weights):raise ValueError('Missing or mismatched updates')
    if any(type(w)!=int or w<=0 for w in weights):raise ValueError('Weights must be positive integer session counts')
    for arrays in updates:validate_arrays(arrays,spec)
    total=sum(weights)
    return [sum((arr[j].astype(np.float64)*(w/total) for arr,w in zip(updates,weights)),start=np.zeros(s['shape'],np.float64)) for j,s in enumerate(spec)]


def check_aggregate(actual,updates,weights,spec,atol,rtol):
    validate_arrays(actual,spec)
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol<0 or rtol<0:raise ValueError('Invalid tolerance')
    expected=weighted_reference(updates,weights,spec);worst=0.;maxratio=0.
    for a,e in zip(actual,expected):
        err=np.abs(a.astype(np.float64)-e);bound=atol+rtol*np.abs(e)
        worst=max(worst,float(err.max()));maxratio=max(maxratio,float(np.max(err/np.maximum(bound,np.finfo(float).tiny))))
        if np.any(err>bound):raise ValueError('FedAvg disagrees with independent weighted reference')
    return {'status':'PASS','max_abs_difference':worst,'max_tolerance_fraction':maxratio,'atol':atol,'rtol':rtol,'weights':weights,'sum_weights':sum(weights),'reference_accumulator':'float64'}


def validate_participants(records,phase,expected_input=None):
    """records: dictionaries with node_id, count, metrics; exactly 3 logical clients."""
    if phase not in ('fit','evaluate'):raise ValueError('Unknown reply phase')
    if len(records)!=3:raise ValueError('Expected exactly three successful replies')
    from .task import cfg
    conf=cfg();labels=[];nodes=[]
    for r in records:
        m=r['metrics'];label=m.get('client')
        if label not in ('C1','C2','C3'):raise ValueError('Unknown logical client')
        labels.append(label);nodes.append(r['node_id'])
        if type(r['count'])!=int or r['count']!=(conf['training_sessions_per_client'][label] if phase=='fit' else conf['validation_sessions_per_client'][label]):raise ValueError('Incorrect aggregation count')
        if expected_input and m.get('input_digest')!=expected_input:raise ValueError('Client used stale or different global model')
        if phase=='fit' and (m.get('steps')!=conf['local_updates_per_round'][label] or m.get('draws')!=conf['training_sessions_per_client'][label]):raise ValueError('Wrong local training budget')
        if phase!='fit':
            for name in ['train_loss','validation_loss']:
                if not isinstance(m.get(name),(int,float)) or not math.isfinite(m[name]) or m[name]<0:raise ValueError('Nonfinite/negative evaluation loss')
            if m.get('train_count')!=conf['training_sessions_per_client'][label] or m.get('validation_count')!=conf['validation_sessions_per_client'][label]:raise ValueError('Wrong fixed evaluation membership count')
            if m.get('weights_unchanged') is not True:raise ValueError('Evaluation mutated model')
    if len(set(nodes))!=3 or sorted(labels)!=['C1','C2','C3']:raise ValueError('Duplicated node/client or missing participant')
    return sorted(records,key=lambda r:r['metrics']['client'])


def summarise_evaluation(records,expected_digest):
    rows=validate_participants(records,'evaluate',expected_digest)
    return {'train_loss':math.fsum(r['metrics']['train_loss']*r['metrics']['train_count'] for r in rows)/sum(r['metrics']['train_count'] for r in rows),
        'validation_loss':math.fsum(r['metrics']['validation_loss']*r['metrics']['validation_count'] for r in rows)/sum(r['metrics']['validation_count'] for r in rows),
        'training_sessions':sum(r['metrics']['train_count'] for r in rows),'validation_sessions':sum(r['metrics']['validation_count'] for r in rows),
        'per_client':{r['metrics']['client']:{k:r['metrics'][k] for k in ['train_loss','validation_loss','train_count','validation_count']} for r in rows}}
