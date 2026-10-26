"""Training-only fixed smoke schedules; no model scores are used."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import itertools
import math
from pathlib import Path
import numpy as np

CLIENTS = ('C1', 'C2', 'C3')
CONDITIONS = ('reference', 'noniid')


def derived_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little') % 2**32


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    import io
    s=io.StringIO(newline='')
    w=csv.DictWriter(s, fieldnames=fields, lineterminator='\n');w.writeheader();w.writerows(rows)
    return s.getvalue().encode()


def validate_membership(rows: list[dict], cfg: dict) -> None:
    if len(rows) != 1152 or len({r['session_id'] for r in rows}) != len(rows):
        raise ValueError('Expected 1152 distinct fixed train/validation members')
    counts=Counter((r['client'],r['partition']) for r in rows)
    expected={(c,p):n for c in CLIENTS for p,n in [('training',256),('validation',128)]}
    if dict(counts)!=expected:
        raise ValueError('Wrong client/partition membership, or test partition requested')
    for r in rows:
        key=hashlib.sha256(f"{cfg['selection_seed']}|smoke|{r['client']}|{r['session_id']}".encode()).hexdigest()
        if key!=r['rank_key']:
            raise ValueError('Membership rank is not the original preselected rank')


def validate_departments(rows: list[dict], members: list[dict]) -> None:
    expected={(r['client'],r['session_id']) for r in members if r['partition']=='training'}
    actual={(r['client'],r['session_id']) for r in rows}
    if len(rows)!=len(actual) or actual!=expected or any(not r['department'] for r in rows):
        raise ValueError('Department map must cover exactly the 768 selected training sessions')


def sample_indices(departments: list[str], dominant_names: list[str], condition: str, seed: int) -> np.ndarray:
    if condition not in CONDITIONS or not departments:
        raise ValueError('Unknown condition or empty training subset')
    n=len(departments);rng=np.random.default_rng(seed)
    if condition=='reference':
        return rng.permutation(n)
    dominant=np.array([d in dominant_names for d in departments],dtype=bool)
    if dominant.all() or not dominant.any():
        raise ValueError('Both dominant and residual groups are required')
    probs=np.where(dominant,0.6/int(dominant.sum()),0.4/int((~dominant).sum()))
    return rng.choice(n,n,replace=True,p=probs)


DRAW_FIELDS=['condition','outer_epoch','client','client_draw_position','session_id','department','sampling_seed']
POOL_FIELDS=['condition','outer_epoch','pooled_position','client','client_draw_position','session_id','pool_seed']


def make_schedules(dept_rows: list[dict],cfg: dict) -> tuple[list[dict],list[dict]]:
    draws=[];pooled=[]
    for condition in CONDITIONS:
        for epoch in range(1,cfg['schedule_audit_epochs']+1):
            epoch_rows=[]
            for c in CLIENTS:
                frame=sorted([r for r in dept_rows if r['client']==c],key=lambda r:r['session_id'])
                seed=derived_seed(cfg['selection_seed'],condition,c,epoch)
                idx=sample_indices([r['department'] for r in frame],cfg['dominant_departments'][c],condition,seed)
                for pos,j in enumerate(idx):
                    epoch_rows.append({'condition':condition,'outer_epoch':epoch,'client':c,
                        'client_draw_position':pos,'session_id':frame[int(j)]['session_id'],
                        'department':frame[int(j)]['department'],'sampling_seed':seed})
            draws.extend(epoch_rows)
            if epoch<=cfg['outer_epochs']:
                seed=derived_seed(cfg['selection_seed'],condition,'pooled',epoch)
                for pos,j in enumerate(np.random.default_rng(seed).permutation(len(epoch_rows))):
                    r=epoch_rows[int(j)]
                    pooled.append({'condition':condition,'outer_epoch':epoch,'pooled_position':pos,'client':r['client'],
                        'client_draw_position':r['client_draw_position'],'session_id':r['session_id'],'pool_seed':seed})
    return draws,pooled


def jsd(p,q):
    p,q=np.asarray(p,float),np.asarray(q,float)
    if p.shape!=q.shape or not np.isfinite(p).all() or not np.isfinite(q).all() or (p<0).any() or (q<0).any() or p.sum()<=0 or q.sum()<=0:
        raise ValueError('Invalid distributions')
    p,q=p/p.sum(),q/q.sum();m=(p+q)/2
    return sum(float((x[x>0]*np.log2(x[x>0]/m[x>0])).sum()) for x in (p,q))/2


def sampling_audit(depts,draws,pooled,cfg):
    cats=sorted({r['department'] for r in depts});mapping={(r['client'],r['session_id']):r['department'] for r in depts}
    vectors={'reference':[],'expected_noniid':[],'observed_noniid':[]};checks={};cr={}
    for c in CLIENTS:
        f=[r for r in depts if r['client']==c];ds=[r['department'] for r in f];n=len(f)
        counts=Counter(ds);dom=cfg['dominant_departments'][c];nd=sum(d in dom for d in ds)
        if nd==0 or nd==n:raise ValueError('Empty dominant/residual subset')
        vectors['reference'].append([counts[k]/n for k in cats])
        vectors['expected_noniid'].append([counts[k]*(0.6/nd if k in dom else 0.4/(n-nd)) for k in cats])
        nr=[r for r in draws if r['client']==c and r['condition']=='noniid']
        nc=Counter(r['department'] for r in nr);vectors['observed_noniid'].append([nc[k]/len(nr) for k in cats])
        mass=sum(r['department'] in dom for r in nr)/len(nr)
        checks[c+'_mass_within_0_05']=abs(mass-.6)<=.05
        for condition in CONDITIONS:
            for ep in range(1,cfg['schedule_audit_epochs']+1):
                sub=[r for r in draws if r['client']==c and r['condition']==condition and int(r['outer_epoch'])==ep]
                if len(sub)!=n or [int(r['client_draw_position']) for r in sub]!=list(range(n)):
                    raise ValueError('Invalid draw count or positions')
                if any(mapping.get((c,r['session_id']))!=r['department'] for r in sub):raise ValueError('Foreign draw or wrong department')
                if condition=='reference' and len({r['session_id'] for r in sub})!=n:raise ValueError('Reference is not a permutation')
        cr[c]={'selected_training_sessions':n,'dominant_original_sessions':nd,'target_mass':.6,
               'five_epoch_observed_dominant_mass':mass,'first_three_epochs_unique_noniid_sessions':len({r['session_id'] for r in nr if int(r['outer_epoch'])<=3})}
    mean_js={k:float(np.mean([jsd(v[i],v[j]) for i,j in itertools.combinations(range(3),2)])) for k,v in vectors.items()}
    for key in ['expected_noniid','observed_noniid']:
        checks[key+'_jsd_at_least_0_10']=mean_js[key]>=.1
        checks[key+'_increase_at_least_0_05']=mean_js[key]-mean_js['reference']>=.05
    for co in CONDITIONS:
        for ep in range(1,cfg['outer_epochs']+1):
            source=Counter((r['client'],int(r['client_draw_position']),r['session_id']) for r in draws if r['condition']==co and int(r['outer_epoch'])==ep)
            dest=[r for r in pooled if r['condition']==co and int(r['outer_epoch'])==ep]
            if Counter((r['client'],int(r['client_draw_position']),r['session_id']) for r in dest)!=source or [int(r['pooled_position']) for r in dest]!=list(range(768)):
                raise ValueError('Pooling lost or duplicated a draw occurrence')
    checks['all_draws_within_original_client_and_training_membership']=True
    checks['pooled_occurrence_multisets_preserved']=True
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'clients':cr,'mean_pairwise_JSD_bits':mean_js,
        'scope':'Five-epoch schedule audit on fixed smoke TRAINING subset; only first three epochs trained. Not the full-training distribution and not a clinical realism claim.',
        'draw_rows':len(draws),'pooled_rows':len(pooled),'test_data_used':False,'validation_resampled':False}


def minibatches(items,batch_size=64):
    if batch_size<1:raise ValueError('Positive batch size required')
    for i in range(0,len(items),batch_size):yield items[i:i+batch_size]


def weighted_mean(losses,counts):
    if not losses or len(losses)!=len(counts) or any(n<=0 for n in counts) or any(not math.isfinite(x) for x in losses):
        raise ValueError('Finite losses and positive sample counts required')
    return math.fsum(x*n for x,n in zip(losses,counts))/sum(counts)


def choose_checkpoint(history,tol=1e-12):
    eligible=[r for r in history if int(r['epoch'])>0]
    if not eligible:raise ValueError('No trained checkpoint to select')
    if [int(r['epoch']) for r in eligible]!=sorted({int(r['epoch']) for r in eligible}):raise ValueError('Epoch records not ordered and unique')
    minimum=min(float(r['validation_loss']) for r in eligible)
    if not math.isfinite(minimum) or any(not math.isfinite(float(r['validation_loss'])) for r in eligible):raise ValueError('Nonfinite validation loss')
    return next(int(r['epoch']) for r in eligible if float(r['validation_loss'])<=minimum+tol)
