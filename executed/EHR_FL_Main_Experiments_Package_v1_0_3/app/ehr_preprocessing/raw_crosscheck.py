"""Independent raw-to-array spot check using pandas datetime operations.

Select eight development sessions per client/partition from the frozen pilot list.
Test feature values are neither converted nor compared. Does not refit the scaler.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .pipeline import CLIENTS, ALLOWED_PARTITIONS, approved_schema
from .io_utils import BenchmarkSource
from .dataset import SessionDataset


def crosscheck(source_path: Path, package: Path, prepared: Path) -> dict:
    schema=approved_schema(prepared/'encoding_schema.json')
    scaler=json.loads((prepared/'relative_time_scaler.json').read_text())
    with (prepared/'pilot_membership/pilot_membership.csv').open(encoding='utf-8',newline='') as f:
        all_members=list(csv.DictReader(f))
    selected=[]
    for c in CLIENTS:
        for p in ALLOWED_PARTITIONS:
            selected.extend([r for r in all_members if r['client']==c and r['partition']==p][:8])
    ids={r['session_id'] for r in selected}
    names=[f['name'] for f in schema['categorical_fields']]
    cols=names+schema['binary_fields']+['session_id','sequence_position','timestamp_utc']
    with BenchmarkSource(source_path,package/'manifests/source_file_manifest.csv') as src:
        src.verify()
        with src.text('events.csv') as f:
            frames=[]
            for chunk in pd.read_csv(f,usecols=cols,dtype=str,keep_default_na=False,chunksize=10000):
                frames.append(chunk.loc[chunk.session_id.isin(ids)].copy())
            df=pd.concat(frames,ignore_index=True)
    checked_events=0;max_error=0.
    for c in CLIENTS:
        for p in ALLOWED_PARTITIONS:
            sid_list=[r['session_id'] for r in selected if r['client']==c and r['partition']==p]
            ds=SessionDataset(prepared,c,p,sid_list,verify=False)
            for j,sid in enumerate(sid_list):
                rows=df.loc[df.session_id==sid].copy()
                rows['position']=rows.sequence_position.astype(int)
                rows=rows.sort_values('position')
                ts=pd.to_datetime(rows.timestamp_utc,utc=True,errors='raise')
                local=ts.dt.tz_convert('Asia/Kuala_Lumpur')
                hour=(local.dt.hour+local.dt.minute/60+local.dt.second/3600+local.dt.microsecond/3_600_000_000).to_numpy(dtype=np.float64)
                day=local.dt.dayofweek.to_numpy(dtype=np.float64)
                gap=ts.diff().dt.total_seconds().fillna(0).to_numpy(dtype=np.float64)
                elapsed=(ts-ts.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
                time=np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                                      np.sin(2*np.pi*day/7),np.cos(2*np.pi*day/7),
                                      np.log1p(gap),np.log1p(elapsed)])
                time[:,4:]=(time[:,4:]-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])
                expected=[]
                for f in schema['categorical_fields']:
                    k=len(f['categories']);block=np.zeros((len(rows),k),dtype=np.float32)
                    for z,raw in enumerate(rows[f['name']]):
                        ix=f['missing_index'] if raw=='' else (f['categories'].index(raw) if raw in f['categories'] else f['unknown_index'])
                        block[z,ix]=1
                    expected.append(block)
                binary=np.column_stack([rows[name].map({'True':1.0,'False':0.0}).to_numpy(dtype=np.float32) for name in schema['binary_fields']])
                if not np.isfinite(binary).all():
                    raise ValueError('Unexpected Boolean literal in independent cross-check')
                x=np.concatenate([*expected,binary,time.astype(np.float32)],axis=1)
                item=ds[j]
                actual=item['x'].numpy()
                if not np.array_equal(x[:,:113],actual[:,:113]) or not np.allclose(x[:,113:],actual[:,113:],atol=1e-6,rtol=1e-6):
                    raise ValueError('Independent raw-to-array cross-check failed')
                max_error=max(max_error,float(np.max(np.abs(x-actual))))
                checked_events+=len(rows)
    return {'status':'PASS','method':'independent pandas raw-to-array comparison; no scaler refitting',
            'sessions_checked':len(selected),'events_checked':checked_events,'max_abs_feature_error':max_error,
            'temporal_tolerance':{'atol':1e-6,'rtol':1e-6},'categorical_binary_comparison':'exact',
            'test_sessions_selected':0,'schema_modified':False,'scaler_modified':False}
