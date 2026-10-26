"""Same prespecified validation sessions as accepted inference pilot; model+score time only."""
from __future__ import annotations
from pathlib import Path
import time,math
import numpy as np
import torch
from .common import ROOT,CLIENTS,read_csv,write
from ehr_fedavg.task import fresh_global,load_client
from ehr_preprocessing.dataset import collate_sessions
from central_smoke.engine import checked_losses
from pilot_core.model import state_digest,session_losses


def measure(job:Path,checkpoint:Path):
    selection=read_csv(ROOT/'reference/main_inference_selection.csv')
    samples={}
    for c in CLIENTS:
        _,s,_,_=load_client(str(job),c)
        samples.update({r['session_id']:r for r in s['validation']})
    # Field names are validated by the preparation tests; selection is inherited byte-for-byte.
    phases=set(r['phase'] for r in selection)
    if phases!={'warmup','measured'}:raise ValueError('Inference phase schema drift')
    ordered={p:[samples[r['session_id']] for r in selection if r['phase']==p] for p in phases}
    if len(ordered['warmup'])!=20 or len(ordered['measured'])!=200:raise ValueError('Inference selection changed')
    m=fresh_global();m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);m.eval();before=state_digest(m)
    timings=[];scores=[]
    with torch.no_grad():
        for p in ('warmup','measured'):
            for sample in ordered[p]:
                batch=collate_sessions([sample]);st=time.perf_counter_ns()
                loss=checked_losses(m,batch,session_losses);elapsed=time.perf_counter_ns()-st
                if p=='measured':timings.append(elapsed);scores.append(float(loss[0]))
        batch_times=[];batch_scores=[]
        for i in range(0,200,64):
            batch=collate_sessions(ordered['measured'][i:i+64]);st=time.perf_counter_ns()
            loss=checked_losses(m,batch,session_losses);batch_times.append(time.perf_counter_ns()-st);batch_scores.extend(map(float,loss))
    if state_digest(m)!=before or any(p.grad is not None for p in m.parameters()):raise ValueError('Inference changed weights/gradients')
    if not np.allclose(scores,batch_scores,atol=1e-6,rtol=1e-6):raise ValueError('Batch/single inference scores disagree')
    result={'measured_sessions':200,'warmup_sessions':20,'mean_ms':float(np.mean(timings)/1e6),
            'median_ms':float(np.median(timings)/1e6),'p95_ms':float(np.quantile(timings,.95,method='linear')/1e6),
            'throughput_sessions_per_second':200*1e9/sum(batch_times),'batch_sizes':[64,64,64,8],
            'weights_unchanged':True,'secagg_executed_during_inference':False,
            'scope':'prepared-input model and score computation only; preprocessing/I/O/transport excluded',
            'not_repeated_latency_experiment':True}
    write(job/'reports/selected_model_inference.json',result)
    return result
