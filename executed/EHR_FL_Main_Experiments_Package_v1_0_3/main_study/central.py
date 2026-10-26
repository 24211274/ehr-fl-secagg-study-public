"""Full-size pooled A using unchanged model/loss/AdamW; fresh optimiser each outer epoch."""
from __future__ import annotations
import math,time
from pathlib import Path
import torch
from .common import write,read,CLIENTS,sha,csv_write
from .design import pooled_round
from ehr_fedavg.task import cfg,fresh_global,load_client
from pilot_core.model import state_digest,session_losses,new_optimiser
from central_smoke.engine import checked_losses,finite_state,gradient_norm,check_optimizer,evaluate,RSSMonitor
from central_smoke.sampling import choose_checkpoint
from ehr_preprocessing.dataset import collate_sessions


def fit_epoch(model,cache,ordered,conf,*,deadline=None):
    if len(ordered)!=len(cache) or any((r['client'],r['session_id']) not in cache for r in ordered):raise ValueError('Pooled draw membership/count')
    if any(cache[(r['client'],r['session_id'])]['partition']!='training' for r in ordered):raise ValueError('Nontraining draw')
    opt=new_optimiser(model);check_optimizer(opt,conf);model.train();history=[]
    start=time.perf_counter_ns();cpu=time.process_time_ns()
    for i in range(0,len(ordered),conf['batch_sessions']):
        if deadline is not None and time.perf_counter()>deadline:raise TimeoutError('A outer-epoch watchdog')
        part=ordered[i:i+conf['batch_sessions']]
        batch=collate_sessions([cache[(r['client'],r['session_id'])] for r in part])
        opt.zero_grad(set_to_none=True);loss=checked_losses(model,batch,session_losses).mean();loss.backward()
        g=gradient_norm(model);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
        after=gradient_norm(model)
        if after>1.+1e-6:raise ValueError('A gradient bound')
        opt.step();finite_state(model)
        steps=len(history)+1
        if {int(v['step'].item()) for v in opt.state.values()}!={steps}:raise ValueError('A optimiser reset/step count')
        history.append({'batch':steps,'sessions':len(part),'loss':float(loss.detach()),'gradient_before':g,'gradient_after':after})
    return history,{'fit_wall_ns':time.perf_counter_ns()-start,'fit_cpu_ns':time.process_time_ns()-cpu,'fresh_optimizer':True}


def run_central(job:Path):
    job=Path(job);conf=cfg();condition=read(job/'job.json')['condition'];R=conf['rounds']
    results=job/'central_results';results.mkdir(exist_ok=False);cp=job/'checkpoints'/condition;cp.mkdir(parents=True,exist_ok=False)
    train=[];val=[];draws=[]
    for c in CLIENTS:
        _,samples,d,_=load_client(str(job),c);train.extend(samples['training']);val.extend(samples['validation']);draws.extend(d)
    train.sort(key=lambda s:(s['client'],s['session_id']));val.sort(key=lambda s:(s['client'],s['session_id']))
    if len(train)!=sum(conf['training_sessions_per_client'].values()) or len(val)!=sum(conf['validation_sessions_per_client'].values()):raise ValueError('A full count')
    cache={(s['client'],s['session_id']):s for s in train}
    model=fresh_global();initial=state_digest(model)
    if initial!=read(job/'runtime.json')['expected_initial_digest']:raise ValueError('A initialisation mismatch')
    torch.save(model.state_dict(),cp/'initial_state.pt');history=[];steps=0;start=time.perf_counter();before=initial
    with RSSMonitor(.1) as rss:
        for epoch in range(R+1):
            span={};batches=[];epoch_start=time.perf_counter()
            if epoch:
                order=pooled_round(draws,conf['selection_seed'],condition,epoch)
                batches,span=fit_epoch(model,cache,order,conf,deadline=time.perf_counter()+3600)
                steps+=len(batches)
                if len(batches)!=math.ceil(len(train)/64) or sum(r['sessions'] for r in batches)!=len(train):raise ValueError('A partial batch lost')
                write(results/f'batches_{epoch:02d}.json',{'batches':batches,**span})
            tr,_=evaluate(model,train,'training',collate_sessions,session_losses,state_digest)
            vr,vrows=evaluate(model,val,'validation',collate_sessions,session_losses,state_digest)
            dig=state_digest(model)
            if epoch and dig==before:raise ValueError('A model did not change')
            h={'epoch':epoch,'train_loss':tr['mean_loss'],'validation_loss':vr['mean_loss'],
               'per_client':vr['per_client'],'state_digest':dig,'steps_total':steps,
               'evaluation_wall_seconds':tr['wall_seconds']+vr['wall_seconds'],**span}
            history.append(h);write(results/'history.json',{'history':history})
            if epoch:torch.save(model.state_dict(),cp/f'epoch_{epoch:02d}.pt')
            before=dig
            print(f"[A/{condition}] Epoch {epoch}/{R} | routine validation loss = {vr['mean_loss']:.8f} | updates = {steps}",flush=True)
            if time.perf_counter()-epoch_start>3600 or time.perf_counter()-start>conf['condition_watchdog_seconds']:raise TimeoutError('A operational watchdog')
        best=choose_checkpoint(history,conf['checkpoint_tie_tolerance'])
        model.load_state_dict(torch.load(cp/f'epoch_{best:02d}.pt',map_location='cpu',weights_only=True),strict=True)
        vr,rows=evaluate(model,val,'validation',collate_sessions,session_losses,state_digest)
        if state_digest(model)!=history[best]['state_digest'] or abs(vr['mean_loss']-history[best]['validation_loss'])>1e-7:raise ValueError('A checkpoint reload failed')
        # Per-client scoring batches match B/C during calibration (pooled batch boundaries can differ in float32).
        from ehr_fedavg.task import local_eval
        calibration_rows=[]
        for c in CLIENTS:
            _,r,_=local_eval(model,[s for s in val if s['client']==c]);calibration_rows.extend(r)
        write(job/'LOCAL_ONLY/selected_validation_scores.json',{'rows':calibration_rows})
        torch.save(model.state_dict(),cp/'selected_state.pt')
    if rss.error:raise ValueError('A resource sampler failed')
    result={'status':'PASS','configuration':'A','condition':condition,'rounds_completed':R,
            'training_seed':conf['training_seed'],'initial_state_digest':initial,'selected_round':best,
            'selected_state_digest':state_digest(model),'final_validation_loss':history[-1]['validation_loss'],
            'selected_validation_loss':vr['mean_loss'],'local_updates':steps,
            'fit_wall_ns_total':sum(h.get('fit_wall_ns',0) for h in history),
            'evaluation_wall_seconds_total':sum(h['evaluation_wall_seconds'] for h in history),
            'elapsed_seconds':time.perf_counter()-start,'sampled_process_tree_peak_rss_MiB':rss.peak/1024**2,
            'test_data_opened':False,'optimizer_reset_each_epoch':True,'partial_last_batch_sessions':len(train)%64 or 64,
            'selected_checkpoint_sha256':sha(cp/'selected_state.pt')}
    write(results/'result.json',result)
    write(job/'reports/central_summary.json',result)
    write(job/'reports/central_history.json',{'history':history})
    return result
