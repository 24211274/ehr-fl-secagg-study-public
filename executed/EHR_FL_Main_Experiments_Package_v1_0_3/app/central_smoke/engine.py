"""Train A only on fixed routine-development subsets with a separate validation set."""
from __future__ import annotations
from collections import Counter
import csv
import math
from pathlib import Path
import threading
import time
import psutil
import torch
from .identity import write_json,sha256_file
from .sampling import CLIENTS,minibatches,choose_checkpoint,weighted_mean


def checked_losses(model,batch,loss_fn):
    sl,el,mask=loss_fn(model(batch['x'],batch['lengths']),batch['categorical'],batch['binary'],
                       batch['temporal'],batch['lengths'],batch['categorical_valid'])
    if sl.shape!=(len(batch['lengths']),) or not torch.equal(mask,batch['padding_mask']):
        raise ValueError('Loss shape/mask mismatch')
    if not torch.isfinite(sl).all() or not torch.isfinite(el).all() or (sl<0).any():
        raise ValueError('Nonfinite/negative reconstruction loss')
    return sl


def finite_state(model):
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite model state')


def gradient_norm(model):
    total=0.
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite gradient: '+name)
        total+=float(p.grad.detach().to(torch.float64).square().sum())
    value=math.sqrt(total)
    if not math.isfinite(value):raise ValueError('Nonfinite total gradient')
    return value


class RSSMonitor:
    """Approximate process-tree sampled RSS; does not measure GPU or Windows-wide memory."""
    def __init__(self,interval=.05):
        self.interval=interval;self.stop=threading.Event();self.peak=0;self.samples=0;self.error=None
    def sample(self):
        try:
            p=psutil.Process();rss=p.memory_info().rss
            for c in p.children(recursive=True):
                try:rss+=c.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            self.peak=max(self.peak,rss);self.samples+=1
        except Exception as e:self.error=type(e).__name__
    def _loop(self):
        while not self.stop.wait(self.interval):self.sample()
    def __enter__(self):
        self.sample();self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=2);self.sample()


def evaluate(model,samples,partition,collate,loss_fn,state_digest,batch_size=64):
    if partition not in ('training','validation') or not samples:
        raise ValueError('Only nonempty routine training/validation are permitted')
    ids=[s['session_id'] for s in samples]
    if len(ids)!=len(set(ids)) or any(s['partition']!=partition or s['client'] not in CLIENTS for s in samples):
        raise ValueError('Wrong evaluation membership/partition')
    before=state_digest(model);mode=model.training;gradflag=torch.is_grad_enabled()
    before_grads=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    rows=[];start=time.perf_counter();cpu=time.process_time()
    model.eval()
    try:
        with torch.no_grad():
            for subset in minibatches(samples,batch_size):
                batch=collate(subset);sl=checked_losses(model,batch,loss_fn)
                if sl.requires_grad:raise ValueError('Evaluation unexpectedly tracks gradients')
                for sample,n,value in zip(subset,batch['lengths'],sl):
                    rows.append({'session_id':sample['session_id'],'client':sample['client'],
                        'partition':partition,'length':int(n),'reconstruction_loss':float(value)})
    finally:model.train(mode)
    if before!=state_digest(model) or torch.is_grad_enabled()!=gradflag:
        raise ValueError('Evaluation changed model state or ambient gradient mode')
    for p,g in zip(model.parameters(),before_grads):
        if (g is None)!=(p.grad is None) or (g is not None and not torch.equal(g,p.grad)):
            raise ValueError('Evaluation changed training gradients')
    per={}
    for c in CLIENTS:
        r=[r for r in rows if r['client']==c]
        if not r:raise ValueError('Missing evaluation client')
        per[c]={'sessions':len(r),'events':sum(x['length'] for x in r),
                'mean_loss':math.fsum(x['reconstruction_loss'] for x in r)/len(r)}
    mean=math.fsum(r['reconstruction_loss'] for r in rows)/len(rows)
    recomputed=weighted_mean([per[c]['mean_loss'] for c in CLIENTS],[per[c]['sessions'] for c in CLIENTS])
    if not math.isclose(mean,recomputed,abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Wrong session-weighted reduction')
    return {'mean_loss':mean,'per_client':per,'sessions':len(rows),'events':sum(r['length'] for r in rows),
            'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'parameters_unchanged':True,'gradients_unchanged':True},rows


def check_optimizer(opt,cfg):
    g=opt.param_groups[0]
    if len(opt.param_groups)!=1 or opt.state:
        raise ValueError('New epoch must start with one fresh empty-state AdamW')
    for k,v in [('lr',cfg['learning_rate']),('weight_decay',cfg['weight_decay']),('eps',cfg['epsilon']),('foreach',False),('fused',False),('amsgrad',False)]:
        if g.get(k)!=v:raise ValueError('Unexpected optimiser '+k)
    if list(g['betas'])!=cfg['betas']:raise ValueError('Unexpected AdamW betas')


def run_condition(condition,cfg,initial_state,cache,pooled,model_factory,set_seed,
                  loss_fn,new_optimizer,state_digest,collate,reports,checkpoints,progress):
    reports.mkdir(parents=True);checkpoints.mkdir(parents=True)
    # Reset RNG, construct exactly the same architecture, then load the fresh common state.
    set_seed(cfg['training_seed']);model=model_factory()
    model.load_state_dict(initial_state,strict=True)
    initial_hash=state_digest(model);finite_state(model)
    if model.output_dropout.p!=cfg['model_dropout_during_training']:raise ValueError('Dropout policy changed')
    train_samples=sorted([v for (c,p,s),v in cache.items() if p=='training'],key=lambda s:(s['client'],s['session_id']))
    val_samples=sorted([v for (c,p,s),v in cache.items() if p=='validation'],key=lambda s:(s['client'],s['session_id']))
    if len(train_samples)!=768 or len(val_samples)!=384:raise ValueError('Wrong smoke subset size')
    torch.save(model.state_dict(),checkpoints/'initial_state.pt')
    history=[];states={};evaluations={};gradients=[];after_norms=[];steps=0;checkpoints_meta=[];reset_checks=[]
    hfields=['condition','epoch','optimiser_steps_total','train_eval_loss','validation_loss','validation_C1','validation_C2','validation_C3','optimisation_dropout_on_mean','fit_wall_seconds','fit_cpu_seconds','evaluation_wall_seconds','epoch_total_seconds']
    bfields=['condition','epoch','batch_index','global_step','sessions','events','training_loss_dropout_on','gradient_before_clip','gradient_after_clip','optimizer_state_size_before_epoch']
    sfields=['condition','epoch','session_id','client','partition','length','reconstruction_loss']
    start=time.perf_counter();cpu_start=time.process_time()
    with RSSMonitor(cfg['memory_sampling_seconds']) as memory, \
         (reports/'loss_history.csv').open('w',newline='') as hf, \
         (reports/'batch_history.csv').open('w',newline='') as bf, \
         (reports/'routine_session_losses.csv').open('w',newline='') as sf:
        hw=csv.DictWriter(hf,fieldnames=hfields,lineterminator='\n');hw.writeheader()
        bw=csv.DictWriter(bf,fieldnames=bfields,lineterminator='\n');bw.writeheader()
        sw=csv.DictWriter(sf,fieldnames=sfields,lineterminator='\n');sw.writeheader()
        for epoch in range(0,cfg['outer_epochs']+1):
            epoch_start=time.perf_counter();fit_wall=fit_cpu=0.;train_online=None
            if epoch:
                draw=[r for r in pooled if r['condition']==condition and int(r['outer_epoch'])==epoch]
                if len(draw)!=768 or Counter(r['client'] for r in draw)!=dict.fromkeys(CLIENTS,256):raise ValueError('Wrong draw count')
                optimizer=new_optimizer(model);check_optimizer(optimizer,cfg)
                reset_checks.append({'epoch':epoch,'state_size_before':len(optimizer.state)})
                model.train();fit_start=time.perf_counter();fit_cpustart=time.process_time();loss_sum=0;consumed=0
                for batch_index,part in enumerate(minibatches(draw,cfg['batch_sessions']),1):
                    if time.perf_counter()-epoch_start>cfg['outer_epoch_watchdog_seconds']:raise TimeoutError('Epoch watchdog exceeded')
                    samples=[cache[(r['client'],'training',r['session_id'])] for r in part]
                    if any(s['partition']!='training' for s in samples):raise ValueError('Nontraining sample in fit')
                    batch=collate(samples);optimizer.zero_grad(set_to_none=True)
                    loss=checked_losses(model,batch,loss_fn).mean();value=float(loss.detach());loss.backward()
                    before=gradient_norm(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_global_l2'],error_if_nonfinite=True,foreach=False)
                    after=gradient_norm(model)
                    if after>cfg['gradient_clip_global_l2']+cfg['post_clip_tolerance']:raise ValueError('Clip bound exceeded')
                    optimizer.step();steps+=1;progress['optimiser_steps_completed']+=1;finite_state(model)
                    gradients.append(before);after_norms.append(after);consumed+=len(part);loss_sum+=value*len(part)
                    bw.writerow({'condition':condition,'epoch':epoch,'batch_index':batch_index,'global_step':steps,
                        'sessions':len(part),'events':int(batch['lengths'].sum()),'training_loss_dropout_on':value,
                        'gradient_before_clip':before,'gradient_after_clip':after,'optimizer_state_size_before_epoch':0});bf.flush()
                    step_counts={int(v['step'].item()) for v in optimizer.state.values()}
                    if step_counts!={batch_index}:raise ValueError('AdamW state did not reset/count correctly')
                if consumed!=768 or steps!=epoch*12:raise ValueError('Wrong epoch optimisation count')
                reset_checks[-1]['per_parameter_steps_after']=12
                fit_wall=time.perf_counter()-fit_start;fit_cpu=time.process_time()-fit_cpustart;train_online=loss_sum/consumed
            tr,trrows=evaluate(model,train_samples,'training',collate,loss_fn,state_digest,cfg['batch_sessions'])
            va,varows=evaluate(model,val_samples,'validation',collate,loss_fn,state_digest,cfg['batch_sessions'])
            h={'condition':condition,'epoch':epoch,'optimiser_steps_total':steps,'train_eval_loss':tr['mean_loss'],
               'validation_loss':va['mean_loss'],**{'validation_'+c:va['per_client'][c]['mean_loss'] for c in CLIENTS},
               'optimisation_dropout_on_mean':train_online,'fit_wall_seconds':fit_wall,'fit_cpu_seconds':fit_cpu,
               'evaluation_wall_seconds':tr['wall_seconds']+va['wall_seconds'],'epoch_total_seconds':time.perf_counter()-epoch_start}
            history.append(h);hw.writerow(h);hf.flush()
            for r in trrows+varows:sw.writerow({'condition':condition,'epoch':epoch,**r})
            sf.flush();states[epoch]=state_digest(model);evaluations[epoch]={'training':tr,'validation':va}
            if epoch:
                path=checkpoints/f'epoch_{epoch:02d}.pt';torch.save(model.state_dict(),path)
                checkpoints_meta.append({'epoch':epoch,'file':path.name,'sha256':sha256_file(path),'bytes':path.stat().st_size,'state_digest':states[epoch]})
            print(f"[{condition}] Epoch {epoch}/3 | train eval loss = {tr['mean_loss']:.8f} | routine validation loss = {va['mean_loss']:.8f} | updates = {steps}",flush=True)
            write_json(reports/'progress.json',{'status':'RUNNING','condition':condition,'epoch':epoch,'updates':steps})
            if time.perf_counter()-epoch_start>cfg['outer_epoch_watchdog_seconds']:raise TimeoutError('Epoch watchdog including evaluation exceeded')
        if not any(x>0 for x in gradients):raise ValueError('All gradients were zero')
        final_hash=state_digest(model)
        final_path=checkpoints/'final_state.pt';torch.save(model.state_dict(),final_path)
        best=choose_checkpoint(history,cfg['checkpoint_tie_tolerance']);selected=checkpoints/f'epoch_{best:02d}.pt'
        loaded=torch.load(selected,map_location='cpu',weights_only=True);model.load_state_dict(loaded,strict=True)
        reval,rrows=evaluate(model,val_samples,'validation',collate,loss_fn,state_digest,cfg['batch_sessions'])
        # Per-session reload equality, not merely a matching mean.
        with (reports/'routine_session_losses.csv').open(newline='') as f:
            expected={r['session_id']:float(r['reconstruction_loss']) for r in csv.DictReader(f) if int(r['epoch'])==best and r['partition']=='validation'}
        reload_equal=state_digest(model)==states[best] and all(expected[r['session_id']]==r['reconstruction_loss'] for r in rrows)
        selected_path=checkpoints/'selected_state.pt';torch.save(model.state_dict(),selected_path)
        probe=collate(val_samples[:64]);model.eval()
        with torch.no_grad():
            scores=checked_losses(model,probe,loss_fn);altered=dict(probe);altered['x']=probe['x'].clone();altered['x'][~probe['padding_mask']]=123.
            delta=float((checked_losses(model,altered,loss_fn)-scores).abs().max())
        padding_ok=delta<=1e-6
        no_test=all(k[1] in ('training','validation') for k in cache)
        gates={'exact_three_epochs_and_36_updates':steps==36,'fresh_optimizer_each_epoch':len(reset_checks)==3 and all(x['state_size_before']==0 and x['per_parameter_steps_after']==12 for x in reset_checks),
             'weights_changed_each_epoch':all(states[e]!=states[e-1] for e in range(1,4)),
             'all_numeric_checks_finite':True,'validation_no_updates':all(x['validation']['parameters_unchanged'] and x['validation']['gradients_unchanged'] for x in evaluations.values()),
             'validation_exact_384_each_evaluation':all(x['validation']['sessions']==384 for x in evaluations.values()),
             'selected_checkpoint_reloads_exactly':reload_equal,'trained_padding_invariance':padding_ok,
             'no_test_access':no_test,'gradient_clip_limit':max(after_norms)<=1.+cfg['post_clip_tolerance']}
        if memory.error:gates['resource_sampler']=False
        else:gates['resource_sampler']=True
        result={'status':'PASS' if all(gates.values()) else 'FAIL','condition':condition,'configuration':'A','gates':gates,
            'training_seed':cfg['training_seed'],'input_dim':119,'optimiser_steps':steps,'training_unique_available':768,
            'training_occurrences_consumed':2304,'training_unique_actually_consumed':len({r['session_id'] for r in pooled if r['condition']==condition}),
            'routine_validation_unique':384,'initial_train_eval_loss':history[0]['train_eval_loss'],'final_train_eval_loss':history[-1]['train_eval_loss'],
            'initial_validation_loss':history[0]['validation_loss'],'final_validation_loss':history[-1]['validation_loss'],
            'selected_epoch':best,'selected_validation_loss':reval['mean_loss'],'selected_validation_per_client':reval['per_client'],
            'initial_state_digest':initial_hash,'final_state_digest':final_hash,'selected_state_digest':states[best],
            'epoch_state_digests':states,'optimizer_resets':reset_checks,'gradient_max_before_clip':max(gradients),
            'gradient_max_after_clip':max(after_norms),'padding_check_max_abs_difference':delta,
            'elapsed_condition_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu_start,
            'sampled_peak_process_tree_rss_MiB':memory.peak/1024**2,'rss_samples':memory.samples,
            'fit_wall_seconds_total':math.fsum(x['fit_wall_seconds'] for x in history),
            'evaluation_wall_seconds_total':math.fsum(x['evaluation_wall_seconds'] for x in history),
            'checkpoints':checkpoints_meta+[{'file':p.name,'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in [checkpoints/'initial_state.pt',final_path,selected_path]],
            'performance_pass_target':None,'loss_outcome_interpretation':'Only routine train/validation reconstruction; not threat accuracy/AP, generalisation to real hospitals or SecAgg overhead.',
            'warnings':[] if history[-1]['validation_loss']<history[0]['validation_loss'] else ['Validation loss did not improve; report the outcome without automatic retuning.'],
            'test_metrics_computed':False,'calibration_performed':False,'flower_executed':False,'secagg_protocol_executed':False,'main_experiments_frozen':False}
    write_json(reports/'condition_result.json',result)
    write_json(reports/'progress.json',{'status':result['status'],'epoch':3,'updates':steps})
    return result
