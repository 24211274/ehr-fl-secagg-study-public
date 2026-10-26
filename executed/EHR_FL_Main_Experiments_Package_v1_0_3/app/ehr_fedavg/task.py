"""Local client computation reused by the actual Flower client and offline kernel tests."""
from __future__ import annotations
import csv,json,time,os
from functools import lru_cache
from pathlib import Path
import torch
from pilot_core.model import CompactLSTMAutoencoder,set_training_seed,new_optimiser,session_losses,state_digest
from pilot_core.core import derived_seed
from ehr_preprocessing.dataset import SessionDataset,collate_sessions
from central_smoke.identity import sha256_file,write_json
from central_smoke.engine import checked_losses,gradient_norm,finite_state,check_optimizer
from .math_checks import arrays_from_model,load_arrays,arrays_digest,spec_from_model
from .resources import read_document, resource_sha256
from mai_measurement import client_span


def cfg():return read_document('fixed_config.json')


def setup_torch():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)


def model_factory():
    s=read_document('encoding_schema.json')
    m=CompactLSTMAutoencoder([len(f['categories']) for f in s['categorical_fields']]).to('cpu',dtype=torch.float32)
    if m.input_dim!=119 or sum(p.numel() for p in m.parameters())!=86487:raise ValueError('Model identity mismatch')
    return m


def fresh_global():
    setup_torch();set_training_seed(cfg()['training_seed']);return model_factory()


@lru_cache(maxsize=3)
def load_client(root_str,client):
    if client not in ('C1','C2','C3'):raise ValueError('Invalid client')
    root=Path(root_str).resolve();d=root/'client_data'/client
    runtime=json.loads((root/'runtime.json').read_text())
    if runtime['fixed_config_sha256']!=resource_sha256('fixed_config.json'):raise ValueError('Runtime config mismatch')
    for r in runtime['client_files'][client]:
        target=d/r['path']
        if not target.resolve().is_relative_to(d) or not target.is_file() or sha256_file(target)!=r['sha256']:raise ValueError('Local client data changed')
    with (d/'membership.csv').open(newline='') as f:members=list(csv.DictReader(f))
    with (d/'draws.csv').open(newline='') as f:draws=list(csv.DictReader(f))
    if any(r['client']!=client or r['partition'] not in ['training','validation'] for r in members):raise ValueError('Wrong client membership')
    data={};samples={}
    for part,n in [('training',cfg()['training_sessions_per_client'][client]),('validation',cfg()['validation_sessions_per_client'][client])]:
        ids=sorted(r['session_id'] for r in members if r['partition']==part)
        if len(ids)!=n or len(set(ids))!=n:raise ValueError('Wrong selected local membership')
        ds=SessionDataset(d,client,part,session_ids=ids,verify=False)
        samples[part]=[ds[i] for i in range(len(ds))]
        data[part]={s['session_id']:s for s in samples[part]}
    if set(data['training'])&set(data['validation']):raise ValueError('Train/validation session overlap')
    if any(r['client']!=client or r['session_id'] not in data['training'] for r in draws):raise ValueError('Wrong local draw origin')
    from collections import Counter
    expected_n=cfg()['training_sessions_per_client'][client]
    expected_cells={(co,r) for co in cfg()['conditions'] for r in range(1,cfg()['rounds']+1)}
    if set((r['condition'],int(r['outer_epoch'])) for r in draws)!=expected_cells:raise ValueError('Unexpected/missing full-size draw cell')
    for co,r in sorted(expected_cells):
        part=sorted([d for d in draws if d['condition']==co and int(d['outer_epoch'])==r],key=lambda d:int(d['client_draw_position']))
        if len(part)!=expected_n or [int(d['client_draw_position']) for d in part]!=list(range(expected_n)):raise ValueError('Draw positions/count mismatch')
        if co=='reference' and len({d['session_id'] for d in part})!=expected_n:raise ValueError('Reference not a full permutation')
    return data,samples,draws,runtime


def local_eval(model,samples):
    """Fixed local sessions, no parameter/gradient updates; no threshold or threat metric."""
    before=state_digest(model);oldmode=model.training
    beforegrad=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.eval();rows=[];start=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(samples),64):
            subset=samples[i:i+64];batch=collate_sessions(subset);losses=checked_losses(model,batch,session_losses)
            for s,loss in zip(subset,losses):rows.append({'session_id':s['session_id'],'client':s['client'],'partition':s['partition'],'events':len(s['x']),'loss':float(loss)})
    model.train(oldmode)
    if state_digest(model)!=before or any((p.grad is None)!=(g is None) or (g is not None and not torch.equal(p.grad,g)) for p,g in zip(model.parameters(),beforegrad)):raise ValueError('Evaluation changed state/gradients')
    import math
    return math.fsum(r['loss'] for r in rows)/len(rows),rows,time.perf_counter()-start


def fit_one_client(model,cache,draw_ids,seed):
    if not cache or len(draw_ids)!=len(cache) or any(s not in cache for s in draw_ids):raise ValueError('Expected one full-length draw list from this client training partition')
    if any(cache[s]['partition']!='training' for s in draw_ids):raise ValueError('Nontraining record in fit')
    set_training_seed(seed);model.train();opt=new_optimiser(model);check_optimizer(opt,cfg())
    initial=state_digest(model);history=[];start=time.perf_counter();cpu=time.process_time()
    for i in range((len(draw_ids)+63)//64):
        batch=collate_sessions([cache[s] for s in draw_ids[i*64:(i+1)*64]])
        opt.zero_grad(set_to_none=True);loss=checked_losses(model,batch,session_losses).mean();loss.backward()
        pre=gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
        post=gradient_norm(model)
        if post>1.+1e-6:raise ValueError('Gradient clipping bound')
        opt.step();finite_state(model)
        if {int(v['step'].item()) for v in opt.state.values()}!={i+1}:raise ValueError('Unexpected AdamW state steps')
        history.append({'batch':i+1,'sessions':len(draw_ids[i*64:(i+1)*64]),'events':int(batch['lengths'].sum()),'loss':float(loss.detach()),'gradient_before_clip':pre,'gradient_after_clip':post})
    if state_digest(model)==initial or not any(x['gradient_before_clip']>0 for x in history):raise ValueError('No learning update')
    return history,{'fit_wall_seconds':time.perf_counter()-start,'fit_cpu_seconds':time.process_time()-cpu,'steps':len(history),'draws':len(draw_ids),'unique_draws':len(set(draw_ids)),'fresh_optimizer':True,'seed':seed}


class LocalTask:
    def __init__(self,root,client,condition):
        setup_torch()
        if condition not in ('reference','noniid'):raise ValueError('Invalid condition')
        self.root=Path(root).resolve();self.client=client;self.condition=condition
        with client_span('data_load_with_checks'):
            self.data,self.samples,self.draws,self.runtime=load_client(str(self.root),client)
        self.output=self.root/'client_outputs'/condition/client
        self.output.mkdir(parents=True,exist_ok=True)
        self.model=fresh_global()
        self.spec=spec_from_model(self.model)
    def validate(self,parameters,config):
        if config.get('condition')!=self.condition or config.get('run_token')!=self.runtime['run_token']:raise ValueError('Mismatched run/condition')
        load_arrays(self.model,parameters)
        digest=state_digest(self.model)
        if digest!=config.get('input_digest'):raise ValueError('Stale global input')
        return digest
    def fit(self,parameters,config):
        before=self.validate(parameters,config);r=int(config['server_round'])
        if r not in range(1,cfg()['rounds']+1):raise ValueError('Wrong round')
        record=self.output/f'fit_round_{r:02d}.json'
        if record.exists():raise ValueError('Duplicate local round; no resume/retry')
        rows=sorted([d for d in self.draws if d['condition']==self.condition and int(d['outer_epoch'])==r],key=lambda d:int(d['client_draw_position']))
        ids=[d['session_id'] for d in rows]
        seed=derived_seed(cfg()['training_seed'],'client-fit',self.client,r)
        with client_span('local_fit_with_checks'):
            history,metrics=fit_one_client(self.model,self.data['training'],ids,seed)
        after=state_digest(self.model)
        write_json(record,{'status':'PASS','client':self.client,'condition':self.condition,'round':r,'input_digest':before,'output_digest':after,'batch_history':history,**metrics,'test_data_opened':False})
        returned=arrays_from_model(self.model)
        return returned,len(ids),{'client':self.client,'input_digest':before,'output_digest':after,'steps':metrics['steps'],'draws':len(ids),'fit_seconds':metrics['fit_wall_seconds'],'unique_draws':metrics['unique_draws']}
    def evaluate(self,parameters,config):
        before=self.validate(parameters,config);r=int(config['server_round']);phase=config.get('phase')
        if r not in range(cfg()['rounds']+1) or phase not in ('initial','round','selected'):raise ValueError('Invalid evaluation request')
        record=self.output/f'evaluate_{phase}_{r:02d}.json'
        if record.exists():raise ValueError('Duplicate evaluation')
        with client_span('train_scoring'):
            tr,trrows,t1=local_eval(self.model,self.samples['training'])
        with client_span('validation_scoring'):
            va,varows,t2=local_eval(self.model,self.samples['validation'])
        probe=collate_sessions(self.samples['validation'][:64]);self.model.eval()
        with torch.no_grad():
            x=checked_losses(self.model,probe,session_losses);alt=dict(probe);alt['x']=probe['x'].clone();alt['x'][~probe['padding_mask']]=123.
            pad=float((x-checked_losses(self.model,alt,session_losses)).abs().max())
        if pad>1e-6 or before!=state_digest(self.model):raise ValueError('Padding/state invariance failure')
        m={'client':self.client,'input_digest':before,'train_loss':tr,'validation_loss':va,'train_count':len(self.samples['training']),'validation_count':len(self.samples['validation']),'weights_unchanged':True,'padding_delta':pad,'eval_seconds':t1+t2}
        write_json(record,{'status':'PASS','phase':phase,'round':r,'condition':self.condition,**m,'session_losses':trrows+varows,'gradients_unchanged':True,'test_data_opened':False})
        return va,len(self.samples['validation']),m
