"""FedAvg remains Flower's implementation; overrides add fail-closed engineering checks."""
from __future__ import annotations
import json,math,time
from pathlib import Path
import torch
from mai_measurement import server_span
from flwr.server.strategy import FedAvg
from flwr.common import FitIns,EvaluateIns,Code,ndarrays_to_parameters,parameters_to_ndarrays
from central_smoke.identity import write_json,sha256_file
from .math_checks import (arrays_digest,arrays_from_model,spec_from_model,validate_arrays,
    check_aggregate,validate_participants,summarise_evaluation,load_arrays)
from .task import cfg,fresh_global

class CheckedFedAvg(FedAvg):
    def __init__(self,root,condition,initial_arrays,run_token):
        self.root=Path(root);self.condition=condition;self.token=run_token;self.cfg=cfg()
        self.folder=self.root/'server_outputs'/condition;self.folder.mkdir(parents=True,exist_ok=True)
        self.checkpoints=self.root/'checkpoints'/condition;self.checkpoints.mkdir(parents=True,exist_ok=True)
        self.model=fresh_global();self.spec=spec_from_model(self.model)
        self.params=ndarrays_to_parameters(initial_arrays)
        self.digest=arrays_digest(initial_arrays,self.spec);self.initial_digest=self.digest
        load_arrays(self.model,initial_arrays)
        self.history=[];self.fit_checks=[];self.participants=None;self.round_start={};self.last_round=0
        torch.save(self.model.state_dict(),self.checkpoints/'initial_state.pt')
        super().__init__(fraction_fit=1.,fraction_evaluate=1.,min_fit_clients=3,min_evaluate_clients=3,
            min_available_clients=3,accept_failures=False,inplace=False,initial_parameters=self.params,
            fit_metrics_aggregation_fn=lambda _: {},evaluate_metrics_aggregation_fn=lambda _: {})
    def instruction(self,r,phase=None):
        d={'server_round':r,'condition':self.condition,'run_token':self.token,'input_digest':self.digest}
        if phase:d['phase']=phase
        return d
    def checked_clients(self,cm):
        if not cm.wait_for(3,timeout=self.cfg['node_wait_seconds']):raise TimeoutError('Three clients did not register')
        if cm.num_available()!=3:raise ValueError('Expected exactly three registered clients')
    def configure_fit(self,server_round,parameters,client_manager):
        if server_round!=self.last_round+1 or server_round not in range(1,self.cfg['rounds']+1):raise ValueError('Unexpected fit round')
        self.checked_clients(client_manager)
        if arrays_digest(parameters_to_ndarrays(parameters),self.spec)!=self.digest:raise ValueError('Workflow global model differs')
        self.round_start[server_round]=time.perf_counter()
        pairs=super().configure_fit(server_round,parameters,client_manager)
        return [(p,FitIns(parameters,self.instruction(server_round))) for p,_ in pairs]
    def aggregate_fit(self,server_round,results,failures):
        if server_round not in range(1,self.cfg['rounds']+1) or server_round!=self.last_round+1 or server_round not in self.round_start:raise ValueError('Unconfigured/repeated fit round')
        if failures or len(results)!=3:raise RuntimeError('Incomplete/failed fit round; no stale-model continuation')
        items=[]
        for p,res in results:
            if res.status.code!=Code.OK:raise RuntimeError('Non-OK client fit status')
            a=parameters_to_ndarrays(res.parameters);validate_arrays(a,self.spec)
            if arrays_digest(a,self.spec)!=res.metrics.get('output_digest'):raise ValueError('Returned parameter digest mismatch')
            items.append({'node_id':p.node_id,'count':res.num_examples,'metrics':res.metrics,'proxy':p,'res':res,'arrays':a})
        ordered=validate_participants(items,'fit',self.digest);self.check_nodes(ordered)
        before=time.perf_counter()
        # Standard Flower average, not a replacement implementation.
        with server_span('fedavg_native_aggregate',server_round):
            params,_=super().aggregate_fit(server_round,[(r['proxy'],r['res']) for r in ordered],[])
        if params is None:raise RuntimeError('Flower did not return an aggregate')
        aggregate=parameters_to_ndarrays(params)
        with server_span('independent_float_average_audit',server_round):
            audit=check_aggregate(aggregate,[r['arrays'] for r in ordered],[r['count'] for r in ordered],self.spec,self.cfg['aggregation_atol'],self.cfg['aggregation_rtol'])
        new_digest=arrays_digest(aggregate,self.spec)
        if new_digest==self.digest:raise ValueError('Global state did not change')
        audit.update(round=server_round,condition=self.condition,input_digest=self.digest,output_digest=new_digest,
            client_labels=[r['metrics']['client'] for r in ordered],client_steps=[r['metrics']['steps'] for r in ordered],
            aggregation_and_check_seconds=time.perf_counter()-before,
            fit_rpc_seconds=time.perf_counter()-self.round_start[server_round],
            theoretical_parameter_bytes=sum(a.nbytes for a in aggregate),
            upload_parameter_tensor_payload_bytes=sum(sum(len(t) for t in r['res'].parameters.tensors) for r in ordered),
            output_parameter_tensor_payload_bytes=sum(len(t) for t in params.tensors),
            wire_bytes_measured=False,local_training_updates=sum(self.cfg['local_updates_per_round'].values()))
        self.fit_checks.append(audit);self.params=params;self.digest=new_digest;self.last_round=server_round
        load_arrays(self.model,aggregate);torch.save(self.model.state_dict(),self.checkpoints/f'round_{server_round:02d}.pt')
        write_json(self.folder/f'aggregate_round_{server_round:02d}.json',audit)
        return params,{'participants':3,'aggregate_max_abs_difference':audit['max_abs_difference']}
    def check_nodes(self,rows):
        mapping={r['metrics']['client']:int(r['node_id']) for r in rows}
        if self.participants is None:self.participants=mapping
        elif mapping!=self.participants:raise ValueError('Node/client mapping changed during run')
    def configure_evaluate(self,server_round,parameters,client_manager):
        self.checked_clients(client_manager)
        if server_round!=self.last_round or arrays_digest(parameters_to_ndarrays(parameters),self.spec)!=self.digest:raise ValueError('Invalid global evaluation model')
        return [(p,EvaluateIns(parameters,self.instruction(server_round,'round'))) for p in client_manager.all().values()]
    def consume_evaluation(self,r,records,phase):
        rows=validate_participants(records,'evaluate',self.digest);self.check_nodes(rows)
        values=summarise_evaluation(rows,self.digest)
        for x in rows:
            if not math.isclose(x['loss'],x['metrics']['validation_loss'],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Client evaluation loss mismatch')
        result={'round':r,'phase':phase,'model_digest':self.digest,**values}
        if phase!='selected':
            if any(x['round']==r for x in self.history):raise ValueError('Duplicate evaluation round')
            self.history.append(result)
            write_json(self.folder/'history.json',{'history':self.history})
            print(f"[B/{self.condition}] Round {r}/{self.cfg['rounds']} | train eval loss = {values['train_loss']:.8f} | routine validation loss = {values['validation_loss']:.8f} | clients = 3/3",flush=True)
        write_json(self.folder/f'evaluation_{phase}_{r:02d}.json',result)
        return result
    def aggregate_evaluate(self,server_round,results,failures):
        if failures or len(results)!=3:raise RuntimeError('Incomplete/failed evaluation round')
        records=[]
        for p,res in results:
            if res.status.code!=Code.OK:raise RuntimeError('Non-OK evaluation status')
            records.append({'node_id':p.node_id,'count':res.num_examples,'metrics':res.metrics,'loss':res.loss})
        val=self.consume_evaluation(server_round,records,'round')
        return val['validation_loss'],{'train_loss':val['train_loss'],'clients':3}
