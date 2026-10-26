"""B: real Flower DefaultWorkflow on a bounded Grid; no raw/encoded client data loaded here."""
from __future__ import annotations
import json,time,traceback
from pathlib import Path
import torch
from flwr.common import Context,Message,EvaluateIns,Code,ndarrays_to_parameters,parameters_to_ndarrays
import flwr.common.recorddict_compat as compat
from flwr.server import ServerApp,Grid,LegacyContext,ServerConfig
from flwr.server.workflow import DefaultWorkflow
from flwr.server.workflow.default_workflows import default_fit_workflow,default_evaluate_workflow
from mai_measurement import ObservedRawGrid, measured_workflow, server_span, reset_server, server_report
from central_smoke.identity import write_json,sha256_file
from .task import fresh_global,cfg
from .resources import resource_sha256
from .strategy import CheckedFedAvg
from .math_checks import arrays_from_model,load_arrays

class BoundedGrid(Grid):
    """Same Flower transport, with an explicit timeout and message-count diagnostics."""
    def __init__(self,inner,seconds):
        from mai_runtime_identity import IdentityCheckedTransport
        self.inner=IdentityCheckedTransport(inner);self.seconds=seconds;self.exchanges=[]
    @property
    def run(self):return self.inner.run
    def set_run(self,run_id):return self.inner.set_run(run_id)
    def create_message(self,*a,**kw):return self.inner.create_message(*a,**kw)
    def get_node_ids(self):return self.inner.get_node_ids()
    def push_messages(self,messages):return self.inner.push_messages(messages)
    def pull_messages(self,message_ids):return self.inner.pull_messages(message_ids)
    def send_and_receive(self,messages,*,timeout=None):
        sent=list(messages);start=time.perf_counter()
        replies=list(self.inner.send_and_receive(sent,timeout=self.seconds if timeout is None else min(timeout,self.seconds)))
        self.exchanges.append({'outgoing_messages':len(sent),'replies':len(replies),'wall_seconds':time.perf_counter()-start,'errors':sum(m.has_error() for m in replies),'types':sorted({m.metadata.message_type for m in sent})})
        # Raise before a stock workflow can convert a missing response to stale success.
        if len(replies)!=len(sent) or any(m.has_error() for m in replies):raise RuntimeError('Message timeout/error or missing reply')
        return replies


def direct_evaluate(grid,strategy,r,phase):
    nodes=sorted(grid.get_node_ids())
    if len(nodes)!=3:raise RuntimeError('Exactly 3 nodes required for evaluation')
    messages=[Message(content=compat.evaluateins_to_recorddict(EvaluateIns(strategy.params,strategy.instruction(r,phase)),True),
        dst_node_id=n,message_type='evaluate',group_id=f'{phase}-{r}',ttl=cfg()['exchange_timeout_seconds']) for n in nodes]
    replies=list(grid.send_and_receive(messages));rows=[]
    for m in replies:
        res=compat.recorddict_to_evaluateres(m.content)
        if res.status.code!=Code.OK:raise RuntimeError('Evaluation failed')
        rows.append({'node_id':m.metadata.src_node_id,'count':res.num_examples,'metrics':res.metrics,'loss':res.loss})
    return strategy.consume_evaluation(r,rows,phase)

app=ServerApp()

@app.main()
def main(grid:Grid,context:Context):
    root=Path(context.run_config['runtime-root']).resolve();condition=context.run_config['condition']
    folder=root/'server_outputs'/condition;folder.mkdir(parents=True,exist_ok=True)
    reset_server();raw_grid=ObservedRawGrid(grid)
    try:
        if condition not in ('reference','noniid') or int(context.run_config['num-server-rounds'])!=cfg()['rounds']:raise ValueError('Only fixed B timing pilot cells permitted')
        runtime=json.loads((root/'runtime.json').read_text());conf=cfg()
        if runtime['fixed_config_sha256']!=resource_sha256('fixed_config.json'):raise ValueError('Runtime identity mismatch')
        bounded=BoundedGrid(raw_grid,conf['exchange_timeout_seconds'])
        t=time.perf_counter()
        while len(list(grid.get_node_ids()))<3:
            if time.perf_counter()-t>conf['node_wait_seconds']:raise TimeoutError('Waiting for 3 SuperNodes')
            time.sleep(.5)
        if len(list(grid.get_node_ids()))!=3:raise ValueError('Unexpected extra nodes')
        from mai_runtime_identity import network_handshake, EXPECTED_BUILD_ID
        identity_handshake=network_handshake(raw_grid,conf['exchange_timeout_seconds'])
        write_json(folder/'implementation_handshake.json',identity_handshake)
        print('FLOWER_PRG_NETWORK_IDENTITY_CHECK_PASS '+condition,flush=True)
        initial=fresh_global();strategy=CheckedFedAvg(root,condition,arrays_from_model(initial),runtime['run_token'])
        if strategy.initial_digest!=runtime['expected_initial_digest']:raise ValueError('Initial weights differ from central reference')
        with server_span('initial_evaluation',0):
            direct_evaluate(bounded,strategy,0,'initial')
        legacy=LegacyContext(context=context,config=ServerConfig(num_rounds=conf['rounds'],round_timeout=conf['exchange_timeout_seconds']),strategy=strategy)
        DefaultWorkflow(fit_workflow=measured_workflow(default_fit_workflow,'fit_workflow'),
                        evaluate_workflow=measured_workflow(default_evaluate_workflow,'evaluation_workflow'))(bounded,legacy)
        if len(strategy.fit_checks)!=conf['rounds'] or [h['round'] for h in strategy.history]!=list(range(conf['rounds']+1)):raise RuntimeError('Workflow did not complete every full-size fit/evaluate round')
        eligible=strategy.history[1:]
        minimum=min(h['validation_loss'] for h in eligible)
        best=next(h for h in eligible if h['validation_loss']<=minimum+conf['checkpoint_tie_tolerance'])
        last_digest=strategy.digest
        finalpath=strategy.checkpoints/'final_state.pt';torch.save(strategy.model.state_dict(),finalpath)
        chosen=torch.load(strategy.checkpoints/f"round_{best['round']:02d}.pt",map_location='cpu',weights_only=True)
        strategy.model.load_state_dict(chosen,strict=True)
        strategy.params=ndarrays_to_parameters(arrays_from_model(strategy.model));strategy.digest=best['model_digest']
        with server_span('selected_checkpoint_evaluation',best['round']):
            verified=direct_evaluate(bounded,strategy,best['round'],'selected')
        if any(abs(verified[k]-best[k])>1e-7 for k in ['validation_loss','train_loss']):raise ValueError('Selected checkpoint re-evaluation mismatch')
        for c in conf['clients']:
            if any(abs(verified['per_client'][c][k]-best['per_client'][c][k])>1e-7 for k in ['train_loss','validation_loss']):raise ValueError('Selected checkpoint client re-evaluation mismatch')
        selectedpath=strategy.checkpoints/'selected_state.pt';torch.save(strategy.model.state_dict(),selectedpath)
        write_json(folder/'grid_exchanges.json',{'exchanges':bounded.exchanges,'full_wire_bytes_measured':False})
        result={'status':'PASS','implementation_build_id':EXPECTED_BUILD_ID,'identity_mismatch_probe_rejected_before_training':True,'configuration':'B','condition':condition,'rounds_completed':conf['rounds'],'clients_per_round':3,'total_local_optimiser_steps':conf['rounds']*sum(conf['local_updates_per_round'].values()),
            'initial_state_digest':strategy.initial_digest,'final_state_digest':last_digest,'selected_round':best['round'],
            'initial_train_loss':strategy.history[0]['train_loss'],'initial_validation_loss':strategy.history[0]['validation_loss'],
            'final_train_loss':strategy.history[-1]['train_loss'],'final_validation_loss':strategy.history[-1]['validation_loss'],
            'selected_validation_loss':verified['validation_loss'],'selected_per_client':verified['per_client'],
            'aggregation_checks':strategy.fit_checks,'node_mapping':strategy.participants,
            'checkpoints':[{'file':p.name,'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(strategy.checkpoints.glob('*.pt'))],
            'flower_executed':True,'deployment_engine_used':True,'secagg_protocol_executed':False,'test_data_opened':False,
            'test_metrics_computed':False,'main_experiments_frozen':False,'raw_client_records_loaded_by_serverapp':False,
            'notes':['Scalar routine loss metrics are unprotected diagnostics.','No claim of OS-enforced client isolation.','Numerical B result; timing/payload summaries are separately audited; no privacy certification.']}
        write_json(folder/'measurement_server.json',server_report(raw_grid))
        write_json(folder/'server_result.json',result)
    except BaseException as e:
        write_json(folder/'measurement_server_partial.json',server_report(raw_grid,partial=True))
        write_json(folder/'server_result.json',{'status':'FAIL','configuration':'B','condition':condition,'exception':type(e).__name__,'message':str(e),'secagg_protocol_executed':False,'main_experiments_frozen':False})
        (root/f'error_{condition}_LOCAL_ONLY.txt').write_text(traceback.format_exc())
        raise
