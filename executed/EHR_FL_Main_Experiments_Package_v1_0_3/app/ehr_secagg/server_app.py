"""C with approved local PRG backport; integration is not privacy certification."""
from __future__ import annotations
import json,time,traceback
from pathlib import Path
import torch
from flwr.common import Context,ndarrays_to_parameters
from flwr.server import ServerApp,Grid,LegacyContext,ServerConfig
from flwr.server.workflow import DefaultWorkflow
from flwr.server.workflow.default_workflows import default_fit_workflow,default_evaluate_workflow
from mai_measurement import ObservedRawGrid, measured_workflow, server_span, reset_server, server_report
from central_smoke.identity import write_json,sha256_file
from ehr_fedavg.task import fresh_global,cfg
from ehr_fedavg.resources import resource_sha256
from ehr_fedavg.math_checks import arrays_from_model
from ehr_fedavg.server_app import direct_evaluate
from .strategy import ProtectedFedAvg
from .workflow import SecAggGrid,StrictSecAggPlusWorkflow
from .settings import scfg,SHA256 as SECURITY_CONFIG_SHA256
app=ServerApp()

@app.main()
def main(grid:Grid,context:Context):
    root=Path(context.run_config['runtime-root']).resolve();condition=context.run_config['condition']
    folder=root/'server_outputs'/condition;folder.mkdir(parents=True,exist_ok=True)
    reset_server();raw_grid=ObservedRawGrid(grid)
    try:
        if condition not in ('reference','noniid') or int(context.run_config['num-server-rounds'])!=cfg()['rounds']:raise ValueError('Only fixed C timing pilot cells permitted')
        runtime=json.loads((root/'runtime.json').read_text());conf=cfg()
        if runtime['fixed_config_sha256']!=resource_sha256('fixed_config.json'):raise ValueError('Runtime identity mismatch')
        if runtime.get('security_config_sha256')!=SECURITY_CONFIG_SHA256:raise ValueError('Security configuration identity mismatch')
        if runtime.get('toy_oracle_passed') is not True:raise RuntimeError('Separate toy protocol oracle not passed')
        bounded=SecAggGrid(raw_grid,conf['exchange_timeout_seconds'])
        t=time.perf_counter()
        while len(list(grid.get_node_ids()))<3:
            if time.perf_counter()-t>conf['node_wait_seconds']:raise TimeoutError('Waiting for 3 SuperNodes')
            time.sleep(.5)
        if len(list(grid.get_node_ids()))!=3:raise ValueError('Unexpected extra nodes')
        from mai_runtime_identity import network_handshake, EXPECTED_BUILD_ID
        identity_handshake=network_handshake(raw_grid,conf['exchange_timeout_seconds'])
        write_json(folder/'implementation_handshake.json',identity_handshake)
        print('FLOWER_PRG_NETWORK_IDENTITY_CHECK_PASS '+condition,flush=True)
        initial=fresh_global();strategy=ProtectedFedAvg(root,condition,arrays_from_model(initial),runtime['run_token'])
        if strategy.initial_digest!=runtime['expected_initial_digest']:raise ValueError('Initial weights differ from central reference')
        with server_span('initial_evaluation',0):
            direct_evaluate(bounded,strategy,0,'initial')
        legacy=LegacyContext(context=context,config=ServerConfig(num_rounds=conf['rounds'],round_timeout=conf['exchange_timeout_seconds']),strategy=strategy)
        secure_workflow=StrictSecAggPlusWorkflow()
        DefaultWorkflow(fit_workflow=measured_workflow(secure_workflow,'fit_workflow'),
                        evaluate_workflow=measured_workflow(default_evaluate_workflow,'evaluation_workflow'))(bounded,legacy)
        if len(secure_workflow.audit)!=conf['rounds'] or len(strategy.protocol_stages)!=4*conf['rounds']:raise RuntimeError("Four secure stages must complete in each round")
        write_json(folder/"protocol_stages.json",{"stages":strategy.protocol_stages,"rounds":secure_workflow.audit})
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
        result={'status':'PASS','implementation_build_id':EXPECTED_BUILD_ID,'identity_mismatch_probe_rejected_before_training':True,'configuration':'C','condition':condition,'rounds_completed':conf['rounds'],'clients_per_round':3,'total_local_optimiser_steps':conf['rounds']*sum(conf['local_updates_per_round'].values()),
            'initial_state_digest':strategy.initial_digest,'final_state_digest':last_digest,'selected_round':best['round'],
            'initial_train_loss':strategy.history[0]['train_loss'],'initial_validation_loss':strategy.history[0]['validation_loss'],
            'final_train_loss':strategy.history[-1]['train_loss'],'final_validation_loss':strategy.history[-1]['validation_loss'],
            'selected_validation_loss':verified['validation_loss'],'selected_per_client':verified['per_client'],
            'aggregation_checks':strategy.fit_checks,'node_mapping':strategy.participants,
            'checkpoints':[{'file':p.name,'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(strategy.checkpoints.glob('*.pt'))],
            'flower_executed':True,'deployment_engine_used':True,'secagg_protocol_executed':True,'test_data_opened':False,
            'test_metrics_computed':False,'main_experiments_frozen':False,'raw_client_records_loaded_by_serverapp':False,
            'security_claim_gate':scfg()['security_claim_gate'],'privacy_claim_validated':False,
            'notes':['Scalar routine evaluation metrics and fixed counts remain unprotected.',
            'No OS-enforced client isolation; host operator is trusted.',
            'Pinned local PRG backport is used; overall security review remains pending; timing measurements are separately audited.',
            'No raw update, output fingerprint, gradient or loss summary in masked-fit metadata.']}
        write_json(folder/'measurement_server.json',server_report(raw_grid))
        write_json(folder/'server_result.json',result)
    except BaseException as e:
        write_json(folder/'measurement_server_partial.json',server_report(raw_grid,partial=True))
        write_json(folder/'server_result.json',{'status':'FAIL','configuration':'C','condition':condition,'exception':type(e).__name__,'message':str(e),'secagg_protocol_completed':False,'main_experiments_frozen':False})
        (root/f'error_{condition}_LOCAL_ONLY.txt').write_text(traceback.format_exc())
        raise
