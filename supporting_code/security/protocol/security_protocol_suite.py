"""Isolated local PUBLIC-array validation; never imports benchmark datasets."""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, sys, tempfile, time, traceback
from pathlib import Path
from types import SimpleNamespace

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--contract',required=True); ap.add_argument('--contract-sha256',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    source=Path(a.source).resolve(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    if sha(a.contract)!=a.contract_sha256: raise RuntimeError('Acceptance contract changed')
    contract=json.loads(Path(a.contract).read_text(encoding='utf-8-sig'))
    sys.path[:0]=[str(source),str(source/'app'),str(source/'validation_support')]
    import numpy as np
    from flwr.client import ClientApp, NumPyClient
    from flwr.client.mod import secaggplus_mod
    from flwr.common import Context, RecordDict, ConfigRecord, MetricRecord, FitIns, Message, Error, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.common import recorddict_compat as compat
    from flwr.server import Grid, LegacyContext, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.common.secure_aggregation.secaggplus_constants import Stage,Key,RECORD_KEY_CONFIGS,RECORD_KEY_STATE
    from flwr.server.workflow.constant import MAIN_PARAMS_RECORD,MAIN_CONFIGS_RECORD,Key as WK
    from ehr_secagg.policy import validate_cohort, validate_public_fit_metrics, local_range_check, restored_float32, conservative_quantization_bound
    from ehr_secagg.workflow import StrictSecAggPlusWorkflow
    from ehr_secagg.wire import check_masked_reply
    from ehr_secagg.settings import scfg
    from mai_runtime_identity import binding_identity
    from prg_validation.public_protocol import run_case as retained_oracle
    from flwr.common.secure_aggregation.crypto.symmetric_encryption import decrypt,encrypt
    from flwr.common.secure_aggregation.secaggplus_utils import share_keys_plaintext_concat,share_keys_plaintext_separate
    mod=importlib.import_module('flwr.client.mod.secure_aggregation.secaggplus_mod')
    params=dict(contract['main_crypto_parameters'])
    for name in ('max_weight','clipping_range','timeout'): params[name]=float(params[name])
    assert scfg()['secagg']==params
    identity=binding_identity()
    script_before=sha(__file__)
    nids=[11,22,33]; public_input='0'*64
    base=[np.array([-1.,0.,1.,2.],np.float32),np.array([2.,1.,-2.,0.],np.float32),np.array([0.,3.,1.,-1.],np.float32)]
    results=[]

    def public_summary(msg,stage,where):
        if msg.has_error(): return {'where':where,'stage':stage,'is_error':True}
        c=msg.content
        populated=sum(len(v)>0 for v in c.array_records.values())
        secure=c.config_records.get(RECORD_KEY_CONFIGS,{})
        blobs=secure.get(Key.MASKED_PARAMETERS,[])
        return {'where':where,'stage':stage,'is_error':False,'array_record_names':sorted(c.array_records),
                'populated_array_records':populated,'config_record_names':sorted(c.config_records),
                'metric_record_names':sorted(c.metric_records),'masked_blob_count':len(blobs),
                'masked_payload_bytes':sum(len(v) for v in blobs if isinstance(v,bytes)),
                'fit_metric_field_names':sorted(c.config_records.get('fitres.metrics',{}))}

    def protocol_case(spec):
        mode=spec.get('mutation'); target=spec.get('stage'); weights=spec.get('weights',[64,128,256])
        values=[x.copy() for x in base]
        if spec.get('off_lattice'): values=[x+np.array([.001,.023,-.041,.079],np.float32)*(i+1) for i,x in enumerate(values)]
        if spec.get('edge'): values=[np.array([-8.,8.,0.,1.],np.float32),np.array([8.,-8.,1.,0.],np.float32),np.array([0.,8.,-8.,1.],np.float32)]
        expected=sum((v.astype(np.float64)*w/sum(weights) for v,w in zip(values,weights)),np.zeros(4,np.float64))
        calls={str(n):0 for n in nids}; ingress=[]; stages=[]; request_counts={}; injected=False
        class ToyClient(NumPyClient):
            def __init__(self,i): self.i=i
            def fit(self,parameters,config):
                calls[str(nids[self.i])]+=1
                local_range_check([values[self.i]])
                return [values[self.i].copy()],weights[self.i],{'client':f'C{self.i+1}','input_digest':public_input,'steps':0,'draws':weights[self.i]}
        def guard(msg,ctx,call_next):
            if msg.metadata.message_type!='train': return call_next(msg,ctx)
            if RECORD_KEY_CONFIGS not in msg.content.config_records: raise RuntimeError('Unprotected fit instruction refused')
            rec=msg.content.config_records[RECORD_KEY_CONFIGS]; st=rec[Key.STAGE]
            if st==Stage.SETUP:
                expected_setup={Key.SAMPLE_NUMBER:3,Key.SHARE_NUMBER:params['num_shares'],Key.THRESHOLD:params['reconstruction_threshold'],Key.CLIPPING_RANGE:params['clipping_range'],Key.TARGET_RANGE:params['quantization_range'],Key.MOD_RANGE:params['modulus_range'],Key.MAX_WEIGHT:params['max_weight']}
                if any(rec.get(k)!=v or type(rec.get(k))!=type(v) for k,v in expected_setup.items()): raise ValueError('Unexpected SecAgg settings')
            if st==Stage.UNMASK and (len(rec[Key.ACTIVE_NODE_ID_LIST])!=3 or rec[Key.DEAD_NODE_ID_LIST]): raise RuntimeError('Incomplete cohort refused before unmasking')
            reply=secaggplus_mod(msg,ctx,call_next)
            if reply.has_error(): raise RuntimeError('Client error')
            check_masked_reply(reply.content,st,toy=True)
            return reply
        ca=ClientApp(client_fn=lambda ctx:ToyClient(int(ctx.node_config['toy-index'])).to_client(),mods=[guard])
        contexts={n:Context(run_id=101,node_id=n,node_config={'toy-index':i},state=RecordDict(),run_config={}) for i,n in enumerate(nids)}
        class Strategy(FedAvg):
            def __init__(self):
                super().__init__(initial_parameters=ndarrays_to_parameters([np.zeros(4,np.float32)]),fit_metrics_aggregation_fn=lambda _: {})
                self.result=None; self.calls=0; self.last_round=spec.get('last_round',0); self.secagg_stage=None; self.protocol_stages=[]
            def configure_fit(self,r,p,cm): return [(SimpleNamespace(node_id=n,cid=str(n)),FitIns(p,{'server_round':r})) for n in nids]
            def aggregate_fit(self,r,rows,failures):
                if self.secagg_stage!=Stage.UNMASK or failures or len(rows)!=3: raise RuntimeError('Rejected aggregation callback')
                for _,res in rows: validate_public_fit_metrics(res.metrics,public_input,toy=True)
                arrs=[parameters_to_ndarrays(res.parameters) for _,res in rows]
                restored,_=restored_float32(arrs,[{'shape':[4]}])
                self.result=restored[0]; self.calls+=1; self.last_round=r
                return ndarrays_to_parameters(restored),{}
        strategy=Strategy(); round_no=spec.get('round',1)
        ctx=LegacyContext(context=Context(run_id=101,node_id=0,node_config={},state=RecordDict(),run_config={}),config=ServerConfig(num_rounds=1),strategy=strategy)
        ctx.state.array_records[MAIN_PARAMS_RECORD]=compat.parameters_to_arrayrecord(ndarrays_to_parameters([np.zeros(4,np.float32)]),True)
        ctx.state.config_records[MAIN_CONFIGS_RECORD]=ConfigRecord({WK.CURRENT_ROUND:round_no})
        def mutate_request(incoming,st):
            c=incoming.content.config_records[RECORD_KEY_CONFIGS]
            if mode=='wrong_stage': c[Key.STAGE]=Stage.COLLECT_MASKED_VECTORS
            elif mode=='wrong_settings': c[Key.THRESHOLD]=1
            elif mode=='incomplete_unmask': c[Key.ACTIVE_NODE_ID_LIST]=[11,22]; c[Key.DEAD_NODE_ID_LIST]=[33]
            elif mode=='duplicate_public_key':
                keys=list(c); c[keys[1]]=list(c[keys[0]])
            elif mode=='own_key_replaced': c['33']=[b'PUBLIC_INVALID_OWN_KEY',c['33'][1]]
            elif mode=='peer_key_malformed':
                keys=[k for k in c if k!='33']; c[keys[0]]=[c[keys[0]][0],b'PUBLIC_INVALID_PEER_KEY']
            elif mode in ('ciphertext_bitflip','wrong_share_source','wrong_share_destination'):
                ciphertexts=list(c[Key.CIPHERTEXT_LIST]); sources=c[Key.SOURCE_LIST]
                if mode=='ciphertext_bitflip':
                    item=bytearray(ciphertexts[0]); item[len(item)//2]^=1; ciphertexts[0]=bytes(item)
                else:
                    local=mod.SecAggPlusState(**contexts[33].state.config_records[RECORD_KEY_STATE]); key=local.ss2_dict[sources[0]]
                    src,dst,b,sk=share_keys_plaintext_separate(decrypt(key,ciphertexts[0]))
                    plaintext=share_keys_plaintext_concat(999 if mode=='wrong_share_source' else src,999 if mode=='wrong_share_destination' else dst,b,sk)
                    ciphertexts[0]=encrypt(key,plaintext)
                c[Key.CIPHERTEXT_LIST]=ciphertexts
            elif mode=='parallel_list_length_mismatch': c[Key.SOURCE_LIST]=list(c[Key.SOURCE_LIST])[:-1]
        def mutate_reply(reply,st):
            c=reply.content
            if mode=='plaintext_array': c.array_records['unexpected_plaintext']=compat.parameters_to_arrayrecord(ndarrays_to_parameters([np.ones(4,np.float32)]),True)
            elif mode=='output_digest': c.config_records['fitres.metrics']['output_digest']='PUBLIC_SENTINEL_OUTPUT_DIGEST'
            elif mode=='loss_metric': c.metric_records['unexpected_loss']=MetricRecord({'loss':.25})
            elif mode=='free_text_status': c.config_records['fitres.status']['message']='PUBLIC_SENTINEL_STATUS'
            elif mode=='unknown_record': c.config_records['unexpected_config']=ConfigRecord({'public_sentinel':'PUBLIC_SENTINEL_CONFIG'})
        class ToyGrid(Grid):
            @property
            def run(self): return SimpleNamespace(run_id=101)
            def set_run(self,r): raise RuntimeError('Unused transport operation')
            def create_message(self,*aa,**kk): raise RuntimeError('Unused transport operation')
            def get_node_ids(self): return nids
            def push_messages(self,m): raise RuntimeError('Unused transport operation')
            def pull_messages(self,m): raise RuntimeError('Unused transport operation')
            def send_and_receive(self,messages,*,timeout=None):
                nonlocal injected
                sent=list(messages); st=sent[0].content.config_records[RECORD_KEY_CONFIGS][Key.STAGE]; stages.append(st)
                request_counts[st]=request_counts.get(st,0)+len(sent); replies=[]
                for request in sent:
                    incoming=copy.deepcopy(request)
                    if st==target and incoming.metadata.dst_node_id==33 and mode not in ('missing','duplicate','foreign','error','plaintext_array','output_digest','loss_metric','free_text_status','unknown_record'):
                        mutate_request(incoming,st); injected=True
                    ingress.append(public_summary(incoming,st,'client_request'))
                    reply=ca(incoming,contexts[incoming.metadata.dst_node_id])
                    if st==target and incoming.metadata.dst_node_id==33 and mode in ('plaintext_array','output_digest','loss_metric','free_text_status','unknown_record'):
                        mutate_reply(reply,st); injected=True
                    if st==target and incoming.metadata.dst_node_id==33 and mode=='error': reply=Message(error=Error(code=1,reason='PUBLIC_INJECTED_ERROR'),reply_to=incoming); injected=True
                    replies.append(reply)
                if st==target:
                    if mode=='missing': replies.pop(); injected=True
                    elif mode=='duplicate': replies[-1]=copy.deepcopy(replies[0]); injected=True
                    elif mode=='foreign':
                        foreign=Message(content=RecordDict(),dst_node_id=999,message_type='train',group_id='public-foreign')
                        replies[-1]=Message(content=replies[-1].content,reply_to=foreign); injected=True
                for reply in replies: ingress.append(public_summary(reply,st,'server_ingress_before_workflow'))
                if any(r.has_error() for r in replies): raise RuntimeError('Injected or actual error reply refused')
                validate_cohort([m.metadata.dst_node_id for m in sent],[m.metadata.src_node_id for m in replies])
                for reply in replies: check_masked_reply(reply.content,st,toy=True)
                return replies
        caught=None; returned=False
        try: StrictSecAggPlusWorkflow()(ToyGrid(),ctx); returned=True
        except Exception as e: caught={'type':type(e).__name__,'message':str(e)[:240]}
        error=None
        if strategy.result is not None: error=float(np.max(np.abs(strategy.result.astype(np.float64)-expected)))
        bound=float(conservative_quantization_bound(weights))
        r={'local_fit_callbacks':calls,'total_local_fit_callbacks':sum(calls.values()),'aggregate_callbacks':strategy.calls,'model_advancements':strategy.calls,
           'workflow_returned':returned,'exception':caught,'observed_stages':stages,'fault_injected':injected,
           'max_abs_average_error':error,'quantization_bound':bound,'message_boundary_observations':ingress,
           'test_harness':'real Flower ClientApp+SecAgg modifier, project StrictSecAggPlusWorkflow/cohort/wire/range/aggregate validators; public-toy budgets',
           'main_crypto_parameters':params,'benchmark_data_read':False,'trained_model_loaded':False,'secrets_exported':False}
        if spec['expectation']=='reject': r['status']='PASS' if caught and caught['type'] in spec['expected_exception_types'] and strategy.calls==0 else 'FAIL'
        else:
            wire_rows=[x for x in ingress if x['where']=='server_ingress_before_workflow' and x['stage']==Stage.COLLECT_MASKED_VECTORS]
            r['status']='PASS' if returned and strategy.calls==1 and error<=bound and len(wire_rows)==3 and all(x['populated_array_records']==0 and x['masked_blob_count']>0 for x in wire_rows) else 'FAIL'
        contexts.clear()
        return r

    def direct_case(spec):
        kind=spec['direct']; got=None; value=None
        try:
            if kind=='B_plaintext_control':
                class BClient(NumPyClient):
                    def fit(self,p,c): return [base[0].copy()],64,{'client':'C1','input_digest':public_input,'steps':0,'draws':64}
                app=ClientApp(client_fn=lambda ctx:BClient().to_client())
                message=Message(content=compat.fitins_to_recorddict(FitIns(ndarrays_to_parameters([np.zeros(4,np.float32)]),{}),True),dst_node_id=11,message_type='train')
                reply=app(message,Context(run_id=102,node_id=11,node_config={},state=RecordDict(),run_config={}))
                row=public_summary(reply,'fit','server_ingress_before_workflow')
                decoded=parameters_to_ndarrays(compat.recorddict_to_fitres(reply.content,keep_input=True).parameters)
                value={'plaintext_control_visible':row['populated_array_records']>0 and np.array_equal(decoded[0],base[0]),'message_boundary_observations':[row]}
            elif kind=='range':
                fixture=spec['fixture']; arr=np.array([-8.,8.],np.float32)
                if fixture=='above':arr[1]=8.001
                elif fixture=='nan':arr[0]=np.nan
                elif fixture=='inf':arr[0]=np.inf
                elif fixture=='dtype':arr=arr.astype(np.float64)
                value=local_range_check([] if fixture=='empty' else [arr])
            elif kind=='weight': value=float(conservative_quantization_bound(spec['weights']))
            elif kind=='carrier_mismatch': restored_float32([[np.array([1.],np.float64)],[np.array([2.],np.float64)],[np.array([1.],np.float64)]],[{'shape':[1]}])
            elif kind=='aggregate_shape': restored_float32([[np.ones(2,np.float64)]]*3,[{'shape':[1]}])
            elif kind=='stale_digest': validate_public_fit_metrics({'client':'C1','input_digest':'1'*64,'steps':0,'draws':64},public_input,toy=True)
            elif kind=='bad_stage':mod.check_stage(Stage.UNMASK,ConfigRecord({Key.STAGE:Stage.SHARE_KEYS}))
            elif kind=='repeat_collect_stage':mod.check_stage(Stage.COLLECT_MASKED_VECTORS,ConfigRecord({Key.STAGE:Stage.COLLECT_MASKED_VECTORS}))
            elif kind=='setup_restart_characterization':
                mod.check_stage(Stage.SHARE_KEYS,ConfigRecord({Key.STAGE:Stage.SETUP})); value={'upstream_setup_restart_allowed':True,'warning_expected':True,'not_a_replay_resistance_guarantee':True}
            elif kind=='repeated_local_fit_file_gate':
                from ehr_secagg.local_task import SecAggLocalTask
                with tempfile.TemporaryDirectory() as tmp:
                    task=object.__new__(SecAggLocalTask); task.output=Path(tmp); task.validate=lambda p,c:public_input
                    (Path(tmp)/'fit_round_01.json').write_text('{}')
                    task.fit([],{'server_round':1})
            elif kind=='unprotected_fit':
                from ehr_secagg.client_app import guarded_secaggplus_mod
                called=[0]
                def callback(*args):called[0]+=1; raise RuntimeError('Forbidden callback reached')
                msg=Message(content=RecordDict(),dst_node_id=11,message_type='train')
                guarded_secaggplus_mod(msg,None,callback)
            elif kind=='project_aggregate_outside_unmask':
                from ehr_secagg.strategy import ProtectedFedAvg
                obj=object.__new__(ProtectedFedAvg); obj.secagg_stage=Stage.SETUP
                obj.aggregate_fit(1,[],[])
            elif kind=='project_aggregate_missing_participants':
                from ehr_secagg.strategy import ProtectedFedAvg
                obj=object.__new__(ProtectedFedAvg); obj.secagg_stage=Stage.UNMASK; obj.last_round=0; obj.round_start={1:0}
                obj.aggregate_fit(1,[],[])
            else: raise KeyError('Unknown direct fixture')
        except Exception as e:got={'type':type(e).__name__,'message':str(e)[:240]}
        expected=spec['expectation']; status=('PASS' if got and got['type'] in spec['expected_exception_types'] else 'FAIL') if expected=='reject' else ('PASS' if got is None else 'FAIL')
        if kind=='B_plaintext_control' and not got and not value['plaintext_control_visible']:status='FAIL'
        return {'status':status,'exception':got,'public_result':value,'aggregate_callbacks':0,'model_advancements':0,
                'local_fit_callbacks':{},'scope':'direct named code path; no benchmark execution','benchmark_data_read':False}

    for spec in contract['cases']:
        start=time.time(); print('BEGIN '+spec['id'],flush=True)
        try:
            if spec.get('oracle'):
                result=retained_oracle(spec['weights'],lattice=not spec.get('off_lattice',False),mode='candidate')
                result.update(aggregate_callbacks=result['aggregate_calls'],model_advancements=result['aggregate_calls'])
            elif spec.get('direct'): result=direct_case(spec)
            else: result=protocol_case(spec)
        except Exception as e:
            result={'status':'ERROR','exception':{'type':type(e).__name__,'message':str(e)[:240]},'harness_traceback':traceback.format_exc()}
        result.update(id=spec['id'],group=spec['group'],expectation=spec['expectation'],fixture=spec,end_unix=time.time(),start_unix=start)
        dump(out/(spec['id']+'.json'),result); results.append(result)
        print('END '+spec['id']+' '+result['status'],flush=True)
    report={'status':'PASS' if all(r['status']=='PASS' for r in results) else 'FAIL' if any(r['status']=='FAIL' for r in results) else 'ERROR',
            'cases':results,'counts':{s:sum(r['status']==s for r in results) for s in ['PASS','FAIL','ERROR','NOT_RUN','OUT_OF_SCOPE']},
            'script_sha256':script_before,'script_unchanged':script_before==sha(__file__),'contract_sha256':a.contract_sha256,
            'runtime_identity':identity,'main_crypto_parameters':params,'not_main_training_budget':True,
            'benchmark_training_or_inference':False,'no_raw_dataset_read':True,'privacy_certification':False,
            'runtime_or_original_source_modified':False,'scope':'In-memory local public toy tests plus direct named validator paths; no TLS/host isolation/collusion/security proof.'}
    dump(out/'Protocol_Results.json',report)
    print(json.dumps({'status':report['status'],'counts':report['counts']}))
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':sys.exit(main())
