"""Measurement regressions. Stand-in tests are not Flower deployment evidence."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,sys,tempfile,time,types,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from flower_message_fixture import received_message
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'app')]
import mai_measurement as mm
from timing_support.analysis import overhead,descriptive,reconcile_boundary,summarise,extract_cell,write_csv
from timing_support.data import verify_timing_inputs
from ehr_fedavg.task import cfg
HAS_FLOWER=importlib.util.find_spec('flwr') is not None

class Sized(dict):
    def __init__(self,n=0,*a,**k):super().__init__(*a,**k);self.n=n
    def count_bytes(self):return self.n
class Content:
    def __init__(self,arrays=None,metrics=None,configs=None):
        self.array_records=arrays or {};self.metric_records=metrics or {};self.config_records=configs or {}
    def __len__(self):return len(self.array_records)+len(self.metric_records)+len(self.config_records)
class Msg:
    def __init__(self,content,kind='train',group='1',ident='abc',node=11,reply=False):
        self.content=content;self.metadata=types.SimpleNamespace(message_type=kind,group_id=group,message_id=ident,
            dst_node_id=node,src_node_id=node,reply_to_message_id=ident)
    def has_error(self):return False

def count(n=1):return {'array_bytes':n,'metric_bytes':0,'config_bytes':0,'total_bytes':n}

def fixture(mode):
    rows=[];msgs=[];i=0
    for ci,c in enumerate(('C1','C2','C3'),11):
        phases=[(0,'identity','handshake')]*2+[(0,'evaluate','initial')]
        for r in range(1,7):
            phases += [(r,s,'fit') for s in (('fit',) if mode=='B' else ('setup','share_keys','collect_masked_vectors','unmask'))]+[(r,'evaluate','round')]
        phases += [(6,'evaluate','selected')]
        for r,stage,phase in phases:
            i+=1;typ='query' if phase=='handshake' else 'evaluate' if stage=='evaluate' else 'train'
            desc={'round':r,'stage':stage,'phase':phase,'message_type':typ,'node_id':ci,'request_id':str(i)}
            local=[]
            if stage in ('fit','collect_masked_vectors'):
                local=[{'name':n,'wall_ns':10,'cpu_ns':8,'completed':True} for n in ('local_fit_with_checks','data_load_with_checks')]
            rows.append({'status':'PASS','client':c,**desc,'incoming_record_bytes':count(10),'outgoing_record_bytes':count(20),
                'record_values_written':False,'wire_bytes_measured':False,'callback_wall_ns':50,'callback_cpu_ns':40,
                'byte_counter_wall_ns':3,'local_spans':local})
            msgs.append({**desc,'outgoing_record_bytes':count(10),'incoming_record_bytes':count(20)})
    return {'exchanges':[{'messages':msgs}]},rows

class MeasurementTests(unittest.TestCase):
    def test_count_values_not_exported(self):
        c=Content(configs={'masked':Sized(18,secret=b'do-not-export')},arrays={'x':Sized(4)})
        row=mm.record_bytes(c);self.assertEqual(row['total_bytes'],22)
        self.assertNotIn('secret',json.dumps(row));self.assertNotIn('do-not-export',json.dumps(row))
    def test_empty_content_count(self):self.assertEqual(mm.record_bytes(Content())['total_bytes'],0)
    def test_negative_size_refused(self):
        with self.assertRaises(ValueError):mm.record_bytes(Content(arrays={'x':Sized(-1)}))
    def test_nonint_size_refused(self):
        with self.assertRaises(ValueError):mm.record_bytes(Content(arrays={'x':Sized(1.2)}))
    def test_bool_size_refused(self):
        with self.assertRaises(ValueError):mm.record_bytes(Content(arrays={'x':Sized(True)}))
    def test_descriptor_regular_train(self):
        m=Msg(Content(configs={'fitins.config':{'server_round':2}}))
        self.assertEqual(mm.descriptor(m)['round'],2)
    def test_descriptor_secagg(self):
        m=Msg(Content(configs={'secaggplus_configs':{'stage':'unmask'}}),group='6')
        self.assertEqual(mm.descriptor(m)['stage'],'unmask')
    def test_unknown_secagg_stage_refused(self):
        with self.assertRaises(ValueError):mm.descriptor(Msg(Content(configs={'secaggplus_configs':{'stage':'invented'}})))
    def test_forbidden_round_refused(self):
        with self.assertRaises(ValueError):mm.descriptor(Msg(Content(configs={'fitins.config':{'server_round':30}})))
    def test_initial_eval_classification(self):
        m=Msg(Content(configs={'evaluateins.config':{'server_round':0,'phase':'initial'}}),kind='evaluate')
        self.assertEqual(mm.descriptor(m)['phase'],'initial')
    def test_unknown_type_refused(self):
        with self.assertRaises(ValueError):mm.descriptor(Msg(Content(),kind='predict'))
    def test_real_span_integer_and_serializable(self):
        mm.reset_server()
        with mm.server_span('fixture',1):time.sleep(.001)
        r=mm.server_report(types.SimpleNamespace(rows=[]));json.dumps(r,allow_nan=False)
        self.assertGreaterEqual(r['events'][0]['wall_ns'],1000000)
    def test_exception_span_recorded_false(self):
        mm.reset_server()
        with self.assertRaises(RuntimeError):
            with mm.server_span('fixture',1):raise RuntimeError('expected')
        self.assertFalse(mm._SERVER_EVENTS[-1]['completed'])
    def test_client_span_records_only_in_context(self):
        events=[];token=mm._ACTIVE_CLIENT.set(events)
        try:
            with mm.client_span('local_fit_with_checks'):pass
        finally:mm._ACTIVE_CLIENT.reset(token)
        self.assertEqual(len(events),1);self.assertIsInstance(events[0]['wall_ns'],int)
    def test_unknown_client_span_refused(self):
        with self.assertRaises(ValueError):
            with mm.client_span('dump_keys'):pass
    def test_outer_counter_no_payload_modification(self):
        with tempfile.TemporaryDirectory() as td:
            context=types.SimpleNamespace(run_config={'runtime-root':td,'condition':'reference'},node_config={'client-id':'C1'})
            request=Msg(Content(configs={'fitins.config':Sized(20,server_round=1)}))
            reply=Msg(Content(configs={'masked':Sized(100,blob=b'private')}))
            got=mm.timing_boundary_mod(request,context,lambda m,c:reply)
            self.assertIs(got,reply)
            raw=next(Path(td).rglob('*.json')).read_text();r=json.loads(raw)
            self.assertEqual(r['incoming_record_bytes']['total_bytes'],20)
            self.assertEqual(r['outgoing_record_bytes']['total_bytes'],100)
            self.assertNotIn('private',raw);self.assertNotIn('blob',raw)
    def test_duplicate_client_request_refused(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=types.SimpleNamespace(run_config={'runtime-root':td,'condition':'reference'},node_config={'client-id':'C1'})
            f=lambda:Msg(Content(configs={'fitins.config':Sized(2,server_round=1)}))
            mm.timing_boundary_mod(f(),ctx,lambda m,c:Msg(Content()))
            with self.assertRaises(ValueError):mm.timing_boundary_mod(f(),ctx,lambda m,c:Msg(Content()))
    def test_client_exception_logs_type_not_content(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=types.SimpleNamespace(run_config={'runtime-root':td,'condition':'reference'},node_config={'client-id':'C1'})
            def fail(m,c):raise RuntimeError('do_not_export_secret')
            with self.assertRaises(RuntimeError):mm.timing_boundary_mod(Msg(Content(configs={'fitins.config':Sized(2,server_round=1)})),ctx,fail)
            raw=next(Path(td).rglob('*.json')).read_text();self.assertNotIn('do_not_export_secret',raw);self.assertEqual(json.loads(raw)['status'],'FAIL')
    def test_server_counts_before_reply_mutation(self):
        request=Msg(Content(configs={'fitins.config':Sized(10,server_round=1)}))
        reply=Msg(Content(configs={'x':Sized(30)}))
        inner=types.SimpleNamespace(send_and_receive=lambda s,timeout=None:[reply])
        o=mm.ObservedRawGrid(inner);o.send_and_receive([request]);reply.content.config_records.clear()
        self.assertEqual(o.rows[0]['messages'][0]['incoming_record_bytes']['total_bytes'],30)
    def test_missing_response_refused(self):
        o=mm.ObservedRawGrid(types.SimpleNamespace(send_and_receive=lambda s,timeout=None:[]))
        with self.assertRaises(RuntimeError):o.send_and_receive([Msg(Content(configs={'fitins.config':Sized(10,server_round=1)}))])

class ReconciliationTests(unittest.TestCase):
    def test_B_match(self):
        s,c=fixture('B');r=reconcile_boundary(s,c,'B');self.assertEqual(r['matched_messages'],48);self.assertFalse(r['server_totals_added'])
    def test_C_match(self):
        s,c=fixture('C');self.assertEqual(reconcile_boundary(s,c,'C')['matched_messages'],102)
    def test_no_double_count(self):
        s,c=fixture('B');r=reconcile_boundary(s,c,'B')
        self.assertEqual(r['client_total_incoming_record_bytes']+r['client_total_outgoing_record_bytes'],48*30)
    def test_missing_client_row(self):
        s,c=fixture('B');c.pop()
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'B')
    def test_duplicate_client_row(self):
        s,c=fixture('B');c.append(copy.deepcopy(c[0]))
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'B')
    def test_changed_byte_count(self):
        s,c=fixture('C');c[-1]['outgoing_record_bytes']=count(21)
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'C')
    def test_unknown_client(self):
        s,c=fixture('C');c[5]['client']='C4'
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'C')
    def test_invalid_timer(self):
        s,c=fixture('C');c[0]['callback_wall_ns']=-1
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'C')
    def test_missing_fit_span(self):
        s,c=fixture('C');next(r for r in c if r['stage']=='collect_masked_vectors')['local_spans']=[]
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'C')
    def test_plain_values_claim_refused(self):
        s,c=fixture('B');c[0]['record_values_written']=True
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'B')
    def test_incorrect_phase_refused(self):
        s,c=fixture('C');c[0]['phase']='fit'
        with self.assertRaises(ValueError):reconcile_boundary(s,c,'C')

class SummaryTests(unittest.TestCase):
    def cells(self):
        policy=json.loads((ROOT/'config/timing_policy.json').read_text());cs=[]
        for p in policy['plan']:
            n=p['pair'];factor=1 if p['configuration']=='B' else 2
            cs.append({'status':'PASS','pair':n,'configuration':p['configuration'],'initial_state_digest':'same','measured_rounds':[2,3,4,5,6],
                       'per_run_medians':{'fit_workflow_wall_ns':factor*100*n,'fit_record_bytes_total':factor*1000}})
        return cs,policy['plan']
    def test_extract_full_six_round_fixture_and_csv(self):
        from central_smoke.identity import write_json
        for mode in ('B','C'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);server,clients=fixture(mode);events=[]
                for r in range(1,7):
                    for name,duration in [('fit_workflow',100 if r>1 else 9999),('evaluation_workflow',50),('outer_round_including_evaluation',10100)]:
                        events.append({'name':name,'round':r,'start_ns':0,'end_ns':duration,'wall_ns':duration,'cpu_ns':0,'completed':True})
                    names=['fedavg_native_aggregate','independent_float_average_audit'] if mode=='B' else ['secagg_'+x for x in ('setup','share_keys','collect_masked_vectors','unmask')]
                    for name in names:events.append({'name':name,'round':r,'start_ns':0,'end_ns':10,'wall_ns':10,'cpu_ns':0,'completed':True})
                server['events']=events
                write_json(root/'server_outputs/reference/measurement_server.json',server)
                write_json(root/'server_outputs/reference/server_result.json',{'initial_state_digest':'fixture','final_validation_loss':1.})
                for i,row in enumerate(clients):write_json(root/'measurement_client/reference'/str(i)/'row.json',row)
                result=extract_cell(root,mode,1)
                self.assertEqual(result['per_run_medians']['fit_workflow_wall_ns'],100)
                self.assertEqual(len(result['rounds']),6);self.assertTrue(result['rounds'][0]['warmup'])
                write_csv(root/'rounds.csv',result['rounds']);self.assertEqual(len((root/'rounds.csv').read_text().splitlines()),7)
    def test_paired_three_not_thirty(self):
        c,p=self.cells();r=summarise(c,p)
        self.assertEqual(r['paired_repetitions'],3);self.assertEqual(r['across_run_descriptives']['fit_workflow_wall_ns']['B']['n'],3)
        self.assertFalse(r['confidence_intervals_or_significance_tests'])
    def test_pair_ratios(self):
        c,p=self.cells();r=summarise(c,p)
        self.assertTrue(all(x['overhead_percent']==100 for x in r['contrasts']))
    def test_negative_overhead_valid(self):self.assertEqual(overhead(2,1),-50)
    def test_zero_denominator_undefined(self):self.assertIsNone(overhead(0,1))
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):overhead(float('nan'),1)
    def test_incomplete_repeats_refused(self):
        c,p=self.cells()
        with self.assertRaises(ValueError):summarise(c[:-1],p)
    def test_warmup_changed_refused(self):
        c,p=self.cells();c[0]['measured_rounds']=[1,2,3,4,5]
        with self.assertRaises(ValueError):summarise(c,p)
    def test_different_initial_refused(self):
        c,p=self.cells();c[0]['initial_state_digest']='different'
        with self.assertRaises(ValueError):summarise(c,p)
    def test_wrong_order_refused(self):
        c,p=self.cells();c[0],c[1]=c[1],c[0]
        with self.assertRaises(ValueError):summarise(c,p)
    def test_csv_keeps_warmup_and_all_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'t.csv';write_csv(p,[{'round':1,'warmup':True},{'round':2,'warmup':False}]);self.assertIn('1,True',p.read_text())
    def test_raw_intervals_retained(self):
        self.assertIn("'rounds':rows",(ROOT/'timing_support/analysis.py').read_text())
    def test_no_target_performance_cutoff(self):
        c,p=self.cells();self.assertIsNone(summarise(c,p)['performance_acceptance_threshold'])

class ProtocolTests(unittest.TestCase):
    def test_numerical_kernel_source_matches_accepted_predecessor(self):
        rec=json.loads((ROOT/'reference/unchanged_numerical_source.json').read_text())
        source=(ROOT/'app/ehr_fedavg/task.py').read_text()
        functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        for name,digest in rec['task_functions'].items():self.assertEqual(hashlib.sha256(functions[name].encode()).hexdigest(),digest)
        for name,digest in rec['whole_files'].items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
    def test_optimisation_fields_unchanged(self):
        old=json.loads((ROOT/'reference/fullsize_B.json').read_text());new=cfg()
        allowed={'stage','version','conditions','rounds','training_sessions_per_client','validation_sessions_per_client',
                 'local_updates_per_round','expected_total_updates','membership_sha256','schedule_sha256',
                 'resource_sample_seconds','implementation_decisions'}
        self.assertEqual({k:v for k,v in old.items() if k not in allowed},{k:v for k,v in new.items() if k not in allowed})
    def test_predeclared_counts(self):
        c=cfg();self.assertEqual(c['rounds'],6);self.assertEqual(c['training_sessions_per_client'],dict.fromkeys(['C1','C2','C3'],256))
        self.assertEqual(c['validation_sessions_per_client'],dict.fromkeys(['C1','C2','C3'],128))
    def test_schedule_replay(self):self.assertEqual(verify_timing_inputs(ROOT)['reference_draw_occurrences_per_run'],4608)
    def test_predefined_orders(self):
        p=json.loads((ROOT/'config/timing_policy.json').read_text())
        inherited=json.loads((ROOT/'reference/pilot_settings_v1_1.json').read_text())['pilot']
        self.assertEqual([p['plan'][i]['configuration'] for i in range(6)],['B','C','C','B','B','C'])
        self.assertEqual(p['repetitions'],inherited['timing_repetitions']);self.assertEqual(p['warmup_rounds'],[inherited['timing_warmup_rounds']])
    def test_round_count_in_project(self):
        from ehr_fedavg.host import write_project
        import tomllib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pyproject.toml';write_project(p,Path(d),'reference',19093)
            self.assertEqual(tomllib.loads(p.read_text())['tool']['flwr']['app']['config']['num-server-rounds'],6)
    def test_security_math_unchanged(self):
        from ehr_secagg.settings import scfg
        old=json.loads((ROOT/'reference/pilot_settings_v1_1.json').read_text())['secagg']
        for k,v in scfg()['secagg'].items():self.assertEqual(v,old[k])
    def test_observer_is_outermost(self):
        for m in ['ehr_fedavg','ehr_secagg']:
            s=(ROOT/'app'/m/'client_app.py').read_text();self.assertIn('mods=[timing_boundary_mod,implementation_identity_mod',s)
    def test_no_network_or_install_in_launcher(self):
        s=(ROOT/'scripts/run_on_wsl.sh').read_text();self.assertNotIn('pip install',s);self.assertNotIn('curl ',s)
    def test_no_cryptographic_random_reseed_in_observer(self):
        s=(ROOT/'app/mai_measurement.py').read_text();self.assertNotIn('random.seed',s);self.assertNotIn('np.random',s)

@unittest.skipUnless(HAS_FLOWER,'Native Flower measurement contracts required on target host')
class NativeMeasurementTests(unittest.TestCase):
    def test_record_counter_matches_flower_rule(self):
        from flwr.common import RecordDict,ConfigRecord,MetricRecord
        c=RecordDict({'config':ConfigRecord({'x':b'abc','words':'xyz','seq':[b'1',b'22']}),'m':MetricRecord({'n':2})})
        self.assertEqual(mm.record_bytes(c)['total_bytes'],sum(r.count_bytes() for r in c.values()))
    def test_real_array_and_config_count(self):
        import numpy as np
        from flwr.common import RecordDict,ConfigRecord,ndarrays_to_parameters
        from flwr.common.recorddict_compat import parameters_to_arrayrecord
        c=RecordDict({'a':parameters_to_arrayrecord(ndarrays_to_parameters([np.arange(10,dtype=np.float32)]),True),'c':ConfigRecord({'x':b'123'})})
        self.assertEqual(mm.record_bytes(c)['total_bytes'],sum(r.count_bytes() for r in c.values()))
    def test_real_client_query_outer_counter(self):
        from flwr.common import Message,RecordDict,ConfigRecord,Context
        from ehr_secagg.client_app import app
        from mai_runtime_identity import ID_RECORD,EXPECTED_BUILD_ID,QUERY_TYPE
        with tempfile.TemporaryDirectory() as td:
            m=Message(content=RecordDict({ID_RECORD:ConfigRecord({'build_id':EXPECTED_BUILD_ID})}),dst_node_id=11,message_type=QUERY_TYPE,ttl=60)
            m=received_message(m)
            ctx=Context(run_id=1,node_id=11,node_config={'client-id':'C1'},state=RecordDict(),run_config={'runtime-root':td,'condition':'reference'})
            r=app(m,ctx);log=json.loads(next(Path(td).rglob('*.json')).read_text())
            self.assertEqual(log['outgoing_record_bytes']['total_bytes'],sum(x.count_bytes() for x in r.content.values()))
    def test_real_workflow_wrapper_round_and_exception(self):
        from flwr.common import RecordDict,ConfigRecord
        from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD,Key
        ctx=types.SimpleNamespace(state=RecordDict({MAIN_CONFIGS_RECORD:ConfigRecord({Key.CURRENT_ROUND:1})}))
        mm.reset_server();calls=[]
        mm.measured_workflow(lambda g,c:calls.append('fit'),'fit_workflow')(None,ctx)
        mm.measured_workflow(lambda g,c:calls.append('eval'),'evaluation_workflow')(None,ctx)
        self.assertEqual(calls,['fit','eval']);self.assertEqual([e['name'] for e in mm._SERVER_EVENTS],['fit_workflow','evaluation_workflow','outer_round_including_evaluation'])
    def test_native_public_secure_protocol_with_outer_counter(self):
        from flwr.client import ClientApp
        from prg_validation.public_protocol import run_case
        with tempfile.TemporaryDirectory() as td:
            class ObservedToyClientApp(ClientApp):
                def __call__(self,msg,ctx):
                    msg=received_message(msg)
                    ctx.node_config['client-id']='C'+str(int(ctx.node_config['toy-index'])+1)
                    ctx.run_config.update({'runtime-root':td,'condition':'reference'})
                    return mm.timing_boundary_mod(msg,ctx,super().__call__)
            with patch('flwr.client.ClientApp',ObservedToyClientApp):result=run_case([256,256,256],mode='candidate')
            logs=[json.loads(p.read_text()) for p in Path(td).rglob('*.json')]
            self.assertEqual(result['status'],'PASS');self.assertEqual(len(logs),12)
            self.assertTrue(all(x['outgoing_record_bytes']['array_bytes']==0 for x in logs))
            self.assertTrue(all(x['record_values_written'] is False for x in logs))
    def test_native_api_workflow_parameters(self):
        import inspect
        from flwr.server.workflow import DefaultWorkflow
        self.assertIn('fit_workflow',inspect.signature(DefaultWorkflow).parameters);self.assertIn('evaluate_workflow',inspect.signature(DefaultWorkflow).parameters)

if __name__=='__main__':unittest.main()
