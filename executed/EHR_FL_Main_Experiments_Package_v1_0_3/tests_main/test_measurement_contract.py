"""Regression of the actual observer, all rounds, both conditions and B/C envelopes.

Local doubles are expressly not Flower Deployment evidence. Native subprocess
cases below use installed Flower, not a substitute; skip only where absent.
"""
from __future__ import annotations
import copy, importlib.util, json, subprocess, sys, tempfile, types, unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from main_study.common import ROOT, write
from main_study.design import cell_config,bind_app
import mai_measurement as mm
from test_measurement import Content, Sized, Msg
from timing_support.analysis import reconcile_boundary
HAS_FLOWER=importlib.util.find_spec('flwr') is not None

@contextmanager
def job_scope(co='reference', rounds=30, stage='MAIN_CONTROLLED_COMPARISON'):
    conf={'rounds':rounds,'conditions':[co],'stage':stage}
    with patch('ehr_fedavg.resources.read_document',return_value=conf):
        yield

def req(r,phase='fit',secure=None,ident='probe',node=11):
    if phase=='handshake':return Msg(Content(configs={'q':Sized(1)}),kind='query.measurement_probe',ident=ident,node=node)
    if phase=='fit':
        records={'secaggplus_configs':Sized(12,stage=secure)} if secure else {'fitins.config':Sized(16,server_round=r)}
        return Msg(Content(configs=records),group=str(r),ident=ident,node=node)
    return Msg(Content(configs={'evaluateins.config':Sized(22,server_round=r,phase=phase)}),kind='evaluate',ident=ident,node=node)


def full_fake_run(co,mode, *, drop_last=False):
    with tempfile.TemporaryDirectory() as td,job_scope(co):
        calls=[]
        def inner(messages,timeout=None):
            replies=[]
            for msg in messages:
                cid={11:'C1',12:'C2',13:'C3'}[msg.metadata.dst_node_id]
                ctx=types.SimpleNamespace(run_config={'runtime-root':td,'condition':co,'num-server-rounds':30},node_config={'client-id':cid})
                def callback(m,c):
                    d=mm.descriptor(m);calls.append(d)
                    if d['stage'] in ('fit','collect_masked_vectors'):
                        with mm.client_span('local_fit_with_checks'):pass
                        with mm.client_span('data_load_with_checks'):pass
                    return Msg(Content(configs={'ack':Sized(1)}),ident=m.metadata.message_id,node=m.metadata.dst_node_id)
                replies.append(mm.timing_boundary_mod(msg,ctx,callback))
            return replies
        grid=mm.ObservedRawGrid(types.SimpleNamespace(send_and_receive=inner));seq=0
        def send(r,phase,stage=None):
            nonlocal seq
            seq+=1
            grid.send_and_receive([req(r,phase,stage,ident=f'p{seq}-{n}',node=n) for n in (11,12,13)])
        send(0,'handshake');send(0,'handshake');send(0,'initial')
        for r in range(1,31):
            if drop_last and r==30:break
            for st in (('fit',) if mode=='B' else ('setup','share_keys','collect_masked_vectors','unmask')):
                send(r,'fit',None if mode=='B' else st)
            send(r,'round')
        send(30,'selected')
        clients=[json.loads(p.read_text()) for p in Path(td).rglob('*.json')]
        result=reconcile_boundary({'exchanges':grid.rows},clients,mode,rounds=30)
        assert all(r['condition']==co and r['measurement_scope']['rounds']==30 for r in clients)
        return result,calls

class MainMeasurementScopeTests(unittest.TestCase):
    def test_legacy_six_round_pilot_still_rejects_seven(self):
        with self.assertRaises(ValueError):mm.descriptor(req(7))
    def test_main_accepts_round_six_seven_and_thirty(self):
        with job_scope():
            for r in (1,5,6,7,29,30):self.assertEqual(mm.descriptor(req(r))['round'],r)
    def test_main_rejects_round_31(self):
        with job_scope(),self.assertRaisesRegex(ValueError,'max_round=30'):mm.descriptor(req(31))
    def test_secure_all_stages_all_main_rounds(self):
        with job_scope('noniid'):
            for st in ('setup','share_keys','collect_masked_vectors','unmask'):
                for r in range(1,31):self.assertEqual(mm.descriptor(req(r,secure=st))['stage'],st)
    def test_initial_and_each_selected_checkpoint_round(self):
        with job_scope():
            self.assertEqual(mm.descriptor(req(0,'initial'))['round'],0)
            for r in range(1,31):self.assertEqual(mm.descriptor(req(r,'selected'))['round'],r)
    def test_initial_cannot_be_nonzero(self):
        with job_scope(),self.assertRaises(ValueError):mm.descriptor(req(1,'initial'))
    def test_fit_cannot_be_zero(self):
        with job_scope(),self.assertRaises(ValueError):mm.descriptor(req(0))
    def test_selected_cannot_be_zero(self):
        with job_scope(),self.assertRaises(ValueError):mm.descriptor(req(0,'selected'))
    def test_unknown_phase_refused(self):
        with job_scope(),self.assertRaises(ValueError):mm.descriptor(req(2,'secret_dump'))
    def test_float_and_boolean_rounds_refused(self):
        with job_scope():
            for x in (True,1.5,None,'1.0','-1'):
                with self.assertRaises(ValueError):mm.descriptor(req(x))
    def test_main_budget_cannot_be_silently_shortened(self):
        with job_scope(rounds=6),self.assertRaises(ValueError):mm.descriptor(req(1))
    def test_invalid_round_budgets(self):
        for r in (0,31,True,30.,None):
            with job_scope(rounds=r),self.assertRaises(ValueError):mm.measurement_scope()
    def test_invalid_condition_set(self):
        for co in ('unknown','test_known','test_held_out'):
            with job_scope(co),self.assertRaises(ValueError):mm.measurement_scope()
    def test_mutation_of_returned_scope_does_not_modify_bundled_config(self):
        first=mm.measurement_scope();first['rounds']=300
        self.assertEqual(mm.measurement_scope()['rounds'],6)
    def test_bound_wrong_condition_refused_before_callback(self):
        with tempfile.TemporaryDirectory() as td,job_scope('noniid'):
            calls=[];ctx=types.SimpleNamespace(run_config={'runtime-root':td,'condition':'reference','num-server-rounds':30},node_config={'client-id':'C1'})
            with self.assertRaises(ValueError):mm.timing_boundary_mod(req(7),ctx,lambda *a:calls.append(1))
            self.assertEqual(calls,[]);self.assertEqual(list(Path(td).rglob('*.json')),[])
    def test_context_round_budget_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as td,job_scope('noniid'):
            ctx=types.SimpleNamespace(run_config={'runtime-root':td,'condition':'noniid','num-server-rounds':6},node_config={'client-id':'C1'})
            with self.assertRaises(ValueError):mm.timing_boundary_mod(req(7),ctx,lambda *a:None)
    def test_missing_instruction_refused_before_transport(self):
        calls=[]
        with job_scope():
            grid=mm.ObservedRawGrid(types.SimpleNamespace(send_and_receive=lambda *a,**k:calls.append(1)))
            with self.assertRaises(ValueError):grid.send_and_receive([Msg(Content())])
        self.assertEqual(calls,[])
    def test_B_reference_full_horizon_and_counters(self):
        result,calls=full_fake_run('reference','B');self.assertEqual(result['matched_messages'],192)
    def test_B_noniid_full_horizon_and_counters(self):
        result,calls=full_fake_run('noniid','B');self.assertEqual(result['matched_messages'],192)
    def test_C_reference_full_horizon_and_counters(self):
        result,calls=full_fake_run('reference','C');self.assertEqual(result['matched_messages'],462)
    def test_C_noniid_full_horizon_and_counters(self):
        result,calls=full_fake_run('noniid','C');self.assertEqual(result['matched_messages'],462)
    def test_missing_final_round_not_hidden_by_warmup_or_partial_summary(self):
        with self.assertRaises(ValueError):full_fake_run('reference','B',drop_last=True)
    def test_missing_secure_final_round_refused(self):
        with self.assertRaises(ValueError):full_fake_run('noniid','C',drop_last=True)
    def test_main_cost_code_keeps_every_round(self):
        text=(ROOT/'main_study/costs.py').read_text()
        self.assertIn("'warmup_rounds_excluded':[]",text);self.assertIn('range(1,rounds+1)',text)

@unittest.skipUnless(HAS_FLOWER,'Native Flower envelope check requires the pinned installed library')
class NativeMainMeasurementHorizonTests(unittest.TestCase):
    def case(self,co,mode):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            job=Path(td)/'job';app=job/'apps'/co
            shutil.copytree(ROOT/'app',app,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            bind_app(app,cell_config(11,co,30),configuration=mode)
            code='''import json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
from main_study.measurement_gate import check_measurement_horizon
v=check_measurement_horizon(Path(sys.argv[3]),sys.argv[4],sys.argv[5])
Path(sys.argv[6]).write_text(json.dumps(v,allow_nan=False))
'''
            report=job/'gate.json'
            p=subprocess.run([sys.executable,'-I','-B','-c',code,str(app),str(ROOT),str(job),mode,co,str(report)],capture_output=True,text=True,timeout=120)
            self.assertEqual(p.returncode,0,(p.stdout+p.stderr)[-7000:])
            r=json.loads(report.read_text());self.assertEqual(r['status'],'PASS')
            self.assertEqual(r['rounds_exercised'],list(range(1,31)))
            self.assertFalse(r['actual_training_performed']);self.assertFalse(r['deployment_engine_used'])
            self.assertEqual(len(r['negative_cases_rejected']),6)
    def test_native_B_reference(self):self.case('reference','B')
    def test_native_B_noniid(self):self.case('noniid','B')
    def test_native_C_reference(self):self.case('reference','C')
    def test_native_C_noniid(self):self.case('noniid','C')
