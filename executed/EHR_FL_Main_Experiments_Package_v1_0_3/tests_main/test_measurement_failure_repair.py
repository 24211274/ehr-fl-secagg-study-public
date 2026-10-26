"""Trusted-host failure evidence and immutable parent reuse; fabricated files only."""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from main_study.common import read,write,sha,ROOT
from main_study.failure_evidence import collect_failure_capsule
from main_study.repair import verify_repair_parent,copy_parent_frozen
from fullsize_support.integrity import package_report
from test_code_repair import parent_fixture
from test_attempt_staging import snapshot

class FailureCapsuleTests(unittest.TestCase):
    def setup_case(self,t,message='Message outside fixed timing rounds/phases'):
        j=Path(t)/'job';write(j/'job.json',{'configuration':'B','condition':'reference'})
        write(j/'server_outputs/reference/server_result.json',{'status':'FAIL','exception':'ValueError','message':message,
               'unexpected_sensitive_field':'must-not-export'})
        (j/'error_reference_LOCAL_ONLY.txt').write_text('Traceback (most recent call last):\n  File "/safe/app/mai_measurement.py", line 96, in descriptor\n    raise ValueError(secret_key)\nValueError: secret_key=do-not-export\n')
        write(j/'server_outputs/reference/history.json',{'history':[{'round':r,'loss':'not-to-export'} for r in range(7)]})
        (j/'checkpoints').mkdir();(j/'checkpoints/private.pt').write_bytes(b'private')
        return j
    def test_safe_summary_and_frames_no_values(self):
        with tempfile.TemporaryDirectory() as t:
            job=self.setup_case(t);before=snapshot(job);out=Path(t)/'report'
            r=collect_failure_capsule(job,out);raw=json.dumps(r)
            self.assertEqual(r['server_exception'],'ValueError');self.assertEqual(r['last_recorded_evaluation_round'],6)
            self.assertEqual(r['frames'][0]['function'],'descriptor')
            self.assertNotIn('secret_key',raw);self.assertNotIn('not-to-export',raw)
            self.assertNotIn('must-not-export',raw);self.assertEqual(snapshot(job),before)
            self.assertEqual([p.name for p in out.iterdir()],['server_failure_capsule.json'])
    def test_unknown_exception_message_redacted(self):
        with tempfile.TemporaryDirectory() as t:
            r=collect_failure_capsule(self.setup_case(t,'secret arbitrary bytes'),Path(t)/'report')
            self.assertNotIn('secret arbitrary bytes',json.dumps(r));self.assertIn('redacted',r['message'])
    def test_structured_scope_error_retains_numeric_cause(self):
        with tempfile.TemporaryDirectory() as t:
            m='Message outside frozen measurement scope: round=31; phase=fit; max_round=30'
            r=collect_failure_capsule(self.setup_case(t,m),Path(t)/'report');self.assertEqual(r['message'],m)
    def test_no_server_result_remains_diagnostic_not_success(self):
        with tempfile.TemporaryDirectory() as t:
            j=Path(t)/'job';write(j/'job.json',{'configuration':'A','condition':'noniid'})
            r=collect_failure_capsule(j,Path(t)/'report');self.assertFalse(r['server_failure_found']);self.assertEqual(r['status'],'DIAGNOSTIC_ONLY')
    def test_failure_capsule_source_cannot_authorise_training(self):
        text=(ROOT/'main_study/failure_evidence.py').read_text()
        self.assertNotIn('approved',text);self.assertNotIn('torch.load',text)


def measurement_parent(root,last=6):
    parent,policy=parent_fixture(root)
    report=parent/'attempts/MAIN-11-reference/attempt_1/reports'
    status=read(report/'block_status.json');status['root_error_type']='RuntimeError';write(report/'block_status.json',status)
    (report/'report_files.json').unlink();package_report(report,parent/'new_fixture_report.zip')
    entry={k:v for k,v in status.items() if k not in ('test_data_opened','privacy_claim_validated')}
    write(parent/'progress.json',{'completed_blocks':[],'attempts':{'MAIN-11-reference':entry},'main_training_started':True})
    fail={'status':'FAIL','configuration':'B','condition':'reference','exception':'ValueError','message':'Message outside fixed timing rounds/phases'}
    b=parent/'attempts/MAIN-11-reference/attempt_1/B'
    write(b/'server_outputs/reference/server_result.json',fail)
    write(b/'server_outputs/reference/history.json',{'history':[{'round':r} for r in range(last+1)]})
    (b/'error_reference_LOCAL_ONLY.txt').write_text('File "/test/mai_measurement.py", line 96, in descriptor\nValueError: Message outside fixed timing rounds/phases\n')
    policy.update(expected_wrapper_error_type='RuntimeError',expected_server_failure=fail,
        failure_report_manifest_sha256=sha(report/'report_files.json'),failure_block_status_sha256=sha(report/'block_status.json'))
    return parent,policy,b

class MeasurementRepairBoundaryTests(unittest.TestCase):
    def case(self,alter=None,last=6):
        with tempfile.TemporaryDirectory() as t:
            p,policy,b=measurement_parent(t,last);n=Path(t)/'successor'
            if alter=='message':write(b/'server_outputs/reference/server_result.json',{'status':'FAIL','exception':'ValueError','message':'different'})
            if alter=='missing_trace':(b/'error_reference_LOCAL_ONLY.txt').write_text('Other failure')
            if alter=='completed_B':write(b/'reports/cell_complete.json',{'status':'PASS'})
            if alter=='completed_C':write(p/'attempts/MAIN-11-reference/attempt_1/C/reports/cell_complete.json',{'status':'PASS'})
            if alter=='rounds':write(b/'server_outputs/reference/history.json',{'history':[{'round':0},{'round':6}]})
            before=snapshot(p)
            with patch('main_study.repair.repair_policy',return_value=policy):
                if alter:
                    with self.assertRaises(ValueError):verify_repair_parent(p,n)
                else:
                    v=verify_repair_parent(p,n);self.assertEqual(v['archived_partial_B']['last_recorded_evaluation_round'],last)
                    result=copy_parent_frozen(p,n,v);self.assertTrue(result['all_sha256_identical'])
                    self.assertFalse((n/'attempts').exists())
                self.assertEqual(before,snapshot(p))
    def test_read_only_review_partial_B_round_six(self):self.case()
    def test_read_only_review_partial_B_round_five(self):self.case(last=5)
    def test_different_internal_failure_refused(self):self.case('message')
    def test_missing_trace_refused(self):self.case('missing_trace')
    def test_completed_B_refused(self):self.case('completed_B')
    def test_completed_C_refused(self):self.case('completed_C')
    def test_noncontiguous_rounds_refused(self):self.case('rounds')
    def test_code_fault_is_not_transient_retry(self):
        from main_study.campaign import transient_type_eligible
        self.assertFalse(transient_type_eligible({'root_error_type':'ValueError'}))
    def test_native_scope_probe_precedes_benchmark_callback(self):
        s=(ROOT/'scripts/job_worker.py').read_text()
        self.assertLess(s.index('check_measurement_horizon(job,mode,co)'),s.index('for c in CLIENTS:load_client'))
    def test_both_sides_use_same_scoped_observer(self):
        for mode in ('ehr_fedavg','ehr_secagg'):
            text=(ROOT/'app'/mode/'server_app.py').read_text();self.assertIn('ObservedRawGrid(grid)',text)
            text=(ROOT/'app'/mode/'client_app.py').read_text();self.assertIn('timing_boundary_mod',text)
