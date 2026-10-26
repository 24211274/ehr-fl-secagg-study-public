"""Campaign/error/result integration on fabricated values; never actual final tests."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from main_study.common import ROOT,CLIENTS,SEEDS,write,sha,file_record,require_authorisation,package_integrity
from main_study.evaluation import calibrate_validation,score_endpoints,complete_analysis
from main_study.costs import summarise_cost_cells,campaign_summary
from main_study.campaign import transient_type_eligible
from main_study.final_evaluator import evaluate,validate_models


def cell():
    val=[{'session_id':f'{c}{i}','client':c,'partition':'validation','loss':float(i)} for c in CLIENTS for i in range(10)]
    cal=calibrate_validation(val,{x['session_id'] for x in val})
    rows=[{'session_id':f'{c}t{i}','client':c,'partition':'test_held_out','label':i%2,'loss':float(i),'scenario':'fixture'} for c in CLIENTS for i in range(10)]
    return score_endpoints(rows,cal)


def cost_rows():
    return [{'configuration':m,'condition':c,'seed':s,'instrumented_fit_seconds':2. if m=='C' else 1.,
             'evaluation_seconds':.1,'application_fit_record_bytes':None if m=='A' else (150 if m=='C' else 100),
             'inference_single_median_ms':1.,'inference_single_p95_ms':2.,'inference_batch_sessions_per_second':1000.}
            for m in 'ABC' for c in ('reference','noniid') for s in SEEDS]


class ResultIntegrationTests(unittest.TestCase):
    def test_all_60_fixture_cells_and_zero_spread(self):
        metric=cell();rows=[{'configuration':m,'condition':c,'seed':s,'test':p,'metrics':metric}
          for m in 'ABC' for c in ('reference','noniid') for s in SEEDS for p in ('test_held_out','test_known')]
        x=complete_analysis(rows);self.assertEqual(x['status'],'COMPLETE');self.assertEqual(len(x['analysis']),4)
        y=x['analysis']['reference/test_held_out']['global_AP']['C_minus_B']
        self.assertEqual(y['values'],[0.]*5);self.assertIsNone(y['interval_95'])
    def test_substitute_seed_rejected(self):
        with self.assertRaises(ValueError):complete_analysis([{'configuration':'B','condition':'reference','seed':123,'test':'test_known','metrics':cell()}])
    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):complete_analysis([{'configuration':'D','condition':'reference','seed':11,'test':'test_known','metrics':cell()}])
    def test_missing_cells_not_complete(self):self.assertEqual(complete_analysis([])['status'],'INCOMPLETE')
    def test_duplicate_result_rejected(self):
        row={'configuration':'B','condition':'reference','seed':11,'test':'test_known','metrics':cell()}
        with self.assertRaises(ValueError):complete_analysis([row,row])
    def test_main_cost_ratios_and_missing_A_volume(self):
        x=summarise_cost_cells(cost_rows());v=x['summary']['reference']['application_fit_record_bytes']
        self.assertEqual(v['C_vs_B_percent_per_seed']['values'],[50.]*5);self.assertIsNone(v['by_method']['A']['mean'])
    def test_cost_duplicate_rejected(self):
        r=cost_rows();r[-1]=r[0]
        with self.assertRaises(ValueError):summarise_cost_cells(r)
    def test_cost_missing_rejected(self):
        with self.assertRaises(ValueError):summarise_cost_cells(cost_rows()[:-1])
    def test_main_cost_file_adapter(self):
        # Exercise the exact JSON field names written by each main worker.
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);models=[]
            for i,row in enumerate(cost_rows()):
                j=root/str(i);d=j/'reports'
                write(d/'selected_model_inference.json',{'median_ms':1.,'p95_ms':2.,'throughput_sessions_per_second':1000.})
                if row['configuration']=='A':write(d/'central_summary.json',{'fit_wall_ns_total':10**9,'evaluation_wall_seconds_total':.1,'elapsed_seconds':2.})
                else:write(d/'main_cost_summary.json',{'totals_all_rounds':{'fit_workflow_wall_ns':int(row['instrumented_fit_seconds']*1e9),'evaluation_workflow_wall_ns':10**8,'fit_record_bytes_total':row['application_fit_record_bytes']},'host':{'elapsed_seconds':2.}})
                models.append({**row,'job':str(i)})
            self.assertEqual(campaign_summary(root,models)['status'],'PASS')


class WorkflowBoundaryTests(unittest.TestCase):
    def test_final_input_never_called_before_authorisation(self):
        with tempfile.TemporaryDirectory() as t,patch('main_study.final_evaluator.verify_campaign'),patch('main_study.final_evaluator.validate_models',return_value=[]),patch('main_study.final_evaluator.sha',return_value='a'*64),patch('main_study.final_evaluator.require_authorisation',side_effect=PermissionError('no test release')),patch('main_study.final_evaluator.load_final_inputs') as load:
            with self.assertRaises(PermissionError):evaluate(Path(t),Path(t)/'no',Path(t)/'nonexistent')
            load.assert_not_called();self.assertFalse((Path(t)/'final_evaluation').exists())
    def test_existing_final_evaluation_refused_without_data_read(self):
        with tempfile.TemporaryDirectory() as t,patch('main_study.final_evaluator.verify_campaign'),patch('main_study.final_evaluator.validate_models',return_value=[]),patch('main_study.final_evaluator.sha',return_value='a'*64),patch('main_study.final_evaluator.require_authorisation'),patch('main_study.final_evaluator.load_final_inputs') as load:
            p=Path(t);(p/'final_evaluation').mkdir()
            with self.assertRaises(FileExistsError):evaluate(p,p/'no',p/'no')
            load.assert_not_called()
    def test_only_explicit_test_selection_hash(self):
        with tempfile.TemporaryDirectory() as t,patch('main_study.common.package_integrity',return_value='b'*64):
            p=Path(t);write(p/'campaign_manifest.json',{'x':1});a=p/'a.json'
            write(a,{'approved':True,'action':'EVALUATE_FINAL_TESTS','package_manifest_sha256':'b'*64,'campaign_manifest_sha256':sha(p/'campaign_manifest.json'),'selected_models_manifest_sha256':'c'*64})
            with self.assertRaises(PermissionError):require_authorisation(a,'EVALUATE_FINAL_TESTS',p,expected_models_sha='d'*64)
    def test_code_failure_not_retryable(self):self.assertFalse(transient_type_eligible({'error_type':'RuntimeError','root_error_type':'ValueError'}))
    def test_assertion_failure_not_retryable(self):self.assertFalse(transient_type_eligible({'root_error_type':'AssertionError'}))
    def test_permission_failure_not_retryable(self):self.assertFalse(transient_type_eligible({'error_type':'PermissionError'}))
    def test_unknown_wrapper_failure_not_retryable(self):self.assertFalse(transient_type_eligible({'error_type':'RuntimeError'}))
    def test_timeout_only_type_eligible_not_authorisation(self):self.assertTrue(transient_type_eligible({'root_error_type':'TimeoutError'}))
    def test_network_type_eligible(self):self.assertTrue(transient_type_eligible({'error_type':'ConnectionResetError'}))
    def test_tampered_source_not_hidden_by_unlisted_file(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a.py').write_text('x=1\n');(p/'PACKAGE_SHA256SUMS.txt').write_text(sha(p/'a.py')+'  a.py\n')
            with patch('main_study.common.ROOT',p):
                self.assertEqual(package_integrity(),sha(p/'PACKAGE_SHA256SUMS.txt'))
                (p/'extra.py').write_text('bad=1')
                with self.assertRaises(ValueError):package_integrity()
    def test_generated_report_does_not_change_source(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a.py').write_text('x=1\n');(p/'PACKAGE_SHA256SUMS.txt').write_text(sha(p/'a.py')+'  a.py\n');(p/'Main_Preparation_Report_fixture.zip').write_bytes(b'report')
            with patch('main_study.common.ROOT',p):self.assertEqual(package_integrity(),sha(p/'PACKAGE_SHA256SUMS.txt'))
    def test_selected_manifests_require_all_30_before_paths(self):
        with tempfile.TemporaryDirectory() as t,patch('main_study.final_evaluator.package_integrity',return_value='b'*64):
            p=Path(t);write(p/'campaign_manifest.json',{'x':1});write(p/'selected_models_manifest.json',{'package_manifest_sha256':'b'*64,'campaign_manifest_sha256':sha(p/'campaign_manifest.json'),'models':[]})
            with self.assertRaises(ValueError):validate_models(p)
    def test_prepare_has_no_final_evaluator_call(self):
        import inspect
        from main_study.campaign import prepare
        s=inspect.getsource(prepare);self.assertNotIn('load_final_inputs',s);self.assertNotIn('BenchmarkSource',s)
    def test_no_automatic_next_block_loop(self):
        import inspect
        from main_study.campaign import train_next
        s=inspect.getsource(train_next);self.assertEqual(s.count('execute_block('),1)
    def test_no_install_in_main_shell(self):
        text=(ROOT/'scripts/run_on_wsl.sh').read_text();self.assertNotIn('pip install',text);self.assertNotIn('apt ',text)
