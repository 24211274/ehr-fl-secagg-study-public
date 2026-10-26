"""Main-stage regressions on fabricated data only; no final benchmark test rows."""
from __future__ import annotations
import ast,copy,hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from main_study.common import ROOT,CLIENTS,SEEDS,read,read_csv,write,sha,safe_child,verify_rows,file_record,require_authorisation
from main_study.design import plan,ordered_blocks,make_draws,pooled_round,cell_config,bind_app,decisions
from main_study.evaluation import (binary_metrics,calibrate_validation,join_scores_labels,score_endpoints,
                                   paired_summary,complete_analysis)
from main_study.test_inputs import encode_rows
from central_smoke.sampling import choose_checkpoint
from pilot_core.model import CompactLSTMAutoencoder,set_training_seed,state_digest


def validation():
    return [{'session_id':f'{c}-{i}','client':c,'partition':'validation','loss':float(i)} for c in CLIENTS for i in range(10)]


def depts():
    names={'C1':'emergency','C2':'internal_medicine','C3':'surgery'}
    return [{'client':c,'session_id':f'{c}-{i}','hospital_id':c+'h','department':names[c] if i<5 else 'other'} for c in CLIENTS for i in range(10)]


class MainPlanTests(unittest.TestCase):
    def test_30_exact_jobs(self):self.assertEqual(len(plan()),30)
    def test_10_blocks(self):self.assertEqual(len(ordered_blocks()),10)
    def test_each_block_has_three_methods(self):
        for b in ordered_blocks():self.assertEqual({r['configuration'] for r in plan() if r['block_id']==b},set('ABC'))
    def test_seeds(self):self.assertEqual({int(r['training_seed']) for r in plan()},set(SEEDS))
    def test_fixed_budget(self):self.assertTrue(all(int(r['epochs_or_rounds'])==30 for r in plan()))
    def test_preparation_authority_only(self):
        d=decisions();self.assertTrue(d['authorised_to_prepare_main_runner']);self.assertFalse(d['authorised_to_launch_main']);self.assertFalse(d['authorised_to_open_test_data'])
    def test_no_new_tuning(self):self.assertEqual(decisions()['adopted_closures']['P1_tuning']['additional_hyperparameter_search_budget'],0)
    def test_publication_not_invented(self):self.assertIsNone(decisions()['data']['public_release']['kaggle_url'])
    def test_replay(self):self.assertEqual(make_draws(11,'reference',3,depts()),make_draws(11,'reference',3,depts()))
    def test_reference_permutation(self):
        x=make_draws(11,'reference',30,depts())
        for c in CLIENTS:
            for r in range(1,31):self.assertEqual(len({a['session_id'] for a in x if a['client']==c and a['outer_epoch']==r}),10)
    def test_noniid_with_replacement(self):
        x=make_draws(11,'noniid',30,depts());self.assertEqual(len(x),900)
        self.assertTrue(any(len({r['session_id'] for r in x if r['client']==c and r['outer_epoch']==1})<10 for c in CLIENTS))
    def test_pool_preserves_occurrences(self):
        from collections import Counter
        x=make_draws(11,'noniid',1,depts());y=pooled_round(x,11,'noniid',1)
        key=lambda a:(a['client'],a['client_draw_position'],a['session_id'])
        self.assertEqual(Counter(map(key,x)),Counter(map(key,y)))
    def test_seed_changes_schedules(self):self.assertNotEqual(make_draws(11,'reference',1,depts()),make_draws(29,'reference',1,depts()))
    def test_reject_test_condition(self):
        with self.assertRaises(ValueError):make_draws(11,'test_known',30,depts())
    def test_full_geometry(self):
        c=cell_config(11,'reference',30);self.assertEqual(sum(c['training_sessions_per_client'].values()),10833)
        self.assertEqual(sum(c['local_updates_per_round'].values()),171);self.assertEqual(math.ceil(10833/64),170)
    def test_budget_limits_not_early_stopping(self):
        c=cell_config(11,'reference',30);self.assertEqual(c['rounds'],30);self.assertEqual(c['exchange_timeout_seconds'],600)
    def test_native_project_constant_is_bound_explicitly(self):
        s=(ROOT/'main_study/design.py').read_text();self.assertIn('num-server-rounds = 6',s);self.assertIn("conf['rounds']",s)
    def test_resource_bytes_for_every_seed_condition(self):
        from scripts.generate_bundled_resources import render
        with tempfile.TemporaryDirectory() as t:
            a=Path(t);(a/'ehr_fedavg').mkdir();(a/'ehr_secagg').mkdir()
            (a/'ehr_fedavg/encoding_schema.json').write_bytes((ROOT/'config/encoding_schema.json').read_bytes())
            for seed in SEEDS:
                for co in ('reference','noniid'):
                    conf=cell_config(seed,co,30);conf['schedule_sha256']='d'*64;ch=bind_app(a,conf,configuration='C')
                    ns={};exec((a/'ehr_fedavg/_embedded_resources.py').read_text(),ns)
                    d=json.loads(ns['RESOURCE_BYTES']['fixed_config.json']);self.assertEqual(d['training_seed'],seed);self.assertEqual(d['rounds'],30)
                    ns2={};exec((a/'ehr_secagg/settings.py').read_text(),ns2);self.assertEqual(ns2['scfg']()['base_fedavg_config_sha256'],ns['RESOURCE_SHA256']['fixed_config.json'])
                    self.assertEqual(ch,ns2['SHA256'])


class CalibrationAndMetricTests(unittest.TestCase):
    def test_perfect_ranking(self):self.assertEqual(binary_metrics([0,0,1,1],[0,1,2,3],[0,0,1,1])['AP'],1.)
    def test_tied_AP_prevalence(self):self.assertAlmostEqual(binary_metrics([0,0,1,1],[1,1,1,1],[0,0,0,0])['AP'],.5)
    def test_all_negatives_undefined(self):
        v=binary_metrics([0,0],[.1,.2],[0,0]);self.assertIsNone(v['AP']);self.assertIsNone(v['AUROC']);self.assertIsNone(v['F1'])
    def test_all_positives_undefined(self):self.assertIsNone(binary_metrics([1,1],[.1,.2],[0,1])['AP'])
    def test_no_predictions_with_threats(self):
        v=binary_metrics([0,1],[.1,.2],[0,0]);self.assertIsNone(v['precision']);self.assertEqual(v['F1'],0.)
    def test_confusion_counts(self):
        v=binary_metrics([0,0,1,1],[.1,.2,.3,.4],[0,1,0,1]);self.assertEqual([v[k] for k in ['TN','FP','FN','TP']],[1]*4);self.assertEqual(v['false_alerts_per_1000_legitimate'],500.)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):binary_metrics([0,1],[0,float('nan')],[0,0])
    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):binary_metrics([0,2],[0,1],[0,0])
    def test_validation_rule(self):
        r=validation();c=calibrate_validation(r,{x['session_id'] for x in r});self.assertEqual(c['clients']['C1']['median'],4.5)
        self.assertAlmostEqual(c['clients']['C1']['thresholds']['0.99'],8.91)
    def test_unit_scale_fallback(self):
        r=validation()
        for x in r:x['loss']=1.
        c=calibrate_validation(r,{x['session_id'] for x in r})
        self.assertTrue(all(x['scale_method']=='unit_fallback' and x['scale']==1. for x in c['clients'].values()))
    def test_IQR_fallback(self):
        r=[]
        for c in CLIENTS:
            for i,x in enumerate([0.,1.,1.,1.,1.,1.,2.]):r.append({'client':c,'partition':'validation','session_id':c+str(i),'loss':x})
        # At least 50% same => both MAD and sometimes IQR zero. Use asymmetric mass for positive IQR.
        r=[]
        for c in CLIENTS:
            for i,x in enumerate([0.,0.,1.,1.,1.]):r.append({'client':c,'partition':'validation','session_id':c+str(i),'loss':x})
        out=calibrate_validation(r,{x['session_id'] for x in r});self.assertEqual(out['clients']['C1']['scale_method'],'IQR')
    def test_test_calibration_rejected(self):
        r=validation();r[0]['partition']='test_known'
        with self.assertRaises(ValueError):calibrate_validation(r,{x['session_id'] for x in r})
    def test_duplicate_validation_rejected(self):
        r=validation();ids={x['session_id'] for x in r};r[-1]=dict(r[0])
        with self.assertRaises(ValueError):calibrate_validation(r,ids)
    def test_strict_greater_threshold(self):
        v=validation();cal=calibrate_validation(v,{x['session_id'] for x in v})
        rows=[{'session_id':c+str(i),'client':c,'partition':'test_known','loss':cal['clients'][c]['thresholds']['0.99'],'label':i,'scenario':'fixture'} for c in CLIENTS for i in (0,1)]
        s=score_endpoints(rows,cal)['quantiles']['0.99']['global'];self.assertEqual(s['predicted_positives'],0)
    def test_join_missing_rejected(self):
        with self.assertRaises(ValueError):join_scores_labels([],[],'test_known',{'a'})
    def test_join_wrong_client_rejected(self):
        a={'session_id':'a','client':'C1','partition':'test_known','loss':1.};b={'session_id':'a','client':'C2','partition':'test_known','label':1}
        with self.assertRaises(ValueError):join_scores_labels([a],[b],'test_known',{'a'})
    def test_join_wrong_partition_rejected(self):
        a={'session_id':'a','client':'C1','partition':'test_known','loss':1.};b={'session_id':'a','client':'C1','partition':'test_held_out','label':1}
        with self.assertRaises(ValueError):join_scores_labels([a],[b],'test_known',{'a'})
    def test_missing_seed_no_interval(self):
        out=paired_summary({s:1. for s in SEEDS[:-1]},{s:2. for s in SEEDS},interval=True);self.assertIsNone(out['interval_95']);self.assertEqual(out['n'],4)
    def test_zero_SD_no_certainty(self):self.assertIsNone(paired_summary(dict.fromkeys(SEEDS,.2),dict.fromkeys(SEEDS,.3),interval=True)['interval_95'])
    def test_interval_five_paired(self):
        from scipy.stats import t
        a=dict.fromkeys(SEEDS,0.);b=dict(zip(SEEDS,[.1,.2,.3,.4,.5]));o=paired_summary(a,b,interval=True)
        width=float(t.ppf(.975,4))*np.std(list(b.values()),ddof=1)/np.sqrt(5)
        self.assertAlmostEqual(o['interval_95'][0],.3-width)
    def test_seed_substitution_rejected(self):
        with self.assertRaises(ValueError):paired_summary({42:0.},{42:1.},interval=True)
    def test_no_interval_on_secondary_metric(self):self.assertIsNone(paired_summary(dict.fromkeys(SEEDS,0.),dict(zip(SEEDS,range(5))),interval=False)['interval_95'])
    def test_earliest_near_tie_against_global_minimum(self):
        h=[{'epoch':0,'validation_loss':0.},{'epoch':1,'validation_loss':1.+1.5e-12},{'epoch':2,'validation_loss':1.+.8e-12},{'epoch':3,'validation_loss':1.}]
        self.assertEqual(choose_checkpoint(h),2)
    def test_round_zero_not_eligible(self):self.assertEqual(choose_checkpoint([{'epoch':0,'validation_loss':0.},{'epoch':1,'validation_loss':1.}]),1)
    def test_summary_incomplete_not_success(self):self.assertEqual(complete_analysis([])['status'],'INCOMPLETE')


class EncodingAndBoundaryTests(unittest.TestCase):
    def fixture(self):
        schema=read(ROOT/'config/encoding_schema.json');scaler=read(ROOT/'protocol/decisions/evidence/S26_relative_time_scaler.json')
        from ehr_preprocessing.pipeline import CLIENTS as hospitals
        hid=hospitals['C1'];r={'session_id':'fake-session','hospital_id':hid,'event_id':'fake-event','sequence_position':'0','timestamp_utc':'2025-01-01T10:00:00Z'}
        for f in schema['categorical_fields']:r[f['name']]=next(x for x in f['categories'] if not x.startswith('__'))
        for f in schema['binary_fields']:r[f]='False'
        return schema,scaler,hid,r
    def test_encoding_119_and_model_forward(self):
        from ehr_preprocessing.dataset import collate_sessions
        schema,sc,hid,r=self.fixture();s=encode_rows('fake-session','C1',hid,'test_known',[r],schema,sc)
        self.assertEqual(s['x'].shape,(1,119));b=collate_sessions([s]);m=CompactLSTMAutoencoder([len(f['categories']) for f in schema['categorical_fields']])
        m.eval()
        with torch.no_grad():self.assertEqual(m(b['x'],b['lengths'])[1].shape,(1,1,3))
    def test_label_never_a_feature(self):
        schema,sc,hid,r=self.fixture();r['binary_label']='0';a=encode_rows('fake-session','C1',hid,'test_known',[r],schema,sc)
        r['binary_label']='1';r['threat_family']='CHANGED';b=encode_rows('fake-session','C1',hid,'test_known',[r],schema,sc)
        self.assertTrue(torch.equal(a['x'],b['x']))
    def test_unknown_maps_masks(self):
        schema,sc,hid,r=self.fixture();r[schema['categorical_fields'][0]['name']]='never-in-training-fixture'
        s=encode_rows('fake-session','C1',hid,'test_known',[r],schema,sc);self.assertFalse(s['categorical_valid'][0,0])
    def test_duplicate_positions_fail(self):
        schema,sc,hid,r=self.fixture()
        with self.assertRaises(ValueError):encode_rows('fake-session','C1',hid,'test_known',[r,dict(r,event_id='e2')],schema,sc)
    def test_wrong_hospital_fail(self):
        schema,sc,hid,r=self.fixture();r['hospital_id']='foreign'
        with self.assertRaises(ValueError):encode_rows('fake-session','C1',hid,'test_known',[r],schema,sc)
    def test_negative_gap_fail(self):
        schema,sc,hid,r=self.fixture();second=dict(r,event_id='fake2',sequence_position='1',timestamp_utc='2024-01-01T00:00:00Z')
        with self.assertRaises(ValueError):encode_rows('fake-session','C1',hid,'test_known',[r,second],schema,sc)
    def test_no_final_source_in_preparation(self):
        # Architectural check supplements runtime authorisation tests; no raw dataset is read here.
        s=(ROOT/'main_study/campaign.py').read_text();self.assertNotIn('load_final_inputs(',s);self.assertNotIn('BenchmarkSource(',s)
    def test_inference_selection_stays_validation(self):
        rows=read_csv(ROOT/'reference/main_inference_selection.csv');self.assertEqual(len(rows),220)
        self.assertTrue(all(r['partition']=='validation' for r in rows));self.assertEqual(len({r['session_id'] for r in rows}),220)
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):safe_child(Path(t),'../outside')
    def test_altered_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a';p.write_text('x');r=file_record(p,Path(t));p.write_text('y')
            with self.assertRaises(ValueError):verify_rows(Path(t),[r])
    def test_false_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a';write(p,{'approved':False,'action':'TRAIN'})
            with self.assertRaises(PermissionError):require_authorisation(p,'TRAIN',Path(t))
    def test_mismatched_campaign_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);write(t/'campaign_manifest.json',{});write(t/'a',{'approved':True,'action':'TRAIN','campaign_manifest_sha256':'bad'})
            with self.assertRaises(PermissionError):require_authorisation(t/'a','TRAIN',t)
    def test_test_release_bound_to_models(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);write(t/'campaign_manifest.json',{})
            write(t/'a',{'approved':True,'action':'EVALUATE_FINAL_TESTS','campaign_manifest_sha256':sha(t/'campaign_manifest.json'),'package_manifest_sha256':'p','selected_models_manifest_sha256':'wrong'})
            with patch('main_study.common.package_integrity',return_value='p'):
                with self.assertRaises(PermissionError):require_authorisation(t/'a','EVALUATE_FINAL_TESTS',t,expected_models_sha='right')


class CheckpointContractTests(unittest.TestCase):
    def test_actual_B_C_expressions_match_global_tie_contract(self):
        histories=[
          [{'round':0,'validation_loss':0.},{'round':1,'validation_loss':1.},{'round':2,'validation_loss':1.-.7e-12},{'round':3,'validation_loss':1.-1.4e-12}],
          [{'round':0,'validation_loss':0.},{'round':1,'validation_loss':1.},{'round':2,'validation_loss':.5},{'round':3,'validation_loss':.5}],
        ]
        for module in ('ehr_fedavg','ehr_secagg'):
            tree=ast.parse((ROOT/f'app/{module}/server_app.py').read_text())
            assignments={n.targets[0].id:n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('eligible','minimum','best')}
            code=compile(ast.fix_missing_locations(ast.Module(body=[assignments[x] for x in ('eligible','minimum','best')],type_ignores=[])),'fixture-tie','exec')
            for hist in histories:
                from types import SimpleNamespace
                env={'strategy':SimpleNamespace(history=hist),'conf':{'checkpoint_tie_tolerance':1e-12}}
                exec(code,env)
                minimum=min(h['validation_loss'] for h in hist[1:])
                expected=next(h['round'] for h in hist[1:] if h['validation_loss']<=minimum+1e-12)
                self.assertEqual(env['best']['round'],expected)


class CentralKernelTests(unittest.TestCase):
    def test_actual_partial_batch_and_reset(self):
        from ehr_fedavg.task import model_factory,setup_torch
        from main_study.central import fit_epoch
        setup_torch();set_training_seed(20260906);m=model_factory();schema=read(ROOT/'config/encoding_schema.json')
        cache={};order=[]
        for i in range(65):
            x=torch.zeros(1,119);cat=torch.zeros(1,13,dtype=torch.int64)
            for f in schema['categorical_fields']:x[:,f['one_hot_start_inclusive']]=1
            s={'x':x,'categorical':cat,'categorical_valid':torch.ones(1,13,dtype=torch.bool),'binary':x[:,110:113].clone(),'temporal':x[:,113:119].clone(),'session_id':str(i),'client':'C1','partition':'training'}
            cache[('C1',str(i))]=s;order.append({'client':'C1','session_id':str(i)})
        conf=cell_config(11,'reference',30);before=state_digest(m)
        a,info=fit_epoch(m,cache,order,conf);b,info2=fit_epoch(m,cache,order,conf)
        self.assertEqual([r['sessions'] for r in a],[64,1]);self.assertEqual([r['sessions'] for r in b],[64,1]);self.assertNotEqual(before,state_digest(m));self.assertTrue(info2['fresh_optimizer'])
    def test_five_distinct_initialisations(self):
        schema=read(ROOT/'config/encoding_schema.json');hashes=[]
        for seed in SEEDS:
            set_training_seed(seed);m=CompactLSTMAutoencoder([len(f['categories']) for f in schema['categorical_fields']]);hashes.append(state_digest(m))
        self.assertEqual(len(set(hashes)),5)
