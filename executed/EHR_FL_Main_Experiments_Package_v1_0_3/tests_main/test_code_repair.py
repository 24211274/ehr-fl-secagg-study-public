"""Code repair provenance and no-training boundaries, using fabricated artefacts."""
import hashlib, json, shutil, sys, tempfile, types, unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from main_study.common import ROOT, read, write, sha, file_record, require_authorisation
from main_study.repair import verify_repair_parent, copy_parent_frozen
from fullsize_support.integrity import package_report
from test_attempt_staging import make_template, snapshot


def parent_fixture(root):
    parent=Path(root)/'parent';parent.mkdir()
    source=parent/'source';source.mkdir();(source/'a.py').write_text('x=1\n')
    (source/'PACKAGE_SHA256SUMS.txt').write_text(sha(source/'a.py')+'  a.py\n')
    write(parent/'origins.json',{'inference_run':str(Path(root)/'inference'),'accepted_run':str(Path(root)/'accepted'),'preprocessing_run':str(Path(root)/'pre')})
    (parent/'frozen/initials').mkdir(parents=True);initials={}
    for seed in (11,29,47,71,97):
        path=parent/'frozen/initials'/f'seed_{seed}.pt';path.write_bytes(f'fixture {seed}'.encode())
        initials[str(seed)]={'sha256':sha(path),'state_digest':str(seed)*8}
        for co in ('reference','noniid'):
            d=parent/'frozen/schedules'/f'{seed}_{co}.csv';d.parent.mkdir(exist_ok=True);d.write_text('fixture\n')
    write(parent/'frozen/initial_states.json',initials)
    write(parent/'frozen/development/fixture.json',{'only_development':True})
    a=parent/'attempts/MAIN-11-reference/attempt_1/A'
    cp=a/'checkpoints/reference/selected_state.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'archived trained fixture')
    write(a/'calibration.json',{'fixed':'fixture'});write(a/'LOCAL_ONLY/selected_validation_scores.json',{'rows':[]})
    complete={'status':'PASS','configuration':'A','condition':'reference','seed':11,'rounds':30,
              'selected_checkpoint_sha256':sha(cp),'calibration_sha256':sha(a/'calibration.json'),
              'validation_scores_sha256':sha(a/'LOCAL_ONLY/selected_validation_scores.json')}
    write(a/'reports/cell_complete.json',complete)
    failure=parent/'attempts/MAIN-11-reference/attempt_1/reports'
    status={'block_id':'MAIN-11-reference','attempt':1,'status':'FAILED',
            'path':'attempts/MAIN-11-reference/attempt_1','root_error_type':'FileExistsError',
            'completed_cells':[complete],'test_data_opened':False,'privacy_claim_validated':False}
    write(failure/'block_status.json',status)
    package_report(failure,parent/'fixture_report.zip')
    entry={k:v for k,v in status.items() if k not in ('test_data_opened','privacy_claim_validated')}
    write(parent/'progress.json',{'completed_blocks':[],'attempts':{'MAIN-11-reference':entry},'main_training_started':True})
    write(parent/'prepare_reports/stage_status.json',{'status':'MAIN_PREPARATION_PASS_NOT_MAIN_TRAINING_OR_TEST'})
    write(parent/'prepare_reports/engineering_A_summary.json',{'status':'PASS','rounds_completed':1,'local_updates':170})
    write(parent/'prepare_reports/sampling_audits.json',{'fixture':True})
    package_report(parent/'prepare_reports',parent/'fixture_prepare_report.zip')
    for mode in 'BC':(parent/'jobs'/f'MAIN-11-reference-{mode}'/'bundle_preflight/reference').mkdir(parents=True)
    rows=[file_record(p,parent) for p in sorted((parent/'frozen').rglob('*')) if p.is_file()]+[file_record(parent/'origins.json',parent)]
    write(parent/'campaign_manifest.json',{'files':rows,'package_manifest_sha256':sha(source/'PACKAGE_SHA256SUMS.txt'),
          'decision_sha256':sha(ROOT/'protocol/decisions/config/main_protocol_decisions.json')})
    policy={'repair_id':'FIXTURE','parent_campaign':str(parent),'parent_campaign_manifest_sha256':sha(parent/'campaign_manifest.json'),
            'parent_source_manifest_sha256':sha(source/'PACKAGE_SHA256SUMS.txt'),'block':'MAIN-11-reference',
            'failure_report_relative':'attempts/MAIN-11-reference/attempt_1/reports',
            'failure_report_manifest_sha256':sha(failure/'report_files.json'),'failure_block_status_sha256':sha(failure/'block_status.json'),
            'old_complete_cell_sha256':sha(a/'reports/cell_complete.json'),
            'parent_preparation_report_manifest_sha256':sha(parent/'prepare_reports/report_files.json')}
    return parent,policy


class RepairProvenanceTests(unittest.TestCase):
    def test_original_FileExistsError_is_not_transient_retry(self):
        from main_study.campaign import transient_type_eligible
        self.assertFalse(transient_type_eligible({'error_type':'RuntimeError','root_error_type':'FileExistsError'}))
    def good(self,kind=None):
        with tempfile.TemporaryDirectory() as t:
            p,policy=parent_fixture(t);new=Path(t)/'successor'
            if kind=='progress':write(p/'progress.json',{'completed_blocks':['other'],'attempts':{},'main_training_started':True})
            if kind=='manifest':(p/'campaign_manifest.json').write_text('{}')
            if kind=='source':(p/'source/a.py').write_text('changed')
            if kind=='source_extra':(p/'source/extra.py').write_text('changed')
            if kind=='frozen':(p/'frozen/initials/seed_11.pt').write_bytes(b'changed')
            if kind=='calibration':(p/'attempts/MAIN-11-reference/attempt_1/A/calibration.json').write_text('{}')
            if kind=='probe_deleted':shutil.rmtree(p/'jobs/MAIN-11-reference-B/bundle_preflight')
            if kind=='final_evaluation':(p/'final_evaluation').mkdir()
            if kind=='selected':write(p/'selected_models_manifest.json',{})
            if kind=='bad_report':(p/'attempts/MAIN-11-reference/attempt_1/reports/block_status.json').write_text('{}')
            before=snapshot(p)
            with patch('main_study.repair.repair_policy',return_value=policy):
                if kind:
                    with self.assertRaises(ValueError):verify_repair_parent(p,new)
                    self.assertFalse(new.exists());return
                audit=verify_repair_parent(p,new)
                self.assertEqual(audit['status'],'PASS');self.assertFalse(audit['old_A_selected_for_primary_analysis'])
                result=copy_parent_frozen(p,new,audit)
                self.assertEqual(result['shared_inodes_with_parent'],0);self.assertTrue(result['all_sha256_identical'])
                self.assertFalse((new/'attempts').exists());self.assertFalse((new/'progress.json').exists())
                self.assertEqual(snapshot(p),before)
                self.assertEqual(verify_repair_parent(p,new),audit)
    def test_review_and_frozen_copy_preserve_parent(self):self.good()
    def test_progressed_parent_refused(self):self.good('progress')
    def test_changed_parent_manifest_refused(self):self.good('manifest')
    def test_changed_parent_source_refused(self):self.good('source')
    def test_unlisted_parent_source_refused(self):self.good('source_extra')
    def test_changed_initial_weights_refused(self):self.good('frozen')
    def test_changed_archived_A_refused(self):self.good('calibration')
    def test_deleted_template_probe_refused(self):self.good('probe_deleted')
    def test_parent_test_access_refused(self):self.good('final_evaluation')
    def test_sealed_final_models_refused(self):self.good('selected')
    def test_changed_report_refused(self):self.good('bad_report')
    def test_existing_successor_frozen_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            p,policy=parent_fixture(t);n=Path(t)/'new'
            with patch('main_study.repair.repair_policy',return_value=policy):audit=verify_repair_parent(p,n)
            (n/'frozen').mkdir(parents=True)
            with self.assertRaises(FileExistsError):copy_parent_frozen(p,n,audit)
    def test_unlisted_frozen_artefact_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p,policy=parent_fixture(t);n=Path(t)/'new'
            with patch('main_study.repair.repair_policy',return_value=policy):audit=verify_repair_parent(p,n)
            (p/'frozen/unlisted').write_text('unknown')
            with self.assertRaises(ValueError):copy_parent_frozen(p,n,audit)
    def test_old_training_receipt_not_valid_for_new_campaign(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);write(p/'campaign_manifest.json',{'new':True});write(p/'old.json',{
                'approved':True,'action':'TRAIN','test_data_authorised':False,'campaign_manifest_sha256':'old','package_manifest_sha256':'old'})
            with self.assertRaises(PermissionError):require_authorisation(p/'old.json','TRAIN',p)


class RepairPreparationBoundaryTests(unittest.TestCase):
    def run_case(self,fail_clone=False):
        from main_study.campaign import prepare
        calls=[];published=[]
        with tempfile.TemporaryDirectory() as t,ExitStack() as stack:
            parent,policy=parent_fixture(t);new=Path(t)/'successor';new.mkdir()
            origins=read(parent/'origins.json');inference=Path(origins['inference_run'])
            write(inference/'results/stage_status.json',{'status':'INFERENCE_SEED_PILOT_PASS_NOT_MAIN_OR_PRIVACY_CERTIFICATION'})
            before=snapshot(parent)
            stack.enter_context(patch('main_study.repair.repair_policy',return_value=policy))
            fake=types.ModuleType('prg_validation.public_protocol');fake.run_oracle=lambda *a:{'status':'PASS','fixture':True}
            stack.enter_context(patch.dict(sys.modules,{'prg_validation.public_protocol':fake}))
            def stage(campaign,job,conf,schedule,initial,mode):
                make_template(job,mode,conf['conditions'][0]);shutil.rmtree(job/'bundle_preflight')
                write(job/'apps'/conf['conditions'][0]/'ehr_fedavg/fixed_config.json',conf)
                write(job/'job.json',{'configuration':mode,'condition':conf['conditions'][0],'rounds':conf['rounds'],'training_seed':conf['training_seed']})
                return job
            def command(args,log,*more,**kw):
                Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text('fixture command log\n')
                action=args[-1];job=Path(args[-2]);calls.append((job.name,action))
                if action!='Probe':raise AssertionError('Repair preparation must not train')
                if fail_clone and 'attempt_clone_probes' in job.parts:raise RuntimeError('fixture cloned native probe failed')
                meta=read(job/'job.json')
                if meta['configuration'] in ('B','C'):(job/'bundle_preflight'/meta['condition']).mkdir(parents=True,exist_ok=False)
                write(job/'reports/job_preflight.json',{'status':'PASS'})
            for name,value in [('stage_cell',stage),('command',command),('environment',lambda *a:{}),
                               ('original_snapshot',lambda *a:{}),('verify_development',lambda *a:{'status':'PASS'}),
                               ('run_tests',lambda *a:None),('check_report',lambda *a:{'status':'PASS'}),
                               ('publish',lambda *a:published.append(a))]:
                stack.enter_context(patch('main_study.campaign.'+name,side_effect=value))
            stack.enter_context(patch('main_study.campaign.shutil.disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()))
            rc=prepare(new,inference,Path(origins['accepted_run']),Path(origins['preprocessing_run']),repair_parent=parent)
            status=read(new/'prepare_reports/stage_status.json')
            self.assertEqual(snapshot(parent),before)
            self.assertTrue(all(action=='Probe' for _,action in calls))
            self.assertFalse(status['test_data_opened']);self.assertFalse(status['main_training_performed'])
            self.assertEqual(len(published),1)
            if fail_clone:
                self.assertEqual(rc,1);self.assertEqual(status['message'],'fixture cloned native probe failed')
                self.assertFalse((new/'campaign_manifest.json').exists())
                self.assertFalse((new/'progress.json').exists())
            else:
                self.assertEqual(rc,0);self.assertEqual(len(calls),50)
                m=read(new/'campaign_manifest.json');self.assertIn('code_repair_amendment.json',{x['path'] for x in m['files']})
                self.assertEqual(read(new/'progress.json'),{'completed_blocks':[],'attempts':{},'main_training_started':False})
                self.assertFalse(read(new/'code_repair_amendment.json')['successor_training_authorised'])
                self.assertEqual(read(new/'frozen/initial_states.json'),read(parent/'frozen/initial_states.json'))
                self.assertFalse((new/'engineering_A').exists())
    def test_successor_preparation_is_50_probes_zero_training(self):self.run_case()
    def test_clone_probe_failure_cannot_seal_or_start_training(self):self.run_case(True)
