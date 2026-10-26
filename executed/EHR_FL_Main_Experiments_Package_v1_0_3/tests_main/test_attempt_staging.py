"""Regression: prepared FAB artefacts must not enter a fresh execution attempt."""
import hashlib, importlib.util, json, shutil, tempfile, tomllib, unittest
from pathlib import Path
from unittest.mock import patch
from main_study.common import ROOT, read, write, sha
from main_study.staging import stage_attempt_job


def make_template(path, mode='B', condition='reference', *, rounds=30):
    path=Path(path); app=path/'apps'/condition
    conf={'rounds':rounds,'conditions':[condition],'training_seed':11,'ports':{'control':19093}}
    write(app/'ehr_fedavg/fixed_config.json',conf)
    (app/'fixture.py').write_text('value = 1\n')
    write(path/'client_data/C1/fixture.json',{'prepared':True})
    write(path/'job.json',{'configuration':mode,'condition':condition,'rounds':rounds,'training_seed':11})
    write(path/'runtime.json',{'run_token':'old-token','expected_initial_digest':'fixed','client_files':{'C1':[]}})
    write(path/'reports/job_preflight.json',{'status':'PASS'})
    (path/'bundle_preflight'/condition).mkdir(parents=True)
    (path/'bundle_preflight'/condition/'verified_app.fab').write_bytes(b'previous probe')
    (path/'probe_flower_home').mkdir()
    (app/'previous.fab').write_bytes(b'old location')
    return path


def snapshot(root):
    return {x.relative_to(root).as_posix():sha(x) for x in root.rglob('*') if x.is_file()}


class AttemptStagingTests(unittest.TestCase):
    def test_original_copy_reproduces_B_failure(self): self.old_failure('B')
    def test_original_copy_reproduces_C_failure(self): self.old_failure('C')
    def old_failure(self, mode):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=make_template(root/'source',mode)
            shutil.copytree(source,root/'old',ignore=shutil.ignore_patterns('probe_flower_home','reports','*.fab','__pycache__'))
            with self.assertRaises(FileExistsError):
                (root/'old/bundle_preflight/reference').mkdir(parents=True,exist_ok=False)
    def good(self, mode, condition):
        with tempfile.TemporaryDirectory(prefix='clone with spaces ') as t:
            root=Path(t);source=make_template(root/'source',mode,condition);before=snapshot(source)
            result=stage_attempt_job(source,root/'new attempt')
            self.assertEqual(set(x.name for x in result.iterdir()),{'apps','client_data','runtime.json','job.json'})
            self.assertEqual(read(result/'job.json'),read(source/'job.json'))
            self.assertEqual(sha(result/'client_data/C1/fixture.json'),sha(source/'client_data/C1/fixture.json'))
            self.assertEqual(sha(result/'apps'/condition/'ehr_fedavg/fixed_config.json'),sha(source/'apps'/condition/'ehr_fedavg/fixed_config.json'))
            r=read(result/'runtime.json');old=read(source/'runtime.json');self.assertNotEqual(r.pop('run_token'),old.pop('run_token'));self.assertEqual(r,old)
            doc=tomllib.loads((result/'apps'/condition/'pyproject.toml').read_text())
            self.assertEqual(doc['tool']['flwr']['app']['config']['num-server-rounds'],30)
            self.assertIn(str(result),(result/'apps'/condition/'pyproject.toml').read_text())
            self.assertEqual(snapshot(source),before)
            # Exact non-overwrite precondition previously violated by the copied directory.
            work=result/'bundle_preflight'/condition;work.mkdir(parents=True,exist_ok=False)
            with self.assertRaises(FileExistsError):work.mkdir(parents=True,exist_ok=False)
            self.assertFalse((result/'apps'/condition/'previous.fab').exists())
    def test_B_reference_copy_reprobe(self):self.good('B','reference')
    def test_B_noniid_copy_reprobe(self):self.good('B','noniid')
    def test_C_reference_copy_reprobe(self):self.good('C','reference')
    def test_C_noniid_copy_reprobe(self):self.good('C','noniid')
    def test_A_copy(self):self.good('A','reference')
    def test_existing_attempt_refused_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=make_template(p/'source');dest=p/'dest';dest.mkdir();(dest/'evidence').write_text('keep')
            with self.assertRaises(FileExistsError):stage_attempt_job(source,dest)
            self.assertEqual((dest/'evidence').read_text(),'keep')
    def test_missing_input_refused_before_copy(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source');(s/'runtime.json').unlink()
            with self.assertRaises(ValueError):stage_attempt_job(s,p/'dest')
            self.assertFalse((p/'dest').exists())
    def test_trained_template_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source');(s/'checkpoints').mkdir()
            with self.assertRaises(ValueError):stage_attempt_job(s,p/'dest')
            self.assertFalse((p/'dest').exists())
    def test_unknown_top_level_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source');(s/'other_state').write_text('unknown')
            with self.assertRaises(ValueError):stage_attempt_job(s,p/'dest')
    def test_symlink_input_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source');(s/'apps/reference/link').symlink_to(s/'job.json')
            with self.assertRaises(ValueError):stage_attempt_job(s,p/'dest')
    def test_nested_attempt_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source')
            with self.assertRaises(ValueError):stage_attempt_job(s,s/'dest')
    def test_config_round_drift_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source');c=read(s/'job.json');c['rounds']=29;write(s/'job.json',c)
            with self.assertRaises(ValueError):stage_attempt_job(s,p/'dest')
    def test_fresh_attempts_have_distinct_tokens(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=make_template(p/'source');a=stage_attempt_job(s,p/'a');b=stage_attempt_job(s,p/'b')
            self.assertNotEqual(read(a/'runtime.json')['run_token'],read(b/'runtime.json')['run_token'])
    def test_B_C_callbacks_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for mode,pkg in [('B','ehr_fedavg'),('C','ehr_secagg')]:
                s=make_template(p/mode,mode);d=stage_attempt_job(s,p/(mode+'new'))
                self.assertIn(pkg+'.server_app:app',(d/'apps/reference/pyproject.toml').read_text())


@unittest.skipUnless(importlib.util.find_spec('flwr') is not None,'Native patched Flower mandatory on target')
class NativeCloneReprobeTests(unittest.TestCase):
    def check(self, mode, condition):
        from main_study.design import cell_config, bind_app
        from ehr_fedavg.host import write_project
        from tests_main._isolated_app_probe import run_isolated_probe
        with tempfile.TemporaryDirectory(prefix='native template clone ') as t:
            root=Path(t);template=make_template(root/'template',mode,condition)
            shutil.rmtree(template/'apps');shutil.rmtree(template/'bundle_preflight')
            app=template/'apps'/condition
            shutil.copytree(ROOT/'app',app,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            conf=cell_config(11,condition,30);bind_app(app,conf,configuration=mode)
            write_project(app/'pyproject.toml',template,condition,19093)
            p=app/'pyproject.toml';s=p.read_text().replace('num-server-rounds = 6','num-server-rounds = 30')
            if mode=='C':s=s.replace('ehr_fedavg.server_app','ehr_secagg.server_app').replace('ehr_fedavg.client_app','ehr_secagg.client_app')
            p.write_text(s)
            # Each probe imports the resources from its own bound application,
            # as scripts/job_worker.py does in the actual preparation workflow.
            # The unittest process already loaded release-default resources.
            first=run_isolated_probe(app,template,mode,condition,
                                      template/'reports/first.json',native=True)
            before=snapshot(template)
            attempt=stage_attempt_job(template,root/'attempt')
            second=run_isolated_probe(attempt/'apps'/condition,attempt,mode,condition,
                                       attempt/'reports/second.json',native=True)
            for result,target in ((first,app),(second,attempt/'apps'/condition)):
                self.assertEqual(result['test_fixture_import']['config_sha256'],
                                 sha(target/'ehr_fedavg/fixed_config.json'))
                self.assertEqual(result['test_fixture_import']['training_seed'],11)
                self.assertEqual(result['test_fixture_import']['rounds'],30)
                self.assertEqual(result['test_fixture_import']['condition'],condition)
            self.assertEqual(first['status'],'PASS');self.assertEqual(second['status'],'PASS')
            self.assertEqual(before,snapshot(template))
            self.assertTrue(second['native_flower_build_executed'])
            self.assertFalse(second['services_started_by_this_gate'])
            self.assertEqual(second['isolated_import']['optimiser_steps'],0)
    def test_native_B_reference(self):self.check('B','reference')
    def test_native_B_noniid(self):self.check('B','noniid')
    def test_native_C_reference(self):self.check('C','reference')
    def test_native_C_noniid(self):self.check('C','noniid')
