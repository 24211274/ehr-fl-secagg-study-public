"""Regression for MAIN-FIXTURE-IMPORT-002, without weakening resource checks.

These tests use generated configuration fixtures only. No source CSV, validation
scores, learned checkpoints, Flower deployment or network service is required.
Native FAB tests remain in test_attempt_staging and are mandatory on the host.
"""
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from main_study.common import ROOT, read, sha, write
from main_study.design import bind_app, cell_config
from main_study.staging import stage_attempt_job
from tests_main._isolated_app_probe import run_isolated_probe


def fixture(root, mode='B', condition='reference', seed=11):
    root = Path(root)
    app = root / 'apps' / condition
    shutil.copytree(ROOT / 'app', app, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    conf = cell_config(seed, condition, 30)
    bind_app(app, conf, configuration=mode)
    write(root / 'job.json', {'configuration': mode, 'condition': condition,
                             'rounds': 30, 'training_seed': seed})
    write(root / 'runtime.json', {'run_token': 'fixture-only-no-data',
                                 'expected_initial_digest': 'not-a-checkpoint'})
    write(root / 'client_data/C1/fixture.json', {'temporary_fixture': True})
    return app


def snapshot(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in path.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


class FixtureResourceIsolationTests(unittest.TestCase):
    def test_old_in_process_comparison_reproduces_reported_failure(self):
        # Loading from the release first is exactly what unittest discovery does.
        res = importlib.import_module('ehr_fedavg.resources')
        before_module = res
        before = res.resource_sha256('fixed_config.json')
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'; app = fixture(job)
            self.assertNotEqual(before, sha(app / 'ehr_fedavg/fixed_config.json'))
            # Even prepending a new path cannot replace the cached module.
            with patch.object(sys, 'path', [str(app)] + sys.path):
                self.assertIs(importlib.import_module('ehr_fedavg.resources'), before_module)
                with self.assertRaisesRegex(ValueError, 'Original JSON and bundled representation differ: fixed_config.json'):
                    res.verify_source_resources(app)
            self.assertEqual(before, res.resource_sha256('fixed_config.json'))

    def run_valid(self, mode, condition):
        with tempfile.TemporaryDirectory(prefix='fixture with spaces ') as tmp:
            job = Path(tmp) / 'job'; app = fixture(job, mode, condition)
            before = snapshot(app)
            report = run_isolated_probe(app, job, mode, condition, job / 'reports/resources.json')
            identity = report['test_fixture_import']
            self.assertEqual(identity['training_seed'], 11)
            self.assertEqual(identity['rounds'], 30)
            self.assertEqual(identity['condition'], condition)
            self.assertEqual(identity['config_sha256'], sha(app / 'ehr_fedavg/fixed_config.json'))
            self.assertEqual(report['source_resource_identity']['status'], 'PASS')
            self.assertFalse(report['native_flower_build_executed'])
            self.assertEqual(before, snapshot(app))
            self.assertFalse(list(app.rglob('*.pyc')))
    def test_B_reference_has_correct_resources(self): self.run_valid('B', 'reference')
    def test_B_noniid_has_correct_resources(self): self.run_valid('B', 'noniid')
    def test_C_reference_has_correct_resources(self): self.run_valid('C', 'reference')
    def test_C_noniid_has_correct_resources(self): self.run_valid('C', 'noniid')

    def test_multiple_seeds_do_not_reuse_parent_module(self):
        res = importlib.import_module('ehr_fedavg.resources')
        before = res.resource_bytes('fixed_config.json')
        module_ids = {k: id(v) for k, v in sys.modules.items()
                      if k == 'ehr_fedavg' or k.startswith('ehr_fedavg.')}
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            for seed in (11, 29, 47):
                job = Path(tmp) / str(seed); app = fixture(job, seed=seed)
                report = run_isolated_probe(app, job, 'B', 'reference', job / 'report.json')
                self.assertEqual(report['test_fixture_import']['training_seed'], seed)
                hashes.append(report['test_fixture_import']['config_sha256'])
            self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(res.resource_bytes('fixed_config.json'), before)
        self.assertEqual(module_ids, {k: id(v) for k, v in sys.modules.items()
                         if k == 'ehr_fedavg' or k.startswith('ehr_fedavg.')})

    def test_isolation_survives_poisoned_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job = root / 'job'; app = fixture(job)
            poison = root / 'poison/ehr_fedavg'; poison.mkdir(parents=True)
            (poison / '__init__.py').write_text("raise RuntimeError('WRONG PYTHONPATH')\n")
            with patch.dict(os.environ, {'PYTHONPATH': str(poison.parent)}):
                report = run_isolated_probe(app, job, 'B', 'reference', job / 'report.json')
            self.assertEqual(report['status'], 'PASS')

    def test_altered_json_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'; app = fixture(job)
            p = app / 'ehr_fedavg/fixed_config.json'; conf = read(p); conf['training_seed'] = 97; write(p, conf)
            with self.assertRaisesRegex(RuntimeError, 'Original JSON and bundled representation differ: fixed_config.json'):
                run_isolated_probe(app, job, 'B', 'reference', job / 'report.json')

    def test_altered_carrier_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'; app = fixture(job)
            p = app / 'ehr_fedavg/_embedded_resources.py'
            p.write_text(p.read_text() + "\nRESOURCE_BYTES['fixed_config.json'] += b' '\n")
            with self.assertRaisesRegex(RuntimeError, 'Bundled resource digest mismatch'):
                run_isolated_probe(app, job, 'B', 'reference', job / 'report.json')

    def test_missing_carrier_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'; app = fixture(job)
            (app / 'ehr_fedavg/_embedded_resources.py').unlink()
            with self.assertRaisesRegex(RuntimeError, '_embedded_resources'):
                run_isolated_probe(app, job, 'B', 'reference', job / 'report.json')

    def test_mismatched_schema_link_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'; app = fixture(job)
            conf = cell_config(11, 'reference', 30); conf['schema_sha256'] = '0' * 64
            bind_app(app, conf, configuration='B')
            with self.assertRaisesRegex(RuntimeError, 'Configuration/schema digest link differs'):
                run_isolated_probe(app, job, 'B', 'reference', job / 'report.json')

    def test_existing_test_report_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'job'; app = fixture(job); report = job / 'report.json'
            write(report, {'existing': 'retain'}); before = report.read_bytes()
            with self.assertRaises(FileExistsError):
                run_isolated_probe(app, job, 'B', 'reference', report)
            self.assertEqual(report.read_bytes(), before)

    def test_template_and_attempt_resources_stay_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job = root / 'template'; app = fixture(job, 'C', 'noniid')
            first = run_isolated_probe(app, job, 'C', 'noniid', job / 'reports/first.json')
            (job / 'bundle_preflight/noniid').mkdir(parents=True)
            (job / 'bundle_preflight/noniid/verified_app.fab').write_bytes(b'fixture probe artefact')
            before = snapshot(job)
            attempt = stage_attempt_job(job, root / 'attempt')
            second = run_isolated_probe(attempt / 'apps/noniid', attempt, 'C', 'noniid',
                                        attempt / 'reports/second.json')
            self.assertEqual(first['test_fixture_import']['config_sha256'],
                             second['test_fixture_import']['config_sha256'])
            self.assertNotEqual(first['test_fixture_import']['resources_module'],
                                second['test_fixture_import']['resources_module'])
            self.assertFalse((attempt / 'bundle_preflight').exists())
            self.assertEqual(before, snapshot(job))

    def test_helper_is_test_only_and_native_tests_remain(self):
        # Guard against a future "fix" that silently removes the original checks.
        native_tests = (ROOT / 'tests_main/test_attempt_staging.py').read_text()
        self.assertEqual(native_tests.count('def test_native_'), 4)
        self.assertIn('native=True', native_tests)
        for folder in ('app', 'main_study', 'scripts'):
            for p in (ROOT / folder).rglob('*.py'):
                self.assertNotIn('_isolated_app_probe', p.read_text(), str(p))
        for pkg in ('ehr_fedavg', 'ehr_secagg'):
            src = (ROOT / 'app' / pkg / 'bundle_checks.py').read_text()
            self.assertIn('source_resource_identity=verify_source_resources(app_directory)', src)


if __name__ == '__main__':
    unittest.main()
