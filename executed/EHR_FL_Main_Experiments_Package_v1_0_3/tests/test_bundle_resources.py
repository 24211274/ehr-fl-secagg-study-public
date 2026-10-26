"""Packaging regression checks; test ZIPs are not labelled native Flower builds."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from unittest.mock import patch

from ehr_fedavg import resources
from ehr_fedavg.resources import resource_bytes, resource_sha256, read_document, verify_source_resources
from ehr_fedavg.host import write_project, local_environment
from ehr_fedavg.bundle_checks import source_files, verify_fab_contents, probe_unpacked_fab

ROOT=Path(__file__).resolve().parents[1]
FLOWER=importlib.util.find_spec('flwr') is not None


def source_app(dest: Path) -> Path:
    app=dest/'reference';shutil.copytree(ROOT/'app',app,ignore=shutil.ignore_patterns('__pycache__'))
    write_project(app/'pyproject.toml',dest/'runtime','reference',19093)
    return app


def filtered_test_zip(app: Path) -> bytes:
    """Test fixture matching documented suffix/filter rules, NOT Flower's builder."""
    source=source_files(app)
    toml=(app/'pyproject.toml').read_text().split('[tool.flwr.federations]')[0]
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        z.writestr('pyproject.toml',toml)
        for name,raw in source.items():z.writestr(name,raw)
        content=''.join(f'{name},{hashlib.sha256(raw).hexdigest()},{len(raw)*8}\n' for name,raw in source.items())
        z.writestr('.info/CONTENT',content)
    return b.getvalue()


def alter_zip(raw: bytes, *, omit: str|None=None, change: tuple[str,bytes]|None=None,
              extra: tuple[str,bytes]|None=None) -> bytes:
    b=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as original,zipfile.ZipFile(b,'w') as new:
        for n in original.namelist():
            if n!=omit:new.writestr(n,change[1] if change and n==change[0] else original.read(n))
        if extra:new.writestr(extra[0],extra[1])
    return b.getvalue()


class ResourceTests(unittest.TestCase):
    def test_config_exact_original_bytes(self):
        self.assertEqual(resource_bytes('fixed_config.json'),(ROOT/'config/timing_B.json').read_bytes())
    def test_schema_exact_original_bytes(self):
        self.assertEqual(resource_bytes('encoding_schema.json'),(ROOT/'config/encoding_schema.json').read_bytes())
    def test_config_semantics_unchanged(self):
        self.assertEqual(read_document('fixed_config.json'),json.loads((ROOT/'config/timing_B.json').read_text()))
    def test_schema_semantics_unchanged(self):
        self.assertEqual(read_document('encoding_schema.json'),json.loads((ROOT/'config/encoding_schema.json').read_text()))
    def test_pinned_schema_digest(self):
        self.assertEqual(resource_sha256('encoding_schema.json'),read_document('fixed_config.json')['schema_sha256'])
    def test_unknown_resource_rejected(self):
        with self.assertRaises(KeyError):resource_bytes('../runtime.json')
    def test_changed_bytes_rejected(self):
        with patch.dict(resources.RESOURCE_BYTES,{'fixed_config.json':b'{}'}):
            with self.assertRaises(ValueError):read_document('fixed_config.json')
    def test_no_mutation_shared_between_reads(self):
        c=read_document('fixed_config.json');c['clients'].clear()
        self.assertEqual(read_document('fixed_config.json')['clients'],['C1','C2','C3'])
    def test_source_resources_checked(self):
        self.assertEqual(verify_source_resources(ROOT/'app')['status'],'PASS')
    def test_changed_original_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=source_app(Path(d));(p/'ehr_fedavg/fixed_config.json').write_text('{}')
            with self.assertRaises(ValueError):verify_source_resources(p)
    def test_generator_is_reproducible(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/generate_bundled_resources.py')],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_consumers_no_adjacent_json_dependency(self):
        for name in ['task.py','server_app.py']:
            s=(ROOT/'app/ehr_fedavg'/name).read_text()
            self.assertNotIn("parent/'fixed_config.json'",s)
            self.assertNotIn("parent/'encoding_schema.json'",s)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='fab with spaces ')
        self.root=Path(self.tmp.name);self.app=source_app(self.root);self.fab=filtered_test_zip(self.app)
    def tearDown(self):self.tmp.cleanup()
    def test_supported_code_carrier_survives_filter(self):
        result=verify_fab_contents(self.fab,self.app)
        self.assertTrue(result['embedded_config_carrier_present']);self.assertTrue(result['original_json_files_absent_as_expected'])
    def test_data_arrays_not_present(self):
        with zipfile.ZipFile(io.BytesIO(self.fab)) as z:
            self.assertFalse(any(n.endswith(('.csv','.npy','.pt','.json')) for n in z.namelist()))
    def test_missing_carrier_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,omit='ehr_fedavg/_embedded_resources.py'),self.app)
    def test_tampered_carrier_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,change=('ehr_fedavg/_embedded_resources.py',b'#tampered')),self.app)
    def test_missing_model_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,omit='pilot_core/model.py'),self.app)
    def test_extra_dataset_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,extra=('data/events.csv',b'a,b')),self.app)
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,extra=('../outside.py',b'x')),self.app)
    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,extra=('/outside.py',b'x')),self.app)
    def test_backslash_path_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,extra=('dir\\outside.py',b'x')),self.app)
    def test_content_ledger_missing_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,omit='.info/CONTENT'),self.app)
    def test_content_ledger_corruption_rejected(self):
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,change=('.info/CONTENT',b'')),self.app)
    def test_toml_changes_rejected(self):
        with zipfile.ZipFile(io.BytesIO(self.fab)) as z:t=z.read('pyproject.toml')
        with self.assertRaises(ValueError):verify_fab_contents(alter_zip(self.fab,change=('pyproject.toml',t.replace(b'num-server-rounds = 6',b'num-server-rounds = 9'))),self.app)
    def test_staged_changed_python_rejected(self):
        p=self.app/'pilot_core/model.py';p.write_bytes(p.read_bytes()+b'\n# change\n')
        with self.assertRaises(ValueError):verify_fab_contents(self.fab,self.app)
    def test_symlink_rejected(self):
        (self.app/'linked.py').symlink_to(self.app/'pilot_core/model.py')
        with self.assertRaises(ValueError):source_files(self.app)
    def test_isolated_unpacked_model_without_json(self):
        result=probe_unpacked_fab(self.fab,self.app,self.root/'probe',None,local_environment(self.root/'home'),require_flower=False)
        self.assertEqual(result['isolated_import']['input_dim'],119)
        self.assertEqual(result['isolated_import']['model_parameters'],86487)
        self.assertEqual(result['isolated_import']['initial_state_digest'],'c009b772108b7735082008635f7894d420500ef645df5450c72eca8bfb6b2e0b')
        self.assertFalse(result['isolated_import']['flower_rounds_executed'])
    def test_source_checkout_cannot_mask_missing_embedded_module(self):
        # Even with source dir on PYTHONPATH, -I must not fall back to the checkout.
        unpacked=self.root/'bad_install';unpacked.mkdir()
        with zipfile.ZipFile(io.BytesIO(self.fab)) as z:z.extractall(unpacked)
        (unpacked/'ehr_fedavg/_embedded_resources.py').unlink()
        env=local_environment(self.root/'home');env['PYTHONPATH']=str(ROOT/'app')
        code='import sys;sys.path.insert(0,sys.argv[1]);from ehr_fedavg.task import cfg;cfg()'
        r=subprocess.run([sys.executable,'-I','-B','-c',code,str(unpacked)],env=env,capture_output=True,text=True,timeout=30,cwd=self.root)
        self.assertNotEqual(r.returncode,0);self.assertIn('_embedded_resources',r.stderr)
    def test_source_forbidden_json_is_not_needed_by_model(self):
        # Paired with the exact byte check: model creation from the filtered ZIP succeeded.
        with zipfile.ZipFile(io.BytesIO(self.fab)) as z:self.assertNotIn('ehr_fedavg/encoding_schema.json',z.namelist())
    def test_probe_refuses_existing_work_directory(self):
        p=self.root/'existing';p.mkdir()
        with self.assertRaises(FileExistsError):probe_unpacked_fab(self.fab,self.app,p,None,local_environment(self.root/'home'),require_flower=False)


@unittest.skipUnless(FLOWER,'Native Flower builder not installed in preparation environment')
class NativeFlowerBundleTests(unittest.TestCase):
    def test_native_builder_and_unpacked_callback_imports(self):
        from flwr.cli.build import build_fab
        with tempfile.TemporaryDirectory(prefix='native fab test ') as d:
            root=Path(d);app=source_app(root);raw,h,_=build_fab(app)
            self.assertEqual(h,hashlib.sha256(raw).hexdigest())
            result=probe_unpacked_fab(raw,app,root/'probe',None,local_environment(root/'home'),require_flower=True)
            self.assertTrue(result['isolated_import']['flower_callbacks_created'])
            self.assertFalse(result['isolated_import']['flower_rounds_executed'])
    def test_native_builder_supported_extensions_pinned(self):
        from flwr.common.constant import FAB_ALLOWED_EXTENSIONS
        self.assertEqual(set(FAB_ALLOWED_EXTENSIONS),{'.py','.toml','.md'})


if __name__=='__main__':unittest.main()
