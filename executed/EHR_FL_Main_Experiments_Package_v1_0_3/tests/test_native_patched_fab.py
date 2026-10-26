"""Actual patched-Flower FAB preflight entry tests, without benchmark datasets.

The same entry point used by each runner executes its exact build identity guard,
real Flower build twice, content checks, and isolated callback/model probe. Only
client dataset loading is excluded by the test adapter; deployment keeps it on.
"""
from __future__ import annotations
import importlib,importlib.util,sys,tempfile,unittest,shutil,json
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from ehr_fedavg.host import write_project,local_environment
from mai_runtime_identity import VERSION
HAS_FLOWER=importlib.util.find_spec('flwr') is not None

@unittest.skipUnless(HAS_FLOWER,'Real patched Flower required: mandatory on target host')
class NativePatchedFabEntryTests(unittest.TestCase):
    def check(self,mode,condition):
        module=importlib.import_module(mode+'.bundle_checks')
        with tempfile.TemporaryDirectory(prefix='actual patched fab ') as folder:
            root=Path(folder);app=root/'app'
            shutil.copytree(ROOT/'app',app,ignore=shutil.ignore_patterns('__pycache__'))
            write_project(app/'pyproject.toml',root/'no_dataset',condition,19093)
            if mode=='ehr_secagg':
                p=app/'pyproject.toml';p.write_text(p.read_text().replace('ehr_fedavg.server_app','ehr_secagg.server_app').replace('ehr_fedavg.client_app','ehr_secagg.client_app'))
            real_probe=module.probe_unpacked_fab
            def no_dataset_probe(fab,appdir,work,runtime,env):
                return real_probe(fab,appdir,work,None,env,require_flower=True)
            report_path=root/'report.json'
            with patch.object(module,'probe_unpacked_fab',side_effect=no_dataset_probe):
                result=module.preflight_native_fab(app,root/'runtime',condition,report_path,local_environment(root/'home'))
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['flower_version'],VERSION)
            self.assertTrue(result['native_flower_build_executed']);self.assertTrue(result['native_rebuild_byte_identical'])
            self.assertTrue(result['isolated_import']['flower_callbacks_created'])
            self.assertFalse(result['services_started_by_this_gate'])
            self.assertEqual(result['isolated_import']['clients_checked'],[])
            self.assertEqual(result['isolated_import']['optimiser_steps'],0)
            self.assertEqual(result['isolated_import']['candidate_implementation_identity']['version'],VERSION)
            self.assertEqual(json.loads(report_path.read_text()),result)
    def test_B_reference_entry(self):self.check('ehr_fedavg','reference')
    def test_B_noniid_entry(self):self.check('ehr_fedavg','noniid')
    def test_C_reference_entry(self):self.check('ehr_secagg','reference')
    def test_C_noniid_entry(self):self.check('ehr_secagg','noniid')

if __name__=='__main__':unittest.main()
