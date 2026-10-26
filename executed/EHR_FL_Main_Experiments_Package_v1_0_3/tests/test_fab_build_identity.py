"""Regression tests for exact local-build admission, using labelled builder stubs.

This file is runnable under the ORIGINAL environment before downloading/installing.
These are not native Flower build or Deployment evidence. The separate native
integration tests exercise the real preflight in the candidate environment.
"""
from __future__ import annotations
import importlib,importlib.metadata,json,sys,tempfile,types,unittest,hashlib,io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'));sys.path.insert(0,str(ROOT/'tests'))
import mai_fab_policy as policy
from mai_runtime_identity import IdentityMismatch
from test_bundle_resources import source_app,filtered_test_zip


def good_identity():
    return {'status':'PASS','version':policy.VERSION,'build_id':policy.EXPECTED_BUILD_ID,
            'function_sha256':policy.FUNCTION_SHA256,'helper_sha256':policy.HELPER_SHA256,
            'client_server_bindings_match':True,'official_release':False,
            'remote_attestation':False,'privacy_claim_validated':False}


class AdmissionTests(unittest.TestCase):
    def test_accept_exact_build_after_binding_check(self):
        with patch.object(policy,'binding_identity',return_value=good_identity()) as checker:
            result=policy.require_fab_runtime(policy.VERSION)
        checker.assert_called_once_with();self.assertEqual(result,good_identity())
    def test_old_build_rejected_before_binding(self):
        with patch.object(policy,'binding_identity') as checker:
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime('1.22.0')
        checker.assert_not_called()
    def test_arbitrary_local_suffix_rejected(self):
        with self.assertRaises(IdentityMismatch):policy.require_fab_runtime('1.22.0+unknown.1')
    def test_other_candidate_revision_rejected(self):
        with self.assertRaises(IdentityMismatch):policy.require_fab_runtime('1.22.0+mai.prg7570.2')
    def test_newer_upstream_rejected(self):
        with self.assertRaises(IdentityMismatch):policy.require_fab_runtime('1.36.0')
    def test_missing_version_rejected(self):
        with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(None)
    def test_forged_build_same_version_rejected(self):
        record=good_identity();record['build_id']='other'
        with patch.object(policy,'binding_identity',return_value=record):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_helper_hash_mismatch_rejected(self):
        record=good_identity();record['helper_sha256']='0'*64
        with patch.object(policy,'binding_identity',return_value=record):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_function_hash_mismatch_rejected(self):
        record=good_identity();record['function_sha256']='0'*64
        with patch.object(policy,'binding_identity',return_value=record):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_client_server_binding_mismatch_rejected(self):
        record=good_identity();record['client_server_bindings_match']=False
        with patch.object(policy,'binding_identity',return_value=record):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_source_check_exception_propagated(self):
        with patch.object(policy,'binding_identity',side_effect=IdentityMismatch('source changed')):
            with self.assertRaisesRegex(IdentityMismatch,'source changed'):policy.require_fab_runtime(policy.VERSION)
    def test_privacy_claim_cannot_be_promoted(self):
        record=good_identity();record['privacy_claim_validated']=True
        with patch.object(policy,'binding_identity',return_value=record):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_missing_identity_record_rejected(self):
        with patch.object(policy,'binding_identity',return_value=None):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_bool_flags_not_numeric_substitutes(self):
        record=good_identity();record['client_server_bindings_match']=1
        with patch.object(policy,'binding_identity',return_value=record):
            with self.assertRaises(IdentityMismatch):policy.require_fab_runtime(policy.VERSION)
    def test_copy_not_shared(self):
        record=good_identity()
        with patch.object(policy,'binding_identity',return_value=record):output=policy.require_fab_runtime(policy.VERSION)
        output['status']='FAIL';self.assertEqual(record['status'],'PASS')
    def test_old_condition_reproduces_false_rejection(self):
        # This was the exact comparison which rejected the approved local build.
        self.assertNotEqual(policy.VERSION,'1.22.0')
        with patch.object(policy,'binding_identity',return_value=good_identity()):
            self.assertEqual(policy.require_fab_runtime(policy.VERSION)['status'],'PASS')


class PreflightEntryStubTests(unittest.TestCase):
    mode='ehr_fedavg'
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='fab admission ')
        self.root=Path(self.temp.name);self.app=source_app(self.root)
        self.module=importlib.import_module(self.mode+'.bundle_checks')
        self.builder=Mock(return_value=(b'test-zip-not-native',hashlib.sha256(b'test-zip-not-native').hexdigest(),{}))
        fake=types.ModuleType('flwr.cli.build');fake.build_fab=self.builder
        const=types.ModuleType('flwr.common.constant');const.FAB_ALLOWED_EXTENSIONS={'.py','.md','.toml'}
        self.modules={'flwr.cli.build':fake,'flwr.common.constant':const};self.constants=const
        self.report=self.root/'report.json'
    def tearDown(self):self.temp.cleanup()
    def invoke(self,version=policy.VERSION,identity=None,condition='reference'):
        def probe(fab,app,work,runtime,env):
            work.mkdir(parents=True)
            return {'contents':{'status':'PASS','test_stub_only':True},'isolated_import':{'status':'PASS','test_stub_only':True}}
        with patch.object(self.module.importlib.metadata,'version',return_value=version), \
             patch.object(policy,'binding_identity',return_value=identity or good_identity()), \
             patch.dict(sys.modules,self.modules), \
             patch.object(self.module,'probe_unpacked_fab',side_effect=probe), redirect_stdout(io.StringIO()):
            return self.module.preflight_native_fab(self.app,self.root/'runtime',condition,self.report,{})
    def test_preflight_accepts_candidate_not_just_helper(self):
        report=self.invoke();self.assertEqual(self.builder.call_count,2)
        self.assertEqual(report['flower_version'],policy.VERSION)
        self.assertEqual(report['candidate_implementation_identity'],good_identity())
        self.assertFalse(report['services_started_by_this_gate'])
    def test_preflight_accepts_noniid(self):self.assertEqual(self.invoke(condition='noniid')['status'],'PASS')
    def test_original_version_rejected_before_builder(self):
        with self.assertRaises(IdentityMismatch):self.invoke(version='1.22.0')
        self.builder.assert_not_called()
        r=json.loads(self.report.read_text());self.assertEqual(r['flower_version'],'1.22.0')
        self.assertEqual(r['expected_flower_version'],policy.VERSION);self.assertFalse(r['native_flower_build_executed'])
    def test_source_identity_rejected_before_builder(self):
        identity=good_identity();identity['function_sha256']='0'*64
        with self.assertRaises(IdentityMismatch):self.invoke(identity=identity)
        self.builder.assert_not_called()
    def test_extension_check_not_relaxed(self):
        self.constants.FAB_ALLOWED_EXTENSIONS.add('.json')
        with self.assertRaisesRegex(RuntimeError,'extension policy differs'):self.invoke()
        self.builder.assert_not_called()
    def test_rebuild_byte_equality_not_relaxed(self):
        self.builder.side_effect=[(b'x',hashlib.sha256(b'x').hexdigest(),{}),(b'y',hashlib.sha256(b'y').hexdigest(),{})]
        with self.assertRaisesRegex(ValueError,'not deterministic'):self.invoke()
    def test_digest_check_not_relaxed(self):
        self.builder.return_value=(b'x','0'*64,{})
        with self.assertRaisesRegex(ValueError,'digest mismatch'):self.invoke()
    def test_missing_flower_metadata_written(self):
        with patch.object(self.module.importlib.metadata,'version',side_effect=importlib.metadata.PackageNotFoundError('flwr')):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                self.module.preflight_native_fab(self.app,self.root/'runtime','reference',self.report,{})
        r=json.loads(self.report.read_text());self.assertEqual(r['status'],'FAIL');self.assertFalse(r['native_flower_build_executed'])


class SecAggPreflightEntryStubTests(PreflightEntryStubTests):
    mode='ehr_secagg'

if __name__=='__main__':unittest.main()
