"""New identity gates. Stand-in transport tests are NOT live Flower evidence."""
from __future__ import annotations
import copy,hashlib,importlib.util,json,sys,types,unittest,tempfile,uuid
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'));sys.path.insert(0,str(ROOT))
import mai_runtime_identity as ident
from flower_message_fixture import received_message
HAS_FLOWER=importlib.util.find_spec('flwr') is not None

class Record(dict):pass
class DictRecord:
    def __init__(self,records=None):self.config_records=dict(records or {});self.array_records={};self.metric_records={}
class Msg:
    def __init__(self,content=None,dst_node_id=None,message_type=None,group_id=None,ttl=None,reply_to=None):
        self.content=content;self.error_flag=False
        if reply_to is None:
            self.metadata=types.SimpleNamespace(dst_node_id=dst_node_id,src_node_id=0,message_type=message_type)
        else:
            self.metadata=types.SimpleNamespace(dst_node_id=reply_to.metadata.src_node_id,src_node_id=reply_to.metadata.dst_node_id,message_type=reply_to.metadata.message_type)
    def has_error(self):return self.error_flag

def mock_common():
    root=types.ModuleType('flwr');root.__spec__=importlib.util.spec_from_loader('flwr',loader=None)
    module=types.ModuleType('flwr.common');module.Message=Msg;module.RecordDict=DictRecord;module.ConfigRecord=Record
    root.common=module;return {'flwr':root,'flwr.common':module}

def binding_stub():return {'status':'PASS','build_id':ident.EXPECTED_BUILD_ID}

class IdentitySchemaTests(unittest.TestCase):
    def test_valid_request(self):ident.require_request({'build_id':ident.EXPECTED_BUILD_ID})
    def test_old_id_rejected(self):
        with self.assertRaises(ident.IdentityMismatch):ident.require_request({'build_id':ident.OLD_BUILD_ID})
    def test_missing_id_rejected(self):
        with self.assertRaises(ident.IdentityMismatch):ident.require_request(None)
    def test_extra_request_field_rejected(self):
        with self.assertRaises(ident.IdentityMismatch):ident.require_request({'build_id':ident.EXPECTED_BUILD_ID,'output_digest':'x'})
    def test_reply_client_required(self):
        with self.assertRaises(ident.IdentityMismatch):ident.require_reply({'build_id':ident.EXPECTED_BUILD_ID})
    def test_unknown_reply_client(self):
        with self.assertRaises(ident.IdentityMismatch):ident.require_reply({'build_id':ident.EXPECTED_BUILD_ID,'client':'C4'})
    def test_update_hash_in_identity_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.require_reply({'build_id':ident.EXPECTED_BUILD_ID,'client':'C1','update_hash':'x'})
    def test_valid_reply(self):ident.require_reply({'build_id':ident.EXPECTED_BUILD_ID,'client':'C1'})
    def test_candidate_identity_links_to_validated_source(self):
        m=json.loads((ROOT/'validation_support/CHANGE_MANIFEST.json').read_text())
        self.assertEqual(ident.FUNCTION_SHA256,m['new_function_sha256']);self.assertEqual(ident.HELPER_SHA256,m['sources']['backport_secaggplus_utils.py'])
    def test_candidate_source_unmodified(self):
        data=(ROOT/'validation_support/sources/backport_secaggplus_utils.py').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),ident.HELPER_SHA256)
    def test_old_new_identifiers_distinct(self):self.assertNotEqual(ident.OLD_BUILD_ID,ident.EXPECTED_BUILD_ID)
    def test_fullsize_configuration_keeps_crypto_identity(self):
        c=json.loads((ROOT/'config/timing_B.json').read_text())
        self.assertEqual(c['input_dim'],119)
        self.assertEqual(c['required_packages']['flwr'],ident.VERSION)
        self.assertEqual(c['rounds'],6)
        self.assertEqual(c['local_epochs'],1)


class IdentityFlowStandinTests(unittest.TestCase):
    def setUp(self):
        self.p=patch.dict(sys.modules,mock_common());self.p.start()
        self.b=patch.object(ident,'binding_identity',side_effect=binding_stub);self.b.start()
        self.context=types.SimpleNamespace(node_config={'client-id':'C1'})
        self.calls=0
    def tearDown(self):self.b.stop();self.p.stop()
    def message(self,kind='train',build=ident.EXPECTED_BUILD_ID):
        return Msg(DictRecord({ident.ID_RECORD:Record({'build_id':build})}),1,kind)
    def callback(self,m,c):
        self.calls+=1
        self.assertNotIn(ident.ID_RECORD,m.content.config_records)
        return Msg(DictRecord({'safe':Record({'value':1})}),reply_to=m)
    def test_valid_fit_reaches_callback_and_adds_only_public_identity(self):
        m=self.message();r=ident.implementation_identity_mod(m,self.context,self.callback)
        self.assertEqual(self.calls,1);ident.require_reply(dict(r.content.config_records[ident.ID_RECORD]))
    def test_wrong_fit_never_reaches_callback(self):
        with self.assertRaises(ident.IdentityMismatch):ident.implementation_identity_mod(self.message(build=ident.OLD_BUILD_ID),self.context,self.callback)
        self.assertEqual(self.calls,0)
    def test_missing_fit_id_rejected_before_callback(self):
        m=self.message();m.content.config_records.clear()
        with self.assertRaises(ident.IdentityMismatch):ident.implementation_identity_mod(m,self.context,self.callback)
        self.assertEqual(self.calls,0)
    def test_wrong_evaluate_never_reaches_callback(self):
        with self.assertRaises(ident.IdentityMismatch):ident.implementation_identity_mod(self.message('evaluate',ident.OLD_BUILD_ID),self.context,self.callback)
        self.assertEqual(self.calls,0)
    def test_positive_query_does_not_load_client(self):
        r=ident.implementation_identity_mod(self.message(ident.QUERY_TYPE),self.context,self.callback)
        self.assertTrue(r.content.config_records[ident.PROBE_RECORD]['accepted']);self.assertEqual(self.calls,0)
    def test_negative_query_does_not_load_client(self):
        r=ident.implementation_identity_mod(self.message(ident.QUERY_TYPE,ident.OLD_BUILD_ID),self.context,self.callback)
        self.assertFalse(r.content.config_records[ident.PROBE_RECORD]['accepted']);self.assertEqual(self.calls,0)
    def test_query_other_fields_refused(self):
        m=self.message(ident.QUERY_TYPE);m.content.config_records['bad']=Record({'x':1})
        with self.assertRaises(ident.IdentityMismatch):ident.implementation_identity_mod(m,self.context,self.callback)
    def test_query_array_refused(self):
        m=self.message(ident.QUERY_TYPE);m.content.array_records['bad']=[1]
        with self.assertRaises(ident.IdentityMismatch):ident.implementation_identity_mod(m,self.context,self.callback)
    def test_failed_binding_never_reaches_callback(self):
        with patch.object(ident,'binding_identity',side_effect=ident.IdentityMismatch('wrong actual source')):
            with self.assertRaises(ident.IdentityMismatch):ident.implementation_identity_mod(self.message(),self.context,self.callback)
        self.assertEqual(self.calls,0)
    def transport(self,drop=False,duplicate=False,forge=False):
        parent=self
        class LocalStandin:
            def get_node_ids(self):return [1,2,3]
            def send_and_receive(self,messages,*,timeout=None):
                replies=[]
                for m in messages:
                    if drop and m.metadata.dst_node_id==3:continue
                    cid='C1' if duplicate else 'C'+str(m.metadata.dst_node_id)
                    ctx=types.SimpleNamespace(node_config={'client-id':cid})
                    r=ident.implementation_identity_mod(copy.deepcopy(m),ctx,parent.callback)
                    if forge and m.metadata.message_type==ident.QUERY_TYPE:r.content.config_records[ident.PROBE_RECORD]['accepted']=True
                    replies.append(r)
                return replies
        return LocalStandin()
    def test_handshake_six_messages_no_training(self):
        report=ident.network_handshake(self.transport(),2)
        ident.verify_handshake_report(report);self.assertEqual(self.calls,0)
    def test_handshake_missing_reply_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.network_handshake(self.transport(drop=True),2)
    def test_handshake_duplicate_logical_client_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.network_handshake(self.transport(duplicate=True),2)
    def test_handshake_unconditional_accept_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.network_handshake(self.transport(forge=True),2)
    def test_shared_setup_content_not_mutated(self):
        common=DictRecord({'secaggplus_configs':Record({'stage':'setup'})})
        originals=[Msg(common,n,'train') for n in [1,2,3]]
        transport=ident.IdentityCheckedTransport(self.transport())
        replies=transport.send_and_receive(originals,timeout=2)
        self.assertNotIn(ident.ID_RECORD,common.config_records)
        self.assertEqual(len(replies),3);self.assertEqual(self.calls,3)
        self.assertTrue(all(ident.ID_RECORD not in r.content.config_records for r in replies))
        self.assertTrue(all(set(r.content.config_records)=={'safe'} for r in replies))
    def test_transport_missing_reply_refused(self):
        tr=ident.IdentityCheckedTransport(self.transport(drop=True));m=[Msg(DictRecord(),n,'train') for n in [1,2,3]]
        with self.assertRaises(ident.IdentityMismatch):tr.send_and_receive(m,timeout=2)
    def test_transport_empty_request_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.IdentityCheckedTransport(self.transport()).send_and_receive([],timeout=2)
    def test_transport_duplicate_destinations_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.IdentityCheckedTransport(self.transport()).send_and_receive([Msg(DictRecord(),1,'train')]*2,timeout=2)
    def test_transport_existing_identity_refused(self):
        with self.assertRaises(ident.IdentityMismatch):ident.IdentityCheckedTransport(self.transport()).send_and_receive([self.message()],timeout=2)
    def test_handshake_tampered_claim_rejected(self):
        r=ident.network_handshake(self.transport(),2);r['remote_attestation']=True
        with self.assertRaises(ident.IdentityMismatch):ident.verify_handshake_report(r)

@unittest.skipUnless(HAS_FLOWER,'Native Flower identity checks must run in the target full candidate environment')
class NativeIdentityTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory(prefix='identity counter fixture ')
    def tearDown(self):self.tmp.cleanup()
    def context(self,n=1):
        from flwr.common import Context,RecordDict
        return Context(run_id=77,node_id=n,node_config={'client-id':'C'+str(n)},state=RecordDict(),run_config={'runtime-root':self.tmp.name,'condition':'reference'})
    def msg(self,n=1,build=ident.EXPECTED_BUILD_ID,kind=ident.QUERY_TYPE,content=None):
        from flwr.common import Message,ConfigRecord,RecordDict
        m=Message(content=content or RecordDict({ident.ID_RECORD:ConfigRecord({'build_id':build})}),dst_node_id=n,message_type=kind,ttl=60)
        return received_message(m)
    def test_real_candidate_binding(self):self.assertEqual(ident.binding_identity()['build_id'],ident.EXPECTED_BUILD_ID)
    def test_real_B_client_identity_query_without_dataset(self):
        from ehr_fedavg.client_app import app
        r=app(self.msg(),self.context());self.assertTrue(r.content.config_records[ident.PROBE_RECORD]['accepted'])
    def test_real_C_client_identity_query_without_dataset(self):
        from ehr_secagg.client_app import app
        r=app(self.msg(),self.context());self.assertTrue(r.content.config_records[ident.PROBE_RECORD]['accepted'])
    def test_real_C_wrong_id_query_refused_without_dataset(self):
        from ehr_secagg.client_app import app
        r=app(self.msg(build=ident.OLD_BUILD_ID),self.context());self.assertFalse(r.content.config_records[ident.PROBE_RECORD]['accepted'])
    def test_real_C_wrong_train_id_refused_before_mod(self):
        from ehr_secagg.client_app import app
        with self.assertRaises(ident.IdentityMismatch):app(self.msg(build=ident.OLD_BUILD_ID,kind='train'),self.context())
    def test_real_message_record_handshake_in_memory(self):
        from ehr_secagg.client_app import app
        parent=self
        class Network:
            def get_node_ids(self):return [1,2,3]
            def send_and_receive(self,messages,*,timeout=None):
                messages=list(messages)
                messages=[received_message(m) for m in messages]
                return [app(m,parent.context(m.metadata.dst_node_id)) for m in messages]
        r=ident.network_handshake(Network(),30);ident.verify_handshake_report(r)
    def test_real_shared_record_and_C_boundary(self):
        from flwr.common import Message,ConfigRecord,RecordDict
        from ehr_secagg.wire import check_masked_reply
        parent=self
        def safe_callback(m,ctx):
            content=RecordDict({'secaggplus_configs':ConfigRecord({'pk1':b'public_fixture_1','pk2':b'public_fixture_2'})})
            return Message(content=content,reply_to=m)
        class Network:
            def send_and_receive(self,messages,*,timeout=None):return [ident.implementation_identity_mod(m,parent.context(m.metadata.dst_node_id),safe_callback) for m in messages]
        shared=RecordDict({'secaggplus_configs':ConfigRecord({'stage':'setup'})})
        sent=[Message(content=shared,dst_node_id=n,message_type='train',ttl=30) for n in [1,2,3]]
        replies=ident.IdentityCheckedTransport(Network()).send_and_receive(sent,timeout=20)
        for r in replies:check_masked_reply(r.content,'setup')
        self.assertNotIn(ident.ID_RECORD,shared.config_records)

if __name__=='__main__':unittest.main()
