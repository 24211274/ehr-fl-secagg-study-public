"""Zero-length ConfigRecord regressions.

SourceMethodTests execute the upstream count method on a labelled dictionary
harness, not the whole Flower package. Native classes require installed Flower;
the target runner rejects skips. No EHR records, real keys, or model updates here.
"""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'app'), str(ROOT/'tests'), str(ROOT/'validation_support')]
import mai_measurement as mm
from flower_message_fixture import received_message

HAS_FLOWER = importlib.util.find_spec('flwr') is not None
_spec = importlib.util.spec_from_file_location('upstream_counter_excerpt', ROOT/'reference/counter_fix/flower122_config_count_method.py')
_excerpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_excerpt)


class SourceConfigRecord(dict):
    """Minimal test dictionary using the actual upstream METHOD, not Flower."""
    count_bytes = _excerpt.count_bytes
    def __init__(self, config_dict=None, keep_input=True):
        super().__init__(config_dict or {})
        if not keep_input and config_dict:
            config_dict.clear()


class Content:
    def __init__(self, config=None, arrays=None, metrics=None):
        self.config_records = {} if config is None else {'config': config}
        self.array_records = arrays or {}
        self.metric_records = metrics or {}
    def __len__(self):
        return len(self.config_records)+len(self.array_records)+len(self.metric_records)


class Sized:
    def __init__(self, size): self.size=size
    def count_bytes(self): return self.size


def expected_value(value):
    """Independent element-wise specification, including zero-length lists."""
    if isinstance(value, list): return sum(expected_value(v) for v in value)
    if type(value) is bool: return 1
    if type(value) in (int, float): return 8
    if type(value) in (bytes, str): return len(value)
    raise TypeError('Unsupported test value')


def expected_record(mapping):
    return sum(len(k)+expected_value(v) for k,v in mapping.items())


class ConfigCountContract:
    def count(self, fields): return mm.record_bytes(Content(self.Record(fields)))['config_bytes']

    def test_reproduce_original_empty_list_error(self):
        with self.assertRaises(IndexError): self.Record({'dead_nids':[]}).count_bytes()
    def test_reproduce_original_empty_scalar_error(self):
        for value in ('',b''):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError): self.Record({'message':value}).count_bytes()
    def test_empty_list_keeps_key_length(self):
        self.assertEqual(self.count({'dead_nids':[]}), len('dead_nids'))
    def test_scalar_empty_text_keeps_key_length(self):
        self.assertEqual(self.count({'message':''}), len('message'))
    def test_scalar_empty_bytes_keeps_key_length(self):
        self.assertEqual(self.count({'blob':b''}), len('blob'))
    def test_all_empty_fields_and_empty_key(self):
        x={'':[], 'a':[], 'message':'', 'blob':b''}
        self.assertEqual(self.count(x), expected_record(x))
    def test_zero_numeric_false_are_not_removed(self):
        x={'none_missing':[], 'false':False, 'zero_int':0, 'zero_float':0.0}
        self.assertEqual(self.count(x),expected_record(x))
    def test_empty_record_is_zero(self):
        self.assertEqual(self.count({}),0)
    def test_unmask_full_participation_record(self):
        x={'stage':'unmask','active_nids':[11,22,33],'dead_nids':[]}
        self.assertEqual(self.count(x),expected_record(x))
    def test_unmask_nonempty_dropout_list(self):
        x={'stage':'unmask','active_nids':[11,22],'dead_nids':[33]}
        self.assertEqual(self.count(x),self.Record(x).count_bytes())
    def test_nonempty_scalars_and_homogeneous_lists_match_native(self):
        vals=[False,True,0,5,-5,2**62,0.0,1.25,'x','مرحبا',b'\x00',
              [False,True],[0,1],[0.0,1.25],['','x','abc'],[b'',b'x',b'abc']]
        for value in vals:
            with self.subTest(value_type=type(value).__name__):
                x={'a':value}
                self.assertEqual(self.count(x),self.Record(x).count_bytes())
                self.assertEqual(self.count(x),expected_record(x))
    def test_nonempty_lists_of_empty_text_or_bytes_still_count_keys(self):
        for x in ({'texts':['','']},{'bytes':[b'',b'']}):
            self.assertEqual(self.count(x), self.Record(x).count_bytes())
    def test_legacy_unicode_length_convention_retained(self):
        x={'حقل':'é','فارغ':[]}
        self.assertEqual(self.count(x),expected_record(x))
        self.assertNotEqual(self.count(x),sum(len(k.encode('utf8')) for k in x)+len('é'.encode('utf8')))
    def test_mixed_config_retains_all_nonempty_values(self):
        x={'stage':'unmask','gone':[], 'empty':'', 'zero':0, 'false':False,
           'values':[0.2,0.3], 'chunks':[b'',b'public'], 'blob':b''}
        self.assertEqual(self.count(x),expected_record(x))
    def test_no_record_mutation_or_list_identity_change(self):
        record=self.Record({'dead_nids':[],'active_nids':[11,22,33],'status':''})
        before=copy.deepcopy(dict(record)); links={k:id(v) for k,v in record.items()}
        method=self.Record.count_bytes
        self.assertEqual(mm.record_bytes(Content(record))['config_bytes'],expected_record(before))
        self.assertEqual(dict(record),before)
        self.assertEqual({k:id(v) for k,v in record.items()},links)
        self.assertIs(self.Record.count_bytes,method)
        with self.assertRaises(IndexError):record.count_bytes()
    def test_array_metric_count_delegation_unchanged(self):
        x={'empty':[],'stage':'unmask'}
        r=mm.record_bytes(Content(self.Record(x),{'a':Sized(123)},{'m':Sized(27)}))
        self.assertEqual(r,{'array_bytes':123,'metric_bytes':27,'config_bytes':expected_record(x),'total_bytes':150+expected_record(x)})
    def test_repeated_count_is_identical(self):
        record=self.Record({'dead_nids':[],'stage':'unmask'})
        results=[mm.record_bytes(Content(record)) for _ in range(3)]
        self.assertEqual(results,[results[0]]*3)
    def test_unrelated_counter_exception_propagates(self):
        for fields in ({'valid':1},{'empty':[],'valid':1}):
            record=self.Record(fields)
            with patch.object(self.Record,'count_bytes',side_effect=RuntimeError('public-test-error')):
                with self.assertRaises(RuntimeError):mm.record_bytes(Content(record))
    def test_invalid_native_return_is_rejected(self):
        for value in (-1,True,1.5):
            with patch.object(self.Record,'count_bytes',return_value=value):
                with self.assertRaises(ValueError):mm.record_bytes(Content(self.Record({'empty':[]})))
    def test_config_repair_does_not_swallow_metric_errors(self):
        class Bad:
            def count_bytes(self):raise IndexError('public-metric-error')
        with self.assertRaises(IndexError):mm.record_bytes(Content(self.Record({'e':[]}),metrics={'m':Bad()}))
    def test_projection_requires_a_configrecord(self):
        class Impostor(dict):
            def count_bytes(self):return 0
        with self.assertRaises(TypeError):mm.record_bytes(Content(Impostor({'e':[]})))
    def test_report_remains_strict_numeric_and_does_not_export_values(self):
        x={'stage':'unmask','secret_field':b'public-test-secret','gone':[]}
        row=mm.record_bytes(Content(self.Record(x)))
        out=json.dumps(row,allow_nan=False)
        self.assertEqual(set(row),{'array_bytes','metric_bytes','config_bytes','total_bytes'})
        self.assertTrue(all(type(v) is int for v in row.values()))
        self.assertNotIn('public-test-secret',out);self.assertNotIn('secret_field',out)
    def test_multiple_records_keep_all_keys_once(self):
        content=Content()
        content.config_records={'first':self.Record({'empty':[]}), 'second':self.Record({'message':'','x':1})}
        self.assertEqual(mm.record_bytes(content)['config_bytes'],len('empty')+len('message')+len('x')+8)


class SourceMethodTests(ConfigCountContract,unittest.TestCase):
    def setUp(self):
        self.Record=SourceConfigRecord
        flower=types.ModuleType('flwr');common=types.ModuleType('flwr.common')
        common.ConfigRecord=SourceConfigRecord;flower.common=common
        self.patch=patch.dict(sys.modules,{'flwr':flower,'flwr.common':common})
        self.patch.start();self.addCleanup(self.patch.stop)


@unittest.skipUnless(HAS_FLOWER,'Requires installed native Flower; not replaced by source harness')
class NativeCounterTests(ConfigCountContract,unittest.TestCase):
    def setUp(self):
        from flwr.common import ConfigRecord
        self.Record=ConfigRecord


@unittest.skipUnless(HAS_FLOWER,'Requires installed native Flower; not replaced by source harness')
class NativeBoundaryTests(unittest.TestCase):
    def test_deflated_config_record_and_message_stay_unchanged(self):
        from flwr.common import ConfigRecord,RecordDict,Message
        r=ConfigRecord({'stage':'unmask','dead_nids':[],'empty':'','blob':b''})
        msg=Message(RecordDict({'secaggplus_configs':r}),11,'train',group_id='1')
        before=r.deflate();before_message=msg.deflate()
        mm.record_bytes(msg.content)
        self.assertEqual(r.deflate(),before);self.assertEqual(msg.deflate(),before_message)

    def check_boundary(self,secagg):
        from flwr.common import ConfigRecord,RecordDict,Message,Context,MetricRecord
        x=({'stage':'unmask','active_nids':[11,22,33],'dead_nids':[]} if secagg else {'server_round':1,'optional':[]})
        key='secaggplus_configs' if secagg else 'fitins.config'
        request=received_message(Message(RecordDict({key:ConfigRecord(x)}),11,'train',group_id='1'))
        y={'public':'KNOWN_TEST_PAYLOAD','empty':b'','message':'','optional':[]}
        content=RecordDict({'public_response':ConfigRecord(y),'metrics':MetricRecord({'values':[],'n':3})})
        with tempfile.TemporaryDirectory() as td:
            ctx=Context(run_id=1,node_id=11,node_config={'client-id':'C1'},state=RecordDict(),run_config={'runtime-root':td,'condition':'reference'})
            def deliver(messages,timeout=None):
                return [mm.timing_boundary_mod(m,ctx,lambda m,c:Message(copy.deepcopy(content),reply_to=m)) for m in messages]
            observer=mm.ObservedRawGrid(types.SimpleNamespace(send_and_receive=deliver))
            observer.send_and_receive([request])
            row=json.loads(next(Path(td).rglob('*.json')).read_text())
            server=observer.rows[0]['messages'][0]
            self.assertEqual(row['incoming_record_bytes'],server['outgoing_record_bytes'])
            self.assertEqual(row['outgoing_record_bytes'],server['incoming_record_bytes'])
            self.assertEqual(row['incoming_record_bytes']['config_bytes'],expected_record(x))
            self.assertEqual(row['outgoing_record_bytes']['config_bytes'],expected_record(y))
            self.assertEqual(row['byte_counting_rule_id'],mm.BYTE_COUNTING_RULE_ID)
            self.assertNotIn('KNOWN_TEST_PAYLOAD',json.dumps(row))
    def test_unmask_two_boundaries_reconcile_with_empty_list(self):self.check_boundary(True)
    def test_B_two_boundaries_reconcile_with_empty_status(self):self.check_boundary(False)
    def test_metric_empty_list_delegates_native(self):
        from flwr.common import RecordDict,MetricRecord,ConfigRecord
        m=MetricRecord({'empty':[],'zero':0});c=ConfigRecord({'empty':[]})
        result=mm.record_bytes(RecordDict({'m':m,'c':c}))
        self.assertEqual(result['metric_bytes'],m.count_bytes())
        self.assertEqual(result['config_bytes'],len('empty'))
    def test_installed_count_method_is_unchanged(self):
        from flwr.common import ConfigRecord
        fn=ConfigRecord.count_bytes
        mm.record_bytes(RecordDictFor(ConfigRecord({'empty':[]})))
        self.assertIs(ConfigRecord.count_bytes,fn)
        with self.assertRaises(IndexError):ConfigRecord({'empty':[]}).count_bytes()

    def test_off_lattice_public_protocol_observed_at_all_stages(self):
        from flwr.client import ClientApp
        from prg_validation.public_protocol import run_case
        with tempfile.TemporaryDirectory() as td:
            class Observed(ClientApp):
                def __call__(self,msg,ctx):
                    msg=received_message(msg)
                    ctx.node_config['client-id']='C'+str(int(ctx.node_config['toy-index'])+1)
                    ctx.run_config.update({'runtime-root':td,'condition':'reference'})
                    return mm.timing_boundary_mod(msg,ctx,super().__call__)
            with patch('flwr.client.ClientApp',Observed):result=run_case([256,256,256],lattice=False,mode='candidate')
            rows=[json.loads(p.read_text()) for p in Path(td).rglob('*.json')]
            self.assertEqual(result['status'],'PASS');self.assertEqual(len(rows),12)
            self.assertEqual(sum(r['stage']=='unmask' for r in rows),3)
            self.assertTrue(all(r['status']=='PASS' and r['record_values_written'] is False for r in rows))
            self.assertTrue(all(r['outgoing_record_bytes']['array_bytes']==0 for r in rows))


def RecordDictFor(r):
    from flwr.common import RecordDict
    return RecordDict({'c':r})


class PackagingTests(unittest.TestCase):
    def test_no_global_monkeypatch_in_production(self):
        text=(ROOT/'app/mai_measurement.py').read_text()
        self.assertNotIn('ConfigRecord.count_bytes =',text)
        self.assertNotIn('except IndexError',text)
    def test_source_method_is_exact_except_dedent(self):
        tree=ast.parse((ROOT/'reference/counter_fix/flower122_config_count_method.py').read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        self.assertEqual([n.name for n in functions],['count_bytes'])
        self.assertIn('v[0]',ast.unparse(functions[0]))
    def test_server_report_declares_rule(self):
        mm.reset_server();report=mm.server_report(types.SimpleNamespace(rows=[]))
        self.assertEqual(report['byte_counting_rule_id'],mm.BYTE_COUNTING_RULE_ID)
        self.assertFalse(report['wire_bytes_measured'])
        json.dumps(report,allow_nan=False)


if __name__=='__main__':unittest.main()
