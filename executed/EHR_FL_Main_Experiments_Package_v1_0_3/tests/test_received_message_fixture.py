"""Delivery-fixture regressions; stand-ins are not Flower protocol evidence."""
from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from flower_message_fixture import received_message

ROOT = Path(__file__).resolve().parents[1]
HAS_FLOWER = importlib.util.find_spec('flwr') is not None
FIELDS = ('run_id', 'src_node_id', 'dst_node_id', 'reply_to_message_id',
          'group_id', 'created_at', 'ttl', 'message_type', 'delivered_at')


class ReadOnlyIdMetadata:
    """Strict stand-in: ID property is read-only, as in pinned Flower 1.22."""
    def __init__(self, **kwargs):
        self._id = kwargs.pop('message_id')
        self.__dict__.update(kwargs)
        self.delivered_at = ''

    @property
    def message_id(self):
        return self._id


class ConstructorMessage:
    """Stand-in supporting ONLY the internal metadata constructor used in tests."""
    def __init__(self, *, metadata, content=None, error=None):
        self.metadata = metadata
        self.content = content
        self.error = error
        self.metadata.delivered_at = ''

    def has_error(self):
        return self.error is not None

    def has_content(self):
        return self.content is not None


def strict_modules():
    root = types.ModuleType('flwr')
    root.__spec__ = importlib.util.spec_from_loader('flwr', loader=None)
    common = types.ModuleType('flwr.common')
    common.Message = ConstructorMessage
    common.Metadata = ReadOnlyIdMetadata
    root.common = common
    return {'flwr': root, 'flwr.common': common}


def standin_request(message_id='', reply_to='', content=None, error=None):
    md = ReadOnlyIdMetadata(run_id=77, message_id=message_id,
        src_node_id=12, dst_node_id=21, reply_to_message_id=reply_to,
        group_id='3', created_at=1_777_777_777.25, ttl=60.0,
        message_type='train')
    msg = ConstructorMessage(metadata=md,
        content={'public': {'x': [1, 2]}} if content is None else content,
        error=error)
    msg.metadata.delivered_at = 'test-delivered'
    return msg


class FixtureStandinTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, strict_modules())
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_original_failure_is_reproduced_by_read_only_contract(self):
        with self.assertRaisesRegex(AttributeError, 'no setter'):
            standin_request().metadata.message_id = 'wrong-assignment'

    def test_constructs_received_copy_with_explicit_id(self):
        sent = standin_request()
        got = received_message(sent, message_id='test-id')
        self.assertIsNot(sent, got)
        self.assertIsNot(sent.metadata, got.metadata)
        self.assertEqual(sent.metadata.message_id, '')
        self.assertEqual(got.metadata.message_id, 'test-id')

    def test_preserves_all_non_id_metadata(self):
        sent = standin_request()
        got = received_message(sent)
        for key in FIELDS:
            self.assertEqual(getattr(sent.metadata, key), getattr(got.metadata, key))

    def test_request_payload_is_equal_and_not_shared(self):
        sent = standin_request()
        got = received_message(sent)
        self.assertEqual(sent.content, got.content)
        got.content['public']['x'].append(3)
        self.assertEqual(sent.content['public']['x'], [1, 2])

    def test_distinct_new_deliveries_get_distinct_ids(self):
        sent = standin_request()
        ids = [received_message(sent).metadata.message_id for _ in range(10)]
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(sent.metadata.message_id, '')

    def test_existing_id_is_preserved_not_regenerated(self):
        sent = standin_request('fixed-id')
        self.assertEqual(received_message(sent).metadata.message_id, 'fixed-id')
        self.assertEqual(received_message(sent, message_id='fixed-id').metadata.message_id, 'fixed-id')

    def test_cannot_overwrite_existing_id(self):
        with self.assertRaisesRegex(ValueError, 'must not be overwritten'):
            received_message(standin_request('fixed-id'), message_id='different-id')

    def test_invalid_explicit_ids_are_rejected(self):
        for value in ['', ' ', 3, True, b'id']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                received_message(standin_request(), message_id=value)

    def test_unexpected_existing_id_is_rejected(self):
        with self.assertRaises(TypeError):
            received_message(standin_request(3))
        with self.assertRaises(ValueError):
            received_message(standin_request(' '))

    def test_reply_is_not_relabelled_as_instruction(self):
        with self.assertRaises(ValueError):
            received_message(standin_request(reply_to='prior-id'))

    def test_error_or_no_content_message_is_rejected(self):
        sent = standin_request(error=object())
        with self.assertRaises(ValueError):
            received_message(sent)
        sent = standin_request(); sent.content = None
        with self.assertRaises(ValueError):
            received_message(sent)

    def test_non_native_input_is_rejected(self):
        with self.assertRaises(TypeError):
            received_message(object())


class FixtureSourceTests(unittest.TestCase):
    def test_fixture_not_imported_by_runtime_or_deployment_code(self):
        for dirname in ('app', 'scripts', 'timing_support', 'fullsize_support', 'validation_support'):
            for p in (ROOT/dirname).rglob('*.py'):
                self.assertNotIn('flower_message_fixture', p.read_text(), str(p))

    def test_original_assignment_sites_removed_from_native_tests(self):
        for name in ('test_integration_identity.py', 'test_measurement.py'):
            tree = ast.parse((ROOT/'tests'/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self.assertFalse(isinstance(target, ast.Attribute) and target.attr == 'message_id')

    def test_fixture_does_not_write_readonly_or_private_metadata(self):
        src = (ROOT/'tests/flower_message_fixture.py').read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        self.assertEqual(target.attr, 'delivered_at')
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ('__dict__', '_message_id', '_metadata'))
        self.assertIn('Message(metadata=metadata, content=', src)


@unittest.skipUnless(HAS_FLOWER, 'Native metadata regressions require installed candidate Flower')
class NativeFixtureTests(unittest.TestCase):
    def request(self, message_id='', *, group_id='3', kind='train'):
        from flwr.common import Message, Metadata, RecordDict, ConfigRecord
        md = Metadata(run_id=101, message_id=message_id, src_node_id=1,
            dst_node_id=11, reply_to_message_id='', group_id=group_id,
            created_at=time.time(), ttl=60.0, message_type=kind)
        return Message(metadata=md, content=RecordDict({'public': ConfigRecord({'x': [1, 2]})}))

    def test_native_read_only_property_stays_read_only(self):
        from flwr.common import Metadata
        self.assertIsNone(Metadata.message_id.fset)
        with self.assertRaisesRegex(AttributeError, 'no setter'):
            self.request().metadata.message_id = 'disallowed'
        self.assertIsNone(Metadata.message_id.fset)

    def test_native_internal_constructor_preserves_metadata(self):
        source = self.request()
        source.metadata.delivered_at = 'public-fixture-delivery'
        got = received_message(source, message_id='public-native-id')
        for key in FIELDS:
            self.assertEqual(getattr(source.metadata, key), getattr(got.metadata, key))
        self.assertEqual(source.metadata.message_id, '')
        self.assertEqual(got.metadata.message_id, 'public-native-id')

    def test_native_reply_is_correlated_with_delivered_request(self):
        from flwr.common import Message, RecordDict
        sent = received_message(self.request(), message_id='correlation-id')
        reply = Message(content=RecordDict(), reply_to=sent)
        self.assertEqual(reply.metadata.reply_to_message_id, 'correlation-id')
        self.assertEqual(reply.metadata.src_node_id, sent.metadata.dst_node_id)
        self.assertEqual(reply.metadata.dst_node_id, sent.metadata.src_node_id)
        self.assertEqual(reply.metadata.run_id, sent.metadata.run_id)
        self.assertEqual(reply.metadata.group_id, sent.metadata.group_id)

    def test_native_payload_copy_does_not_share_records(self):
        source = self.request()
        got = received_message(source)
        got.content.config_records['public']['x'] = [9]
        self.assertEqual(source.content.config_records['public']['x'], [1, 2])

    def test_native_existing_id_cannot_be_changed(self):
        source = self.request('preassigned-id')
        self.assertEqual(received_message(source).metadata.message_id, 'preassigned-id')
        with self.assertRaises(ValueError):
            received_message(source, message_id='replacement-id')

    def test_native_duplicate_counter_is_still_rejected(self):
        from flwr.common import Message, RecordDict, Context
        import mai_measurement as mm
        calls = []
        with tempfile.TemporaryDirectory() as td:
            ctx = Context(run_id=101, node_id=11, node_config={'client-id': 'C1'},
                state=RecordDict(), run_config={'runtime-root': td, 'condition': 'reference'})
            sent = received_message(self.request(kind='query.mai_prg_identity'), message_id='duplicate-id')
            def callback(m, c):
                calls.append(1)
                return Message(content=RecordDict(), reply_to=m)
            mm.timing_boundary_mod(sent, ctx, callback)
            with self.assertRaisesRegex(ValueError, 'Repeated request'):
                mm.timing_boundary_mod(received_message(sent), ctx, callback)
            self.assertEqual(calls, [1])
            self.assertEqual(len(list(Path(td).rglob('*.json'))), 1)

    def test_native_missing_id_not_fabricated_by_production_counter(self):
        from flwr.common import RecordDict, Context
        import mai_measurement as mm
        calls = []
        with tempfile.TemporaryDirectory() as td:
            ctx = Context(run_id=101, node_id=11, node_config={'client-id': 'C1'},
                state=RecordDict(), run_config={'runtime-root': td, 'condition': 'reference'})
            with self.assertRaisesRegex(ValueError, 'must have a message ID'):
                mm.timing_boundary_mod(self.request(kind='query.mai_prg_identity'), ctx,
                    lambda m, c: calls.append(1))
            self.assertEqual(calls, [])
            self.assertEqual(len(list(Path(td).rglob('*.json'))), 0)

    def test_native_reply_not_accepted_as_test_instruction(self):
        from flwr.common import Message, RecordDict
        req = received_message(self.request())
        reply = Message(content=RecordDict(), reply_to=req)
        with self.assertRaises(ValueError):
            received_message(reply)


if __name__ == '__main__':
    unittest.main()
