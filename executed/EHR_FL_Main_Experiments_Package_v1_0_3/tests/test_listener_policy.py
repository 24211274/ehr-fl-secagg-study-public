"""Listener guard regression tests; sockets are local engineering probes, not FL."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import ipaddress
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil
from ehr_fedavg.host import OwnedProcesses, local_environment, ensure_free_ports
from ehr_fedavg.listener_policy import classify_listener_address


def connection(address, port=19091, status=psutil.CONN_LISTEN):
    return SimpleNamespace(status=status, laddr=SimpleNamespace(ip=address,port=port),
                           family=socket.AF_INET6 if ':' in str(address) else socket.AF_INET)


def fake_process(pid, rows=None, error=None):
    def net_connections(kind):
        if error is not None:
            raise error
        return rows or []
    return SimpleNamespace(pid=pid,net_connections=net_connections)


class AddressClassificationTests(unittest.TestCase):
    def test_ipv4_loopback_kept(self):
        a=classify_listener_address('127.0.0.1')
        self.assertTrue(a['allowed']);self.assertEqual(a['normalized_address'],'127.0.0.1')
        self.assertFalse(a['ipv4_mapped'])

    def test_ipv6_loopback_forms(self):
        for value in ['::1','0:0:0:0:0:0:0:1','0000:0000:0000:0000:0000:0000:0000:0001']:
            with self.subTest(value=value):
                a=classify_listener_address(value)
                self.assertTrue(a['allowed']);self.assertEqual(a['normalized_address'],'::1')

    def test_mapped_loopback_forms(self):
        for value in ['::ffff:127.0.0.1','::ffff:7f00:1','0:0:0:0:0:ffff:7f00:1','::FFFF:7F00:0001']:
            with self.subTest(value=value):
                a=classify_listener_address(value)
                self.assertTrue(a['allowed']);self.assertTrue(a['ipv4_mapped'])
                self.assertEqual(a['normalized_address'],'127.0.0.1')

    def test_reproduces_old_literal_comparison_false_positive(self):
        value='::ffff:127.0.0.1'
        self.assertNotIn(value,('127.0.0.1','::1'))
        self.assertTrue(classify_listener_address(value)['allowed'])

    def test_works_even_with_pre_fix_ipv6_is_loopback(self):
        # Reproduce the Python 3.12.3 mapped-address behaviour without requiring
        # a different interpreter. The implementation normalises explicitly.
        with patch.object(ipaddress.IPv6Address,'is_loopback',new=property(lambda self:False)):
            self.assertTrue(classify_listener_address('::ffff:127.0.0.1')['allowed'])
            self.assertTrue(classify_listener_address('::1')['allowed'])

    def test_wildcards_blocked(self):
        for value in ['0.0.0.0','::','0:0:0:0:0:0:0:0','::ffff:0.0.0.0','::ffff:0000:0000']:
            with self.subTest(value=value):
                a=classify_listener_address(value)
                self.assertFalse(a['allowed']);self.assertEqual(a['classification'],'wildcard_not_allowed')

    def test_private_public_linklocal_and_multicast_blocked(self):
        for value in ['192.168.1.2','10.0.0.1','172.16.0.2','169.254.1.1','8.8.8.8',
                      '224.0.0.1','255.255.255.255','fe80::1','fc00::1','ff02::1','2001:db8::1']:
            with self.subTest(value=value):self.assertFalse(classify_listener_address(value)['allowed'])

    def test_mapped_nonloopback_blocked(self):
        for value in ['::ffff:192.168.1.2','::ffff:10.0.0.1','::ffff:8.8.8.8','::ffff:c0a8:0102']:
            with self.subTest(value=value):self.assertFalse(classify_listener_address(value)['allowed'])

    def test_allowlist_not_widened_to_other_loopback_addresses(self):
        for value in ['127.0.0.2','127.1.2.3','::ffff:127.0.0.2']:
            with self.subTest(value=value):self.assertFalse(classify_listener_address(value)['allowed'])

    def test_other_ipv6_translation_formats_not_allowed(self):
        for value in ['::127.0.0.1','64:ff9b::7f00:1','2002:7f00:1::']:
            with self.subTest(value=value):self.assertFalse(classify_listener_address(value)['allowed'])

    def test_invalid_and_nonnumeric_addresses_fail_closed(self):
        for value in [None,1,True,b'127.0.0.1','', 'localhost', '*', '[::1]',
                      '127.0.0.1:19091','127.0.0.1/8',' 127.0.0.1','127.0.0.1 ',
                      '127.000.000.001','::1%lo','::ffff:127.0.0.1%1']:
            with self.subTest(value=value):self.assertFalse(classify_listener_address(value)['allowed'])

    def test_no_dns_resolution_used(self):
        with patch('socket.getaddrinfo',side_effect=AssertionError('No DNS allowed')):
            self.assertFalse(classify_listener_address('localhost')['allowed'])
            self.assertTrue(classify_listener_address('::ffff:7f00:1')['allowed'])


class ListenerAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='listener audit ')
        self.root=Path(self.temp.name)
        self.manager=OwnedProcesses(self.root/'logs')
        self.audit=self.root/'reports/reference/listeners.json'

    def tearDown(self):self.temp.cleanup()

    def check(self, processes, ports=None):
        with patch.object(self.manager,'own_psutil_processes',return_value=processes):
            return self.manager.check_listeners(ports or [19091],self.audit)

    def read(self):return json.loads(self.audit.read_text())

    def test_mapped_listener_records_raw_and_effective(self):
        self.check([fake_process(101,[connection('::ffff:127.0.0.1')])])
        a=self.read();self.assertEqual(a['status'],'PASS');self.assertTrue(a['loopback_only'])
        self.assertEqual(a['listeners'][0]['address'],'::ffff:127.0.0.1')
        self.assertEqual(a['listeners'][0]['normalized_address'],'127.0.0.1')
        self.assertEqual(a['ipv4_mapped_loopback_listeners'],1)

    def test_wildcard_rejected_and_saved_before_raise(self):
        with self.assertRaisesRegex(RuntimeError,'address=0.0.0.0 port=19091'):
            self.check([fake_process(101,[connection('0.0.0.0')])])
        a=self.read();self.assertEqual(a['status'],'FAIL');self.assertIsNone(a['loopback_only'])
        self.assertEqual(len(a['blocked_listeners']),1)

    def test_extra_unsafe_port_blocks_even_when_required_ports_safe(self):
        with self.assertRaises(RuntimeError):
            self.check([fake_process(101,[connection('127.0.0.1'),connection('::',20001)])])
        a=self.read();self.assertEqual(a['missing_required_ports'],[])
        self.assertEqual(a['blocked_listeners'][0]['port'],20001)

    def test_private_network_not_treated_as_localhost(self):
        with self.assertRaises(RuntimeError):self.check([fake_process(101,[connection('192.168.1.5')])])
        self.assertEqual(self.read()['status'],'FAIL')

    def test_all_owned_processes_inspected_not_only_first(self):
        with self.assertRaises(RuntimeError):
            self.check([fake_process(101,[connection('0.0.0.0')]),
                        fake_process(102,[connection('127.0.0.1',19092)])],[19091,19092])
        self.assertEqual(len(self.read()['listeners']),2)

    def test_missing_required_port_stays_failure(self):
        with self.assertRaisesRegex(RuntimeError,'Required ports'):
            self.check([fake_process(101,[connection('127.0.0.1')])],[19091,19092])
        self.assertEqual(self.read()['missing_required_ports'],[19092])

    def test_unowned_listener_cannot_satisfy_required_port(self):
        with self.assertRaises(RuntimeError):self.check([])
        self.assertEqual(self.read()['missing_required_ports'],[19091])

    def test_access_denied_does_not_pass_silently(self):
        with self.assertRaisesRegex(RuntimeError,'Could not inspect'):
            self.check([fake_process(101,[connection('127.0.0.1')]),
                        fake_process(102,error=psutil.AccessDenied(102))])
        a=self.read();self.assertEqual(a['status'],'FAIL');self.assertEqual(len(a['inspection_errors']),1)

    def test_os_error_does_not_pass_silently(self):
        with self.assertRaises(RuntimeError):self.check([fake_process(101,error=OSError('test failure'))])
        self.assertEqual(self.read()['inspection_errors'][0]['error_type'],'OSError')

    def test_vanished_required_service_cannot_pass(self):
        with self.assertRaises(RuntimeError):self.check([fake_process(101,error=psutil.NoSuchProcess(101))])
        self.assertEqual(self.read()['processes_vanished_during_scan'],[101])

    def test_connected_sockets_not_mistaken_for_listeners(self):
        self.check([fake_process(101,[connection('127.0.0.1'),connection('192.168.0.10',4321,'ESTABLISHED')])])
        self.assertEqual(len(self.read()['listeners']),1)

    def test_failed_periodic_scan_replaces_previous_pass(self):
        self.check([fake_process(101,[connection('127.0.0.1')])])
        with self.assertRaises(RuntimeError):self.check([fake_process(101,[connection('::')])])
        self.assertEqual(self.read()['status'],'FAIL')
        self.assertFalse(self.audit.with_suffix('.json.tmp').exists())

    def test_invalid_required_port_sets_rejected(self):
        for ports in [[],[True],[0],[65536],[19091,19091],['19091']]:
            with self.subTest(ports=ports):
                with self.assertRaises(ValueError):self.manager.check_listeners(ports,self.audit)

    def test_original_call_signature_remains_supported(self):
        with patch.object(self.manager,'own_psutil_processes',return_value=[fake_process(101,[connection('127.0.0.1')])]):
            self.assertEqual(len(self.manager.check_listeners([19091])),1)
        self.assertFalse(self.audit.exists())

    def test_unwritable_audit_path_stops_instead_of_claiming_pass(self):
        self.audit.parent.mkdir(parents=True);self.audit.mkdir()
        with self.assertRaises(OSError):self.check([fake_process(101,[connection('127.0.0.1')])])

    def test_invalid_socket_address_is_recorded_and_rejected(self):
        with self.assertRaises(RuntimeError):self.check([fake_process(101,[connection(None)])])
        self.assertEqual(self.read()['blocked_listeners'][0]['classification'],'invalid_numeric_address')


@unittest.skipUnless(sys.platform.startswith('linux') and importlib.util.find_spec('grpc'),
                     'Requires Linux and installed grpcio; no installation performed')
class RealGrpcListenerTest(unittest.TestCase):
    def test_real_grpc_local_bind_is_inspected_and_cleaned(self):
        # This starts an empty gRPC server on a temporary loopback port. It does
        # not start Flower or open any dataset, and cannot validate FL training.
        with tempfile.TemporaryDirectory(prefix='grpc localhost test ') as d:
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
            root=Path(d);m=OwnedProcesses(root/'logs')
            code='''import grpc,time
from concurrent.futures import ThreadPoolExecutor
s=grpc.server(ThreadPoolExecutor(max_workers=1))
assert s.add_insecure_port("127.0.0.1:%d")
s.start()
time.sleep(30)
'''%port
            try:
                proc=m.start('grpc-loopback-test',[sys.executable,'-c',code],root,local_environment(root/'state'))
                m.wait_ports([port],10)
                records=m.check_listeners([port],root/'listeners.json')
                self.assertTrue(records)
                self.assertTrue(all(r['allowed'] for r in records))
                self.assertTrue(all(r['normalized_address'] in ('127.0.0.1','::1') for r in records))
                self.assertEqual(json.loads((root/'listeners.json').read_text())['status'],'PASS')
            finally:m.stop()
            self.assertIsNotNone(proc.poll());ensure_free_ports([port])


if __name__=='__main__':unittest.main()
