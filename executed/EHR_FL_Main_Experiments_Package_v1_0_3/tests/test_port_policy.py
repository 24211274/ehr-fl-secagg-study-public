"""Real Linux socket regressions; no Flower services or benchmark model fitting."""
from __future__ import annotations
import errno
import ipaddress
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ehr_fedavg import port_policy as pp
from ehr_fedavg.host import ensure_free_ports


def spare_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def closing_connection():
    """Server actively closes; return port with TIME_WAIT and no active listener."""
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        port = listener.getsockname()[1]
        client.settimeout(3); client.connect(('127.0.0.1', port))
        with listener.accept()[0] as accepted:
            accepted.settimeout(3)
            accepted.shutdown(socket.SHUT_WR)
            if client.recv(1) != b'': raise AssertionError('Expected FIN')
            client.shutdown(socket.SHUT_WR)
            if accepted.recv(1) != b'': raise AssertionError('Expected peer FIN')
    return port


def table_row(host='0100007F', port='4A98', state='06'):
    return f'  0: {host}:{port} 0100007F:CCCC {state} 00000000:00000000 00:00000000 00000000 1000 0 0\n'


class InputTests(unittest.TestCase):
    def test_empty_ports_rejected(self):
        with self.assertRaises(ValueError): pp.validated_ports([])
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): pp.validated_ports([19096, 19096])
    def test_boolean_port_rejected(self):
        with self.assertRaises(ValueError): pp.validated_ports([True])
    def test_string_port_rejected(self):
        with self.assertRaises(ValueError): pp.validated_ports(['19096'])
    def test_invalid_range_rejected(self):
        for p in [0, 80, 1024, 65536, -1]:
            with self.subTest(port=p), self.assertRaises(ValueError): pp.validated_ports([p])
    def test_valid_port_order_preserved(self):
        self.assertEqual(pp.validated_ports([19096, 19091]), [19096, 19091])
    def test_invalid_phase_rejected(self):
        with self.assertRaises(ValueError): pp.require_available_ports([19096], phase='ignore_errors')
    def test_prestart_cannot_silently_wait(self):
        with self.assertRaises(ValueError): pp.require_available_ports([19096], timeout_seconds=1)
    def test_invalid_timing_rejected(self):
        for v in [-1, float('nan'), float('inf'), 61, True]:
            with self.subTest(value=v), self.assertRaises(ValueError):
                pp.require_available_ports([19096], phase='post_cleanup', timeout_seconds=v)
    def test_invalid_interval_rejected(self):
        for v in [0, -1, float('nan'), float('inf'), 6, True]:
            with self.subTest(value=v), self.assertRaises(ValueError):
                pp.require_available_ports([19096], phase='post_cleanup', interval_seconds=v)
    def test_non_linux_rejected(self):
        with patch.object(pp.sys, 'platform', 'win32'), self.assertRaises(RuntimeError):
            pp.probe_loopback_ports([19096])


class ProcParsingTests(unittest.TestCase):
    def test_ipv4_state_and_exact_port_filter(self):
        # procfs words are host-endian; construct test rows accordingly.
        h=socket.inet_pton(socket.AF_INET,'127.0.0.1'); h=f'{int.from_bytes(h,sys.byteorder):08X}'
        text='header\n'+table_row(h)+table_row(h, '4A97','0A')
        rows=pp.parse_proc_tcp(text,socket.AF_INET,[19096])
        self.assertEqual(rows,[{'address':'127.0.0.1','port':19096,'family':'AF_INET','state':'TIME_WAIT'}])
    def test_ipv6_mapped_decoded(self):
        a=socket.inet_pton(socket.AF_INET6,'::ffff:127.0.0.1')
        h=''.join(f'{int.from_bytes(a[i:i+4],sys.byteorder):08X}' for i in range(0,16,4))
        rows=pp.parse_proc_tcp('header\n'+table_row(h, state='0A'),socket.AF_INET6,[19096])
        self.assertEqual(ipaddress.ip_address(rows[0]['address']).ipv4_mapped,ipaddress.IPv4Address('127.0.0.1'));self.assertEqual(rows[0]['state'],'LISTEN')
    def test_snapshot_does_not_include_remote_addresses(self):
        rows=pp.parse_proc_tcp('header\n'+table_row(),socket.AF_INET,[19096])
        self.assertEqual(set(rows[0]),{'address','port','family','state'})
    def test_invalid_family_rejected(self):
        with self.assertRaises(ValueError):pp.parse_proc_tcp('header',socket.AF_UNIX,[19096])
    def test_invalid_table_does_not_claim_empty(self):
        with self.assertRaises(ValueError):pp.parse_proc_tcp('header\ninvalid',socket.AF_INET,[19096])
    def test_diagnostic_missing_tables_is_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            r=pp.tcp_state_snapshot([19096],Path(d))
            self.assertEqual(r['status'],'UNAVAILABLE');self.assertEqual(len(r['errors']),2)
    def test_diagnostic_partial_is_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'tcp').write_text('header\n'+table_row())
            r=pp.tcp_state_snapshot([19096],Path(d))
            self.assertEqual(r['status'],'PARTIAL');self.assertEqual(len(r['local_endpoints']),1)
    def test_diagnostic_empty_is_not_sufficient_to_pass_failed_probe(self):
        r={'status':'FAIL','failures':[{'port':19096,'operation':'bind','errno':errno.EACCES,'errno_name':'EACCES'}]}
        with patch.object(pp,'probe_loopback_ports',return_value=r),self.assertRaises(pp.PortAvailabilityError):
            pp.require_available_ports([19096])


@unittest.skipUnless(sys.platform.startswith('linux'),'Linux/WSL2 socket regressions')
class KernelSocketTests(unittest.TestCase):
    def test_free_port_admitted_then_released(self):
        port=spare_port();r=ensure_free_ports([port])
        self.assertEqual(r['status'],'PASS')
        with socket.socket() as s:s.bind(('127.0.0.1',port));s.listen(1)
    def test_actual_listener_rejected_and_left_alive(self):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('127.0.0.1',0));s.listen(1);port=s.getsockname()[1]
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([port])
            self.assertEqual(s.getsockopt(socket.SOL_SOCKET,socket.SO_ACCEPTCONN),1)
    def test_actual_listener_with_reuseport_not_shared(self):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
            s.bind(('127.0.0.1',0));s.listen(1)
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([s.getsockname()[1]])
            self.assertEqual(s.getsockopt(socket.SOL_SOCKET,socket.SO_ACCEPTCONN),1)
    def test_wildcard_listener_rejected(self):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('0.0.0.0',0));s.listen(1)
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([s.getsockname()[1]])
    def test_ipv4_mapped_listener_rejected(self):
        with socket.socket(socket.AF_INET6,socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY,0)
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('::ffff:127.0.0.1',0));s.listen(1)
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([s.getsockname()[1]])
    def test_unshared_bound_reservation_rejected(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0))
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([s.getsockname()[1]])
    def test_time_wait_breaks_old_probe_but_new_listener_probe_passes(self):
        port=closing_connection()
        snap=pp.tcp_state_snapshot([port])
        self.assertTrue(any(r['state']=='TIME_WAIT' for r in snap['local_endpoints']))
        with socket.socket() as old:
            with self.assertRaises(OSError) as ctx:old.bind(('127.0.0.1',port))
            self.assertEqual(ctx.exception.errno,errno.EADDRINUSE)
        result=ensure_free_ports([port],phase='post_cleanup',timeout_seconds=1)
        self.assertEqual(result['status'],'PASS');self.assertEqual(result['attempt_count'],1)
        self.assertFalse(result['so_reuseport_enabled'])
    def test_conflict_on_last_port_closes_all_successful_probes(self):
        first=spare_port()
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1',0));occupied.listen(1);last=occupied.getsockname()[1]
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([first,last])
            with socket.socket() as check:check.bind(('127.0.0.1',first));check.listen(1)
    def test_audit_written_on_success_and_failure_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='port audit ') as d, socket.socket() as s:
            p=Path(d)/'ports.json';port=spare_port()
            ensure_free_ports([port],p);self.assertEqual(json.loads(p.read_text())['status'],'PASS')
            s.bind(('127.0.0.1',port));s.listen(1)
            with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([port],p)
            a=json.loads(p.read_text());self.assertEqual(a['status'],'FAIL')
            self.assertEqual(a['attempts'][0]['failures'][0]['errno'],errno.EADDRINUSE)
            self.assertFalse(p.with_suffix('.json.tmp').exists())
    def test_post_cleanup_can_wait_for_graceful_release_without_kill(self):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind(('127.0.0.1',0));s.listen(1);port=s.getsockname()[1]
            t=threading.Timer(.08,s.close);t.start()
            try:a=pp.require_available_ports([port],phase='post_cleanup',timeout_seconds=2,interval_seconds=.025)
            finally:t.join()
            self.assertEqual(a['status'],'PASS');self.assertGreater(a['attempt_count'],1)
            self.assertFalse(a['existing_processes_stopped_by_probe'])
    def test_persistent_actual_listener_blocks_after_bounded_rechecks(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));s.listen(1)
            start=time.monotonic()
            with self.assertRaises(pp.PortAvailabilityError) as ctx:
                pp.require_available_ports([s.getsockname()[1]],phase='post_cleanup',timeout_seconds=.08,interval_seconds=.02)
            self.assertLess(time.monotonic()-start,2)
            self.assertGreater(ctx.exception.audit['attempt_count'],1)
            self.assertEqual(s.getsockopt(socket.SOL_SOCKET,socket.SO_ACCEPTCONN),1)
    def test_socket_errors_other_than_eaddrinuse_not_retried(self):
        r={'status':'FAIL','failures':[{'port':19096,'operation':'bind','errno':errno.EPERM,'errno_name':'EPERM'}]}
        with patch.object(pp,'probe_loopback_ports',return_value=r) as mocked:
            with self.assertRaises(pp.PortAvailabilityError) as ctx:
                pp.require_available_ports([19096],phase='post_cleanup',timeout_seconds=10)
            self.assertEqual(mocked.call_count,1);self.assertEqual(ctx.exception.audit['attempt_count'],1)
    def test_genuine_eaddrinuse_is_never_inferred_to_be_time_wait(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));s.listen(1)
            a=pp.probe_loopback_ports([s.getsockname()[1]])
            self.assertEqual(a['status'],'FAIL')
            self.assertTrue(any(x['state']=='LISTEN' for x in a['before_probe']['local_endpoints']))
    def test_diagnostic_unavailable_does_not_weaken_kernel_check(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));s.listen(1)
            with patch.object(pp,'tcp_state_snapshot',return_value={'status':'UNAVAILABLE'}):
                with self.assertRaises(pp.PortAvailabilityError):ensure_free_ports([s.getsockname()[1]])
    def test_diagnostic_not_required_to_admit_provably_bindable_port(self):
        with patch.object(pp,'tcp_state_snapshot',return_value={'status':'UNAVAILABLE'}):
            self.assertEqual(ensure_free_ports([spare_port()])['status'],'PASS')

if __name__=='__main__':unittest.main()
