"""Regression tests for exit/permission races; no dataset or Flower execution."""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil

from ehr_fedavg.host import OwnedProcesses
from ehr_fedavg.process_inspection import (
    MAX_SOCKET_ATTEMPTS, REINSPECTION_DELAYS_SECONDS,
    inspect_owned_tcp, process_lifecycle,
)


def fake(pid=501, state=psutil.STATUS_SLEEPING, threads=1, rows=None):
    return SimpleNamespace(
        pid=pid, is_running=Mock(return_value=True),
        status=Mock(return_value=state), num_threads=Mock(return_value=threads),
        net_connections=Mock(return_value=rows or []),
    )


def conn(address='127.0.0.1', port=19091):
    return SimpleNamespace(status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(ip=address, port=port), family=socket.AF_INET)


class ProcessLifecycleTests(unittest.TestCase):
    def test_active_process_not_terminal(self):
        p=fake();r=process_lifecycle(p)
        self.assertFalse(r['full_exit_confirmed']);self.assertEqual(r['state'],'sleeping')
        p.num_threads.assert_not_called()

    def test_zombie_identity_still_present_not_assumed_active(self):
        p=fake(state=psutil.STATUS_ZOMBIE)
        r=process_lifecycle(p)
        self.assertTrue(r['identity_present']);self.assertTrue(r['full_exit_confirmed'])
        self.assertEqual(r['disposition'],'confirmed_full_exit')
        self.assertEqual(r['threads_rechecked'],1)

    def test_confirmed_dead_single_thread(self):
        self.assertTrue(process_lifecycle(fake(state=psutil.STATUS_DEAD))['full_exit_confirmed'])

    def test_zombie_leader_with_other_threads_not_confirmed(self):
        p=fake(state=psutil.STATUS_ZOMBIE,threads=3)
        r=process_lifecycle(p)
        self.assertFalse(r['full_exit_confirmed']);self.assertEqual(r['threads'],3)

    def test_thread_count_must_be_exact_integer_one(self):
        for n in [True,0,-1,2,None,'1',1.0]:
            with self.subTest(n=n):
                self.assertFalse(process_lifecycle(fake(state=psutil.STATUS_ZOMBIE,threads=n))['full_exit_confirmed'])

    def test_threads_rechecked_before_confirming(self):
        p=fake(state=psutil.STATUS_ZOMBIE);p.num_threads.side_effect=[1,2]
        self.assertFalse(process_lifecycle(p)['full_exit_confirmed'])

    def test_state_rechecked_before_confirming(self):
        p=fake(state=psutil.STATUS_ZOMBIE);p.status.side_effect=['zombie','running']
        self.assertFalse(process_lifecycle(p)['full_exit_confirmed'])

    def test_gone_or_reused_pid_recognised(self):
        p=fake();p.is_running.return_value=False
        r=process_lifecycle(p)
        self.assertTrue(r['full_exit_confirmed']);self.assertEqual(r['disposition'],'gone_or_pid_reused')
        p.status.assert_not_called()

    def test_pid_disappears_between_status_reads(self):
        p=fake(state='zombie');p.is_running.side_effect=[True,False]
        r=process_lifecycle(p);self.assertTrue(r['full_exit_confirmed']);self.assertFalse(r['identity_present'])

    def test_no_such_process_confirmed(self):
        p=fake();p.status.side_effect=psutil.NoSuchProcess(p.pid)
        self.assertEqual(process_lifecycle(p)['disposition'],'no_such_process')

    def test_zombie_exception_alone_not_enough(self):
        p=fake();p.status.side_effect=psutil.ZombieProcess(p.pid)
        r=process_lifecycle(p);self.assertFalse(r['full_exit_confirmed'])
        self.assertEqual(r['probe_error'],'ZombieProcess')

    def test_unreadable_status_not_treated_as_exit(self):
        p=fake();p.status.side_effect=psutil.AccessDenied(p.pid)
        r=process_lifecycle(p);self.assertFalse(r['full_exit_confirmed'])
        self.assertEqual(r['probe_error'],'AccessDenied')

    def test_unreadable_thread_count_not_treated_as_exit(self):
        p=fake(state='zombie');p.num_threads.side_effect=psutil.AccessDenied(p.pid)
        self.assertFalse(process_lifecycle(p)['full_exit_confirmed'])

    def test_invalid_identity_result_not_exit(self):
        p=fake();p.is_running.return_value=None
        self.assertFalse(process_lifecycle(p)['full_exit_confirmed'])


class SocketInspectionTests(unittest.TestCase):
    def setUp(self):
        self.sleep=patch('ehr_fedavg.process_inspection.time.sleep').start()
        self.addCleanup(patch.stopall)

    def test_active_socket_read(self):
        p=fake(rows=[conn()]);r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'SCANNED');self.assertEqual(r['socket_attempts'],1)
        p.net_connections.assert_called_once_with(kind='tcp');self.sleep.assert_not_called()

    def test_zombie_skips_only_after_confirmation(self):
        p=fake(state='zombie');p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'TERMINATED');self.assertEqual(r['socket_attempts'],0)
        p.net_connections.assert_not_called()

    def test_no_such_process_during_socket_read(self):
        p=fake();p.net_connections.side_effect=psutil.NoSuchProcess(p.pid)
        self.assertEqual(inspect_owned_tcp(p)['status'],'VANISHED')

    def test_access_denied_live_process_remains_error(self):
        p=fake();p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'ERROR');self.assertEqual(r['error_type'],'AccessDenied')
        self.assertEqual(p.net_connections.call_count,MAX_SOCKET_ATTEMPTS)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list],list(REINSPECTION_DELAYS_SECONDS))

    def test_access_denied_state_unknown_remains_error(self):
        p=fake();p.status.side_effect=psutil.AccessDenied(p.pid);p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        self.assertEqual(inspect_owned_tcp(p)['status'],'ERROR')

    def test_access_denied_threads_unknown_remains_error(self):
        p=fake(state='zombie');p.num_threads.side_effect=psutil.AccessDenied(p.pid);p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        self.assertEqual(inspect_owned_tcp(p)['status'],'ERROR')

    def test_access_denied_leader_with_live_threads_remains_error(self):
        p=fake(state='zombie',threads=2);p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        r=inspect_owned_tcp(p);self.assertEqual(r['status'],'ERROR');self.assertEqual(r['lifecycle']['threads'],2)

    def test_empty_leader_fd_list_does_not_hide_live_threads(self):
        p=fake(state='zombie',threads=2,rows=[])
        r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'ERROR')
        p.net_connections.assert_not_called()

    def test_unknown_thread_count_not_replaced_by_empty_socket_list(self):
        p=fake(state='zombie',rows=[]);p.num_threads.side_effect=psutil.AccessDenied(p.pid)
        r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'ERROR')
        p.net_connections.assert_not_called()

    def test_access_denied_then_proven_zombie(self):
        p=fake();p.status.side_effect=['running','zombie','zombie'];p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'TERMINATED');self.assertEqual(r['socket_attempts'],1)
        self.assertEqual(r['attempts'][0]['result'],'AccessDenied')

    def test_access_denied_then_reaped(self):
        p=fake();p.is_running.side_effect=[True,False];p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        self.assertEqual(inspect_owned_tcp(p)['status'],'VANISHED')

    def test_access_denied_then_readable_preserves_evidence(self):
        p=fake();p.net_connections.side_effect=[psutil.AccessDenied(p.pid),[conn()]]
        r=inspect_owned_tcp(p)
        self.assertEqual(r['status'],'SCANNED');self.assertEqual(r['socket_attempts'],2)
        self.assertEqual(r['attempts'][0]['result'],'AccessDenied');self.assertEqual(len(r['connections']),1)

    def test_os_error_while_still_active_fails(self):
        p=fake();p.net_connections.side_effect=OSError('probe failure')
        self.assertEqual(inspect_owned_tcp(p)['status'],'ERROR')

    def test_zombie_exception_does_not_bypass_full_exit_check(self):
        p=fake();p.net_connections.side_effect=psutil.ZombieProcess(p.pid)
        self.assertEqual(inspect_owned_tcp(p)['status'],'ERROR')

    def test_exception_does_not_claim_sockets_read(self):
        p=fake();p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        r=inspect_owned_tcp(p);self.assertEqual(r['connections'],[])
        self.assertTrue(all(a['result']!='SOCKETS_READ' for a in r['attempts']))

    def test_pid_reuse_before_first_scan_not_queried(self):
        p=fake();p.is_running.return_value=False
        r=inspect_owned_tcp(p);self.assertEqual(r['status'],'VANISHED');p.net_connections.assert_not_called()


class GuardIntegrationMockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.manager=OwnedProcesses(self.root/'logs');self.audit=self.root/'listeners.json'
        patch('ehr_fedavg.process_inspection.time.sleep').start();self.addCleanup(patch.stopall)

    def check(self, processes, ports=None):
        with patch.object(self.manager,'own_psutil_processes',return_value=processes):
            return self.manager.check_listeners(ports or [19091],self.audit)

    def test_live_listener_plus_zombies_passes_with_evidence(self):
        live=fake(501,rows=[conn()]);zombies=[fake(i,state='zombie') for i in [502,503,504]]
        for p in zombies:p.net_connections.side_effect=psutil.AccessDenied(p.pid)
        self.check([live,*zombies]);r=json.loads(self.audit.read_text())
        self.assertEqual(r['status'],'PASS');self.assertEqual(len(r['confirmed_terminated_processes']),3)
        self.assertEqual(len(r['process_inspections']),4);self.assertEqual(r['inspection_errors'],[])

    def test_zombie_cannot_satisfy_required_port(self):
        with self.assertRaisesRegex(RuntimeError,'Required ports'):self.check([fake(state='zombie')])

    def test_zombies_do_not_hide_wildcard_listener(self):
        with self.assertRaisesRegex(RuntimeError,'Owned listener'):self.check([fake(501,rows=[conn('0.0.0.0')]),fake(502,state='zombie')])
        self.assertEqual(len(json.loads(self.audit.read_text())['blocked_listeners']),1)

    def test_live_permission_denial_still_stops_with_pid(self):
        bad=fake(502);bad.net_connections.side_effect=psutil.AccessDenied(502)
        with self.assertRaisesRegex(RuntimeError,'pid=502 error=AccessDenied'):
            self.check([fake(501,rows=[conn()]),bad])
        r=json.loads(self.audit.read_text());self.assertEqual(r['inspection_errors'][0]['lifecycle']['state'],'sleeping')

    def test_enumeration_denied_saved_and_stops(self):
        with patch.object(self.manager,'own_psutil_processes',side_effect=psutil.AccessDenied(501)):
            with self.assertRaisesRegex(RuntimeError,'Could not inspect'):self.manager.check_listeners([19091],self.audit)
        self.assertEqual(len(json.loads(self.audit.read_text())['owned_process_enumeration_errors']),1)

    def test_history_retains_pass_and_failed_snapshots(self):
        self.check([fake(501,rows=[conn()])])
        bad=fake(502);bad.net_connections.side_effect=psutil.AccessDenied(502)
        with self.assertRaises(RuntimeError):self.check([fake(501,rows=[conn()]),bad])
        hist=[json.loads(x) for x in (self.root/'listeners_history.jsonl').read_text().splitlines()]
        self.assertEqual([a['status'] for a in hist],['PASS','FAIL'])

    def test_no_systemwide_privileged_fallback(self):
        bad=fake(502);bad.net_connections.side_effect=psutil.AccessDenied(502)
        with patch('psutil.net_connections',side_effect=AssertionError('No global fallback')):
            with self.assertRaises(RuntimeError):self.check([fake(501,rows=[conn()]),bad])



@unittest.skipUnless(sys.platform.startswith('linux'),'Linux lifecycle probe')
class RealUnprivilegedLifecycleTest(unittest.TestCase):
    def test_real_zombies_reproduce_failure_and_live_denial_still_blocks(self):
        import subprocess
        root=Path(__file__).resolve().parents[1]
        env=os.environ.copy();env['PYTHONPATH']=str(root/'app');env['PYTHONDONTWRITEBYTECODE']='1'
        # The target user's normal WSL account uses no privilege change. In the
        # isolated build container ONLY, drop privileges for a representative test.
        kwargs={'user':65534,'group':65534,'extra_groups':[]} if os.geteuid()==0 else {}
        r=subprocess.run([sys.executable,str(root/'tests/process_lifecycle_probe.py')],
                         env=env,text=True,capture_output=True,timeout=45,**kwargs)
        self.assertEqual(r.returncode,0,r.stderr)
        data=json.loads(r.stdout)
        self.assertEqual(data['status'],'REAL_UNPRIVILEGED_LIFECYCLE_PROBES_PASS_NOT_FLOWER')
        self.assertEqual(len(data['zombie_case']['workers']),3)
        self.assertEqual(data['live_permission_denial']['status'],'FAIL')

if __name__=='__main__':unittest.main()
