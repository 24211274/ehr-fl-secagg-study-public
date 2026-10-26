"""A temporary plain Python loopback server tests ownership, not Flower networking."""
import os,socket,sys,tempfile,time,unittest
from pathlib import Path
from ehr_fedavg.host import OwnedProcesses,local_environment,ensure_free_ports
from ehr_fedavg.math_checks import validate_participants

@unittest.skipUnless(sys.platform.startswith('linux'),'Linux orchestration tests')
class LifecycleTests(unittest.TestCase):
    def test_owned_loopback_process_cleaned_without_stopping_other_listener(self):
        with tempfile.TemporaryDirectory(prefix='owned process test ') as d, socket.socket() as other:
            other.bind(('127.0.0.1',0));other.listen(1);other_port=other.getsockname()[1]
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
            root=Path(d);m=OwnedProcesses(root/'logs')
            code='import socket,time; s=socket.socket(); s.bind(("127.0.0.1",'+str(port)+')); s.listen(1); time.sleep(60)'
            try:
                proc=m.start('dummy-loopback',[sys.executable,'-c',code],root,local_environment(root/'state'))
                m.wait_ports([port],10);records=m.check_listeners([port]);self.assertTrue(records)
            finally:result=m.stop()
            self.assertIsNotNone(proc.poll());self.assertEqual(result[0]['pid'],proc.pid)
            ensure_free_ports([port]);self.assertEqual(other.getsockname()[1],other_port)
            with self.assertRaises(RuntimeError):ensure_free_ports([other_port])
    def test_reject_unexpected_service_exit(self):
        with tempfile.TemporaryDirectory() as d:
            m=OwnedProcesses(Path(d)/'logs')
            try:
                p=m.start('dummy-failed',[sys.executable,'-c','raise SystemExit(7)'],d,local_environment(Path(d)/'s'));p.wait(timeout=10)
                with self.assertRaises(RuntimeError):m.alive_services()
            finally:m.stop()
    def test_no_process_stop_is_noop(self):
        with tempfile.TemporaryDirectory() as d:self.assertEqual(OwnedProcesses(Path(d)/'logs').stop(),[])
    def test_invalid_phase_rejected(self):
        with self.assertRaises(ValueError):validate_participants([],'anything')

if __name__=='__main__':unittest.main()
