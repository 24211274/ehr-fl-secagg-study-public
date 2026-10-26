#!/usr/bin/env python3
"""Non-root Linux kernel reproduction, no Flower or benchmark data.

A loopback service intentionally keeps three exited child processes unreaped
for the duration of inspection. The same real Process objects produce the
old failure and a terminal-state-aware success. Children are then reaped.
A separate live non-dumpable child tests that actual denial is STILL rejected.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import psutil
from ehr_fedavg.host import OwnedProcesses, local_environment, ensure_free_ports

ROOT = Path(__file__).resolve().parents[1]

def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]

def wait_file(path, proc, timeout=15):
    start=time.perf_counter()
    while not path.exists():
        if proc.poll() is not None:raise RuntimeError('Fixture exited before ready')
        if time.perf_counter()-start>timeout:raise RuntimeError('Fixture readiness timeout')
        time.sleep(.02)
    return json.loads(path.read_text())

FIXTURE=r'''
import ctypes,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
import psutil
port=int(sys.argv[1]);kind=sys.argv[2];workers=[]
def quit(signum,frame):raise SystemExit(0)
signal.signal(signal.SIGTERM,quit)
s=socket.socket();s.bind(('127.0.0.1',port));s.listen(5)
try:
    if kind=='zombies':
        # Keep Popen instances and do NOT poll/wait before the diagnostic.
        workers=[subprocess.Popen([sys.executable,'-c','pass']) for _ in range(3)]
        t=time.perf_counter()
        while not all(psutil.Process(p.pid).status()==psutil.STATUS_ZOMBIE for p in workers):
            if time.perf_counter()-t>10:raise RuntimeError('No zombies observed')
            time.sleep(.01)
    elif kind=='live_denial':
        # Child-only PR_SET_DUMPABLE: does not change host/user privileges.
        if ctypes.CDLL(None,use_errno=True).prctl(4,0,0,0,0)!=0:raise OSError(ctypes.get_errno())
    else:raise ValueError(kind)
    Path('ready.json').write_text(json.dumps({'pid':os.getpid(),'worker_pids':[p.pid for p in workers]}))
    while True:time.sleep(.05)
finally:
    s.close()
    for p in workers:p.wait(timeout=5)
'''


def main():
    if not sys.platform.startswith('linux') or os.geteuid()==0:
        raise RuntimeError('This probe requires unprivileged Linux execution')
    result={'uid':os.getuid(),'psutil_version':psutil.__version__,
            'flower_executed':False,'benchmark_read':False,
            'scope':'Real local sockets and intentionally short-lived subprocesses only'}
    spec=importlib.util.spec_from_file_location('ehr_fedavg._legacy_host',ROOT/'reference/legacy_host_v1_0_3.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    with tempfile.TemporaryDirectory(prefix='unprivileged lifecycle probe ') as temp:
        root=Path(temp);manager=OwnedProcesses(root/'logs');port=free_port()
        try:
            process=manager.start('lifecycle-probe',[sys.executable,'-c',FIXTURE,str(port),'zombies'],root,local_environment(root/'state'))
            ready=wait_file(root/'ready.json',process);manager.wait_ports([port],10)
            evidence=[]
            for pid in ready['worker_pids']:
                p=psutil.Process(pid);row={'pid':pid,'state':p.status(),'threads':p.num_threads()}
                try:row['socket_rows']=len(p.net_connections(kind='tcp'))
                except psutil.AccessDenied:row['socket_error']='AccessDenied'
                evidence.append(row)
            assert len(evidence)==3 and all(e.get('socket_error')=='AccessDenied' and e['state']=='zombie' and e['threads']==1 for e in evidence),evidence
            legacy=old.OwnedProcesses(root/'old-logs');legacy.items=manager.items
            try:
                legacy.check_listeners([port],root/'old.json')
                raise AssertionError('Old guard unexpectedly passed')
            except RuntimeError as exc:
                result['old_guard_message']=str(exc)
            old_audit=json.loads((root/'old.json').read_text())
            assert old_audit['status']=='FAIL' and len(old_audit['inspection_errors'])==3 and old_audit['missing_required_ports']==[] and old_audit['blocked_listeners']==[]
            manager.check_listeners([port],root/'new.json')
            new=json.loads((root/'new.json').read_text())
            assert new['status']=='PASS' and len(new['confirmed_terminated_processes'])==3
            assert new['inspection_errors']==[] and all(x['allowed'] for x in new['listeners'])
            result['zombie_case']={'workers':evidence,'old_audit':old_audit,'new_audit':new}
        finally:
            manager.stop()
            ensure_free_ports([port])
        result['zombie_fixture_reaped_and_listener_closed']=True
    with tempfile.TemporaryDirectory(prefix='unprivileged real denial probe ') as temp:
        root=Path(temp);manager=OwnedProcesses(root/'logs');port=free_port()
        try:
            process=manager.start('live-denial-probe',[sys.executable,'-c',FIXTURE,str(port),'live_denial'],root,local_environment(root/'state'))
            wait_file(root/'ready.json',process);manager.wait_ports([port],10)
            try:
                manager.check_listeners([port],root/'audit.json')
                raise AssertionError('Live permission denial incorrectly passed')
            except RuntimeError as exc:result['live_denial_message']=str(exc)
            audit=json.loads((root/'audit.json').read_text())
            assert audit['status']=='FAIL' and audit['inspection_errors'][0]['error_type']=='AccessDenied'
            assert audit['process_inspections'][0]['socket_attempts']==3
            assert audit['confirmed_terminated_processes']==[]
            result['live_permission_denial']=audit
        finally:
            manager.stop()
            ensure_free_ports([port])
        result['live_fixture_stopped_and_listener_closed']=True
    result['status']='REAL_UNPRIVILEGED_LIFECYCLE_PROBES_PASS_NOT_FLOWER'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
