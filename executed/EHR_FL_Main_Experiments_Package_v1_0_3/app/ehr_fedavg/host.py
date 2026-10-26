"""Local-only process orchestration. Never kill by executable name or change the host."""
from __future__ import annotations
import json,os,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
import psutil
from .listener_policy import POLICY_ID, classify_listener_address
from .process_inspection import (LIFECYCLE_POLICY_ID, inspect_owned_tcp,
                                 REINSPECTION_DELAYS_SECONDS)


def ensure_free_ports(ports, audit_path=None, *, phase='pre_start', timeout_seconds=0):
    # Linux SO_REUSEADDR + bind/listen distinguishes rebindable closed connections
    # from an endpoint unavailable for a new listener. No SO_REUSEPORT or kills.
    from .port_policy import require_available_ports
    return require_available_ports(ports, audit_path, phase=phase,
                                   timeout_seconds=timeout_seconds)


FLOWER_COMMANDS = ('flwr', 'flower-superlink', 'flower-supernode', 'flower-superexec')


def inspect_flower_command_path(env):
    """Inspect the same child PATH used by Flower, without starting any service.

    Do not resolve the Python executable symlink: venv/bin/python commonly points
    to a system Python, but console-script entry points belong in venv/bin.
    Presence of a global command is not a substitute for the pinned venv command.
    """
    bindir = Path(os.path.abspath(sys.executable)).parent
    commands = []
    failures = []
    for name in FLOWER_COMMANDS:
        expected = bindir / name
        found = shutil.which(name, path=env.get('PATH', ''))
        present = expected.is_file()
        executable = present and os.access(expected, os.X_OK)
        matched = bool(found) and Path(os.path.abspath(found)) == expected
        # Keep account paths out of the shareable report. Raw process logs stay local.
        commands.append({
            'name': name,
            'expected_path': '<PYTHON_BIN>/' + name,
            'file_exists': present,
            'executable': executable,
            'child_path_resolves_to_pinned_command': matched,
        })
        if not present:
            failures.append('Missing executable in the selected Python environment: ' + name)
        elif not executable:
            failures.append('Not executable in the selected Python environment: ' + name)
        elif not matched:
            failures.append('Child PATH does not select the pinned command: ' + name)
    return {
        'status': 'FAIL' if failures else 'PASS',
        'commands': commands,
        'errors': failures,
        'path_change_scope': 'Per-subprocess environment only',
        'permanent_system_path_modified': False,
        'command_help_executed': False,
        'services_started_by_this_check': False,
    }


def local_environment(storage):
    env=os.environ.copy()
    # Runtime-only repair: Flower 1.22 starts flower-superexec by its bare name.
    # Using an absolute path for SuperLink alone does not make this command
    # discoverable by its subprocesses. Never edit .bashrc or the system PATH.
    bindir = str(Path(os.path.abspath(sys.executable)).parent)
    inherited = [part for part in env.get('PATH', '').split(os.pathsep) if part and part != bindir]
    env['PATH'] = os.pathsep.join([bindir, *inherited])
    env.update(FLWR_HOME=str(storage),FLWR_TELEMETRY_ENABLED='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
    # Process-local only: do not send loopback gRPC through a system proxy.
    for key in ['grpc_proxy','https_proxy','http_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy']:
        env.pop(key,None)
    return env


def write_project(path,root,condition,control_port):
    if condition not in ('reference','noniid'):raise ValueError('Unexpected condition')
    text='''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ehr-fedavg-fullsize"
version = "1.0.0"
description = "Local B EHR Flower full-size engineering pilot"
requires-python = ">=3.12,<3.14"
dependencies = []

[tool.hatch.build.targets.wheel]
packages = ["ehr_fedavg", "pilot_core", "ehr_preprocessing", "central_smoke"]

[tool.flwr.app]
publisher = "mai-local"

[tool.flwr.app.components]
serverapp = "ehr_fedavg.server_app:app"
clientapp = "ehr_fedavg.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 6
'''
    text+=f'condition = {json.dumps(condition)}\nruntime-root = {json.dumps(str(root))}\n'
    text+='''
[tool.flwr.federations]
default = "local-deployment"

[tool.flwr.federations.local-deployment]
'''
    text+=f'address = "127.0.0.1:{control_port}"\ninsecure = true\n'
    path.write_text(text)
    import tomllib
    tomllib.loads(text)


class OwnedProcesses:
    def __init__(self,logdir):self.items=[];self.logdir=Path(logdir);self.logdir.mkdir(parents=True,exist_ok=True);self.cleanup_info=[]
    def start(self,name,args,cwd,env):
        log=(self.logdir/(name+'.log')).open('w')
        try:p=subprocess.Popen(args,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:log.close();raise
        self.items.append((name,p,log));return p
    def alive_services(self):
        for name,p,_ in self.items:
            if name!='flwr-run' and p.poll() is not None:raise RuntimeError(f'{name} stopped unexpectedly (exit {p.returncode}); inspect local log')
    def own_psutil_processes(self):
        owned={}
        for _,p,_ in self.items:
            try:
                parent=psutil.Process(p.pid)
                for q in [parent]+parent.children(recursive=True):owned[(q.pid,q.create_time())]=q
            except psutil.NoSuchProcess:pass
        return list(owned.values())
    def check_listeners(self, required_ports, audit_path=None):
        """Check *all* owned TCP listeners, saving evidence before raising.

        Raw addresses and normalised addresses are both retained. Never ignore a
        wildcard or LAN listener, including one on an unlisted port. Failure to
        inspect an active/unclassified owned process remains a failure. A process
        whose full exit is confirmed is recorded separately, not treated as a live
        permission error. Thread-group leaders with live threads are not skipped.
        """
        required = list(required_ports)
        if (not required or len(required) != len(set(required)) or
                any(type(p) is not int or not 1024 < p < 65536 for p in required)):
            raise ValueError('Invalid or empty required listener ports')
        found = set()
        records = []
        errors = []
        vanished = []
        lifecycle_records = []
        terminated = []
        enumeration_errors = []
        try:
            owned_snapshot = self.own_psutil_processes()
        except (psutil.AccessDenied, OSError) as exc:
            owned_snapshot = []
            enumeration_errors.append({'error_type': type(exc).__name__,
                                       'stage': 'owned_process_enumeration'})
            errors.extend(enumeration_errors)
        for process in owned_snapshot:
            scan = inspect_owned_tcp(process)
            lifecycle_records.append({k: v for k, v in scan.items() if k != 'connections'})
            if scan['status'] == 'VANISHED':
                vanished.append(process.pid)
                continue
            if scan['status'] == 'TERMINATED':
                terminated.append({'pid': process.pid, **scan['lifecycle']})
                continue
            if scan['status'] == 'ERROR':
                errors.append({'pid': process.pid, 'error_type': scan['error_type'],
                               'lifecycle': scan['lifecycle'],
                               'socket_attempts': scan['socket_attempts']})
                continue
            for conn in scan['connections']:
                if conn.status != psutil.CONN_LISTEN:
                    continue
                raw = getattr(conn.laddr, 'ip', None)
                port = getattr(conn.laddr, 'port', None)
                check = classify_listener_address(raw)
                valid_port = type(port) is int and 0 < port < 65536
                if not valid_port:
                    check['allowed'] = False
                    check['classification'] = 'invalid_listener_port'
                record = {
                    'pid': process.pid, 'address': raw, 'port': port,
                    'family': getattr(conn.family, 'name', str(conn.family)),
                    **check,
                }
                records.append(record)
                if check['allowed']:
                    found.add(port)
        records.sort(key=lambda r: (r['pid'], str(r['address']), str(r['port'])))
        blocked = [r for r in records if not r['allowed']]
        missing = sorted(set(required) - found)
        passed = not blocked and not missing and not errors
        audit = {
            'status': 'PASS' if passed else 'FAIL',
            'policy_id': POLICY_ID,
            'loopback_only': True if passed else None,
            'allowed_normalized_endpoints': ['127.0.0.1', '::1'],
            'required_ports': sorted(required),
            'found_allowed_ports': sorted(found),
            'missing_required_ports': missing,
            'listeners': records,
            'blocked_listeners': blocked,
            'inspection_errors': errors,
            'owned_process_enumeration_errors': enumeration_errors,
            'process_lifecycle_policy_id': LIFECYCLE_POLICY_ID,
            'process_inspections': lifecycle_records,
            'confirmed_terminated_processes': terminated,
            'socket_reinspection_delays_seconds': list(REINSPECTION_DELAYS_SECONDS),
            'live_access_denied_bypassed': False,
            'root_or_systemwide_socket_fallback': False,
            'processes_vanished_during_scan': vanished,
            'ipv4_mapped_loopback_listeners': sum(r['classification'] == 'ipv4_mapped_loopback' for r in records),
            'scope': 'Point-in-time owned-process TCP listeners; not a firewall or OS isolation proof',
        }
        self.last_listener_audit = audit
        if audit_path is not None:
            target = Path(audit_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + '.tmp')
            tmp.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
            tmp.replace(target)
            # Keep earlier snapshots too, so short-lived workers remain auditable.
            history = target.with_name(target.stem + '_history.jsonl')
            with history.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(audit, separators=(',', ':')) + '\n')
        if blocked:
            witness = blocked[0]
            raise RuntimeError(
                'Owned listener rejected by local-only policy: '
                f"pid={witness['pid']} address={witness['address']} port={witness['port']} "
                f"classification={witness['classification']}; see listeners.json; no bypass"
            )
        if errors:
            first = errors[0]
            raise RuntimeError(
                'Could not inspect all active owned listeners: '
                f"pid={first.get('pid', 'unresolved')} error={first['error_type']}; "
                'see listeners.json process_inspections; no bypass'
            )
        if missing:
            raise RuntimeError(f'Required ports are not owned by the launched services: {missing}; see listeners.json')
        return records
    def wait_ports(self,ports,seconds):
        start=time.perf_counter()
        while True:
            self.alive_services();ok=True
            for port in ports:
                with socket.socket() as s:
                    s.settimeout(.5)
                    if s.connect_ex(('127.0.0.1',port))!=0:ok=False
            if ok:return
            if time.perf_counter()-start>seconds:raise TimeoutError('Service startup timeout; inspect local logs')
            time.sleep(.3)
    def stop(self):
        # Capture own descendant objects before parents disappear; never search/kill unrelated processes.
        owned=self.own_psutil_processes()
        for name,p,_ in reversed(self.items):
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p in owned:
            try:
                if p.is_running():p.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(owned,timeout=8)
        for p in alive:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(alive,timeout=3)
        for name,p,f in self.items:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait(timeout=3)
            self.cleanup_info.append({'name':name,'pid':p.pid,'exit_code':p.returncode});f.close()
        return self.cleanup_info
