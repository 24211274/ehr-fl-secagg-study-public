"""Point-in-time Linux loopback admission/release checks, not a port reservation.

A plain bind can fail for a just-closed TCP socket in TIME_WAIT even after the
service exited. Probe with SO_REUSEADDR (NOT SO_REUSEPORT), then listen too:
bind alone is not enough to rule out a socket which is already bound. Hold all
successful probes together and always close them. No peer is contacted, no
existing process is stopped, no sysctl/firewall settings are modified.

Filtered /proc/self/net/tcp{,6} snapshots are diagnostic only. Socket admission
is decided by the kernel's actual bind/listen results, never by ignoring an
unreadable process or assuming that a failed bind must mean TIME_WAIT.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import errno
import ipaddress
import json
import math
from pathlib import Path
import socket
import sys
import time
from typing import Any, Iterable

PORT_POLICY_ID = 'linux-exact-loopback-reuseaddr-bind-listen-v1'
PROBE_ADDRESS = '127.0.0.1'
_TCP_STATES = {
    '01': 'ESTABLISHED', '02': 'SYN_SENT', '03': 'SYN_RECV',
    '04': 'FIN_WAIT1', '05': 'FIN_WAIT2', '06': 'TIME_WAIT',
    '07': 'CLOSE', '08': 'CLOSE_WAIT', '09': 'LAST_ACK',
    '0A': 'LISTEN', '0B': 'CLOSING', '0C': 'NEW_SYN_RECV',
}


class PortAvailabilityError(RuntimeError):
    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        self.phase = audit['phase']
        failures = audit['attempts'][-1]['failures']
        first = failures[0]
        super().__init__(
            f"Port admission/release failed [{self.phase}]: "
            f"{PROBE_ADDRESS}:{first['port']} operation={first['operation']} "
            f"errno={first.get('errno')} ({first.get('errno_name', 'unknown')}); "
            "see ports_before_start.json or ports_after_cleanup.json. "
            "No competing process was stopped and no port was changed."
        )


def validated_ports(ports: Iterable[int]) -> list[int]:
    values = list(ports)
    if (not values or any(type(p) is not int or not 1024 < p < 65536 for p in values)
            or len(values) != len(set(values))):
        raise ValueError('Invalid, empty or duplicate deployment ports')
    return values


def parse_proc_tcp(text: str, family: int, ports: Iterable[int]) -> list[dict[str, Any]]:
    """Keep local endpoints/states only; no remote endpoints, UIDs or PID scan."""
    if family not in (socket.AF_INET, socket.AF_INET6):
        raise ValueError('Unexpected TCP address family')
    wanted = set(validated_ports(ports))
    rows = []
    for line in text.splitlines()[1:]:
        cols = line.split()
        if not cols:
            continue
        if len(cols) < 4:
            raise ValueError('Incomplete TCP table row')
        host_hex, port_hex = cols[1].split(':')
        port = int(port_hex, 16)
        if port not in wanted:
            continue
        width = 8 if family == socket.AF_INET else 32
        if len(host_hex) != width:
            raise ValueError('Invalid local address width')
        # procfs prints native-endian 32-bit words (both AF_INET and AF_INET6).
        packed = b''.join(int(host_hex[i:i+8], 16).to_bytes(4, sys.byteorder)
                          for i in range(0, width, 8))
        address = str(ipaddress.ip_address(packed))
        code = cols[3].upper()
        if len(code) != 2:
            raise ValueError('Invalid TCP state')
        int(code, 16)
        rows.append({'address': address, 'port': port,
                     'family': 'AF_INET' if family == socket.AF_INET else 'AF_INET6',
                     'state': _TCP_STATES.get(code, 'UNKNOWN_' + code)})
    return sorted(rows, key=lambda r: (r['port'], r['address'], r['state']))


def tcp_state_snapshot(ports: Iterable[int], proc_root: Path | None = None) -> dict[str, Any]:
    """Best-effort diagnostics in this network namespace, not an admission gate."""
    values = validated_ports(ports)
    base = Path('/proc/self/net') if proc_root is None else Path(proc_root)
    records, errors, sources = [], [], []
    for name, family in [('tcp', socket.AF_INET), ('tcp6', socket.AF_INET6)]:
        try:
            text = (base / name).read_text(encoding='ascii')
            records.extend(parse_proc_tcp(text, family, values))
            sources.append(name)
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append({'source': name, 'error_type': type(exc).__name__})
    return {
        'status': 'COMPLETE' if not errors else ('PARTIAL' if sources else 'UNAVAILABLE'),
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'local_endpoints': sorted(records, key=lambda r: (r['port'], r['address'], r['state'])),
        'errors': errors, 'sources_read': sources,
        'filter': 'Only configured local TCP ports; no remote endpoints or credentials',
        'role': 'Diagnostic only; not used to bypass failed bind/listen',
        'scope': 'Current process network namespace; not process ownership proof',
    }


def probe_loopback_ports(ports: Iterable[int]) -> dict[str, Any]:
    """Try binding/listening on all endpoints simultaneously, then close probes."""
    values = validated_ports(ports)
    if not sys.platform.startswith('linux'):
        raise RuntimeError('This port policy is tested for Linux/WSL2 only')
    before = tcp_state_snapshot(values)
    probes, failures = [], []
    with ExitStack() as stack:
        for port in values:
            operation = 'socket'
            try:
                s = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                operation = 'setsockopt'
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Never enable SO_REUSEPORT: that could share an active listener.
                if hasattr(socket, 'SO_REUSEPORT') and s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) != 0:
                    raise OSError(errno.EINVAL, 'Unexpected SO_REUSEPORT on fresh socket')
                operation = 'bind'
                s.bind((PROBE_ADDRESS, port))
                operation = 'listen'
                s.listen(1)
                probes.append({'port': port, 'address': PROBE_ADDRESS,
                               'status': 'BOUND_AND_LISTENING_DURING_PROBE',
                               'so_reuseaddr': True, 'so_reuseport': False})
            except OSError as exc:
                failures.append({'port': port, 'address': PROBE_ADDRESS,
                                 'operation': operation, 'errno': exc.errno,
                                 'errno_name': errno.errorcode.get(exc.errno, 'UNKNOWN'),
                                 'error_type': type(exc).__name__})
    # No listeners created by this function remain after ExitStack exits.
    after = tcp_state_snapshot(values)
    return {'status': 'FAIL' if failures else 'PASS', 'before_probe': before,
            'probes': probes, 'failures': failures, 'after_probes_closed': after,
            'all_probe_sockets_closed': True,
            'all_successful_probes_held_together': True}


def _save_audit(path: Path | str | None, audit: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)


def require_available_ports(ports: Iterable[int], audit_path: Path | str | None = None,
                            *, phase: str = 'pre_start', timeout_seconds: float = 0,
                            interval_seconds: float = .25) -> dict[str, Any]:
    """Bounded *port recheck* after cleanup; never retry training or change port.

    Initial admission is a single check. After owned-service cleanup, callers
    may allow a bounded settle interval. Every failure is retained in the audit.
    EACCES, EPERM and other non-EADDRINUSE errors fail immediately. A persistent
    actual listener still fails. A successful result is not a future reservation;
    existing live owned-listener guards remain mandatory after service startup.
    """
    values = validated_ports(ports)
    if phase not in ('pre_start', 'post_cleanup'):
        raise ValueError('Unknown port check phase')
    if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
            or not 0 <= timeout_seconds <= 60
            or isinstance(interval_seconds, bool) or not math.isfinite(interval_seconds)
            or not 0 < interval_seconds <= 5):
        raise ValueError('Invalid bounded port check timing')
    if phase == 'pre_start' and timeout_seconds != 0:
        raise ValueError('Pre-start admission must not automatically wait/retry')
    start = time.monotonic()
    attempts = []
    while True:
        result = probe_loopback_ports(values)
        attempts.append(result)
        elapsed = time.monotonic() - start
        passed = result['status'] == 'PASS'
        # An access/permission/socket failure is not a TIME_WAIT diagnosis.
        can_wait = (not passed and phase == 'post_cleanup'
                    and elapsed < timeout_seconds
                    and all(f['errno'] == errno.EADDRINUSE for f in result['failures']))
        audit = {
            'policy_id': PORT_POLICY_ID, 'status': 'PASS' if passed else ('WAITING' if can_wait else 'FAIL'),
            'phase': phase, 'probe_address': PROBE_ADDRESS, 'required_ports': values,
            'timeout_seconds': timeout_seconds, 'interval_seconds': interval_seconds,
            'elapsed_seconds': elapsed, 'attempt_count': len(attempts), 'attempts': attempts,
            'so_reuseaddr_on_probe_only': True, 'so_reuseport_enabled': False,
            'listen_checked_in_addition_to_bind': True,
            'temporary_probe_listeners_are_loopback_only': True,
            'probe_sockets_closed': True, 'automatic_training_retry': False,
            'port_numbers_changed': False, 'existing_processes_stopped_by_probe': False,
            'kernel_or_firewall_settings_changed': False,
            'scope': 'Point-in-time bind/listen compatibility at 127.0.0.1, NOT a future reservation or full process cleanup proof',
        }
        _save_audit(audit_path, audit)
        if passed:
            return audit
        if not can_wait:
            raise PortAvailabilityError(audit)
        time.sleep(min(interval_seconds, max(0, timeout_seconds - elapsed)))
