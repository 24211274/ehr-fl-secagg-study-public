"""Inspect owned TCP sockets without confusing process exit with permission loss.

Linux closes a process's file descriptors on full exit. A reaped process, or a
confirmed single-threaded zombie/dead process, cannot retain a TCP listener.
A zombie *thread-group leader* with other threads is NOT enough evidence of
full exit. Live/unclassified AccessDenied remains a hard error after a small,
bounded re-inspection. No root, namespace change, or system-wide socket fallback.

This is a point-in-time diagnostic, not complete historical network isolation.
"""
from __future__ import annotations

import time
from typing import Any

import psutil

LIFECYCLE_POLICY_ID = 'owned-tcp-single-thread-exit-confirmation-v1'
# Re-inspect this PID only; do not retry training, rounds, or failed runs.
REINSPECTION_DELAYS_SECONDS = (0.02, 0.05)
MAX_SOCKET_ATTEMPTS = len(REINSPECTION_DELAYS_SECONDS) + 1
_TERMINAL = (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)


class _ThreadGroupExitUnconfirmed(RuntimeError):
    """A terminal leader does not prove that the entire process has exited."""


def process_lifecycle(process: Any) -> dict[str, Any]:
    """Collect minimal identity/status evidence, without reading cmdline or env.

    is_running() checks PID + creation time in psutil, but returns True for
    zombies. Thus it is an identity check, not by itself an active-process test.
    """
    info: dict[str, Any] = {
        'pid': process.pid,
        'identity_present': None,
        'state': None,
        'threads': None,
        'full_exit_confirmed': False,
        'disposition': 'unresolved',
    }
    try:
        exists = process.is_running()
        if type(exists) is not bool:
            raise ValueError('Invalid process identity result')
        info['identity_present'] = exists
        if not exists:
            info.update(full_exit_confirmed=True, disposition='gone_or_pid_reused')
            return info
        info['state'] = process.status()
        if info['state'] not in _TERMINAL:
            info['disposition'] = 'active_or_nonterminal'
            return info
        count = process.num_threads()
        info['threads'] = count
        if type(count) is not int or count != 1:
            info['disposition'] = 'terminal_leader_without_full_exit_confirmation'
            return info
        # Recheck both the identity and state after reading the thread count.
        if process.is_running() is False:
            info.update(identity_present=False, full_exit_confirmed=True,
                        disposition='gone_or_pid_reused')
            return info
        after = process.status()
        info['state_rechecked'] = after
        if after not in _TERMINAL:
            info['disposition'] = 'state_changed_without_full_exit_confirmation'
            return info
        count_after = process.num_threads()
        info['threads_rechecked'] = count_after
        if type(count_after) is not int or count_after != 1:
            info['disposition'] = 'terminal_leader_without_full_exit_confirmation'
            return info
        info.update(full_exit_confirmed=True, disposition='confirmed_full_exit')
    except psutil.ZombieProcess as exc:
        # ZombieProcess is a NoSuchProcess subclass. Do not mistake the exception
        # alone for proof that every thread in the process has exited.
        info['probe_error'] = type(exc).__name__
    except psutil.NoSuchProcess:
        info.update(identity_present=False, full_exit_confirmed=True,
                    disposition='no_such_process')
    except (psutil.AccessDenied, OSError, AttributeError, ValueError) as exc:
        info['probe_error'] = type(exc).__name__
    return info


def _result(pid: int, status: str, attempts: list[dict[str, Any]],
            lifecycle: dict[str, Any], rows: list[Any] | None = None,
            error_type: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        'pid': pid, 'status': status, 'lifecycle': lifecycle,
        'socket_attempts': len(attempts), 'attempts': attempts,
        'connections': [] if rows is None else rows,
    }
    if error_type is not None:
        result['error_type'] = error_type
    return result


def _exit_result(pid: int, attempts: list[dict[str, Any]],
                 life: dict[str, Any]) -> dict[str, Any]:
    status = 'TERMINATED' if life['disposition'] == 'confirmed_full_exit' else 'VANISHED'
    return _result(pid, status, attempts, life)


def inspect_owned_tcp(process: Any) -> dict[str, Any]:
    """Read every TCP socket of this owned process or explain a confirmed exit.

    Persistent denial or unclassified errors cannot produce a successful result.
    Retry delays are at most 70 ms per process (excluding system-call time).
    Observed sockets are never discarded merely because the process later exits.
    """
    pid = process.pid
    attempts: list[dict[str, Any]] = []
    life = process_lifecycle(process)
    if life['full_exit_confirmed']:
        return _exit_result(pid, attempts, life)
    last_error = 'UnclassifiedSocketInspectionError'
    for index in range(MAX_SOCKET_ATTEMPTS):
        try:
            if life['state'] in _TERMINAL:
                # /proc/<leader>/fd may no longer describe the other threads.
                # Neither an empty socket list nor a permission error proves that
                # a partly exited thread group has no listeners.
                raise _ThreadGroupExitUnconfirmed()
            rows = list(process.net_connections(kind='tcp'))
            attempts.append({'attempt': index + 1, 'result': 'SOCKETS_READ',
                             'connection_count': len(rows)})
            return _result(pid, 'SCANNED', attempts, life, rows=rows)
        except psutil.ZombieProcess as exc:
            last_error = type(exc).__name__
        except psutil.NoSuchProcess as exc:
            attempts.append({'attempt': index + 1, 'result': type(exc).__name__})
            gone = dict(life, identity_present=False, full_exit_confirmed=True,
                        disposition='no_such_process')
            return _result(pid, 'VANISHED', attempts, gone)
        except (psutil.AccessDenied, OSError, _ThreadGroupExitUnconfirmed) as exc:
            last_error = type(exc).__name__
        life = process_lifecycle(process)
        attempts.append({'attempt': index + 1, 'result': last_error,
                         'lifecycle_after_error': life})
        if life['full_exit_confirmed']:
            return _exit_result(pid, attempts, life)
        if index < len(REINSPECTION_DELAYS_SECONDS):
            time.sleep(REINSPECTION_DELAYS_SECONDS[index])
        # Every later read uses the same psutil Process identity; check PID reuse.
        if index < len(REINSPECTION_DELAYS_SECONDS):
            life = process_lifecycle(process)
            if life['full_exit_confirmed']:
                return _exit_result(pid, attempts, life)
    return _result(pid, 'ERROR', attempts, life, error_type=last_error)
