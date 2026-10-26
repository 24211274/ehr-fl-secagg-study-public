#!/usr/bin/env python3
"""Continue this authorized running task only after both real preparation gates pass.

This is a bounded foreground waiting process, not an automation or a scheduler.
It never starts/retries preparation, retries training, or overwrites a chain.
After success it replaces itself with the reviewed full-chain driver via execv.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

DRIVER_SHA256 = '1a20512a08ac3e3cbf8b8311936a1632204b54b02e33844193f4f0d8136e3231'
TRAINER_SHA256 = '668225aeb922e95536577936bdf43ac2954c054c5679804948a8aeeb06d763ad'
PACKAGE_SHA256 = '0e4bc1175bf97a766d46aa60f02c4c891c483d3870957b3ae6ac52ea7734dc88'
COMPLETE_GATE = 'ME06_FRESH_PREPARATION_COMPLETE_NO_FINAL_TESTS'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if exclusive:
        with path.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)


def expected_preparation_command(root):
    return [str(root / '.venv/bin/python'), '-I', '-B', str(root / 'portable/train_reproduction.py'),
        'prepare', '--run-root', str(root), '--source', str(root / 'source'),
        '--campaign', str(root / 'campaign'), '--prepared', str(root / 'preprocessing/bound_trainval'),
        '--frozen-reference', str(root / 'assets/frozen_reference.zip'), '--preprocessing-receipt',
        str(root / 'preprocessing/reports/stage_status.json')]


def evaluate_gates(runner, preparation, campaign_hash, root):
    """Pure decision function; only PASS from both current, bound gates can launch."""
    if runner is not None and runner.get('status') == 'FAILED':
        return 'STOP', 'The real preparation command failed; no retry or continuation'
    prep_status = preparation.get('status', '') if preparation is not None else ''
    if 'FAILED' in prep_status or 'STOP' in prep_status:
        return 'STOP', 'The inner preparation gate failed; no retry or continuation'
    if runner is None:
        return 'WAIT', 'Real preparation command receipt is not readable yet'
    if runner.get('status') not in ('RUNNING', 'PASS'):
        return 'STOP', 'Unknown preparation command status'
    if (runner.get('command') != expected_preparation_command(root)
            or runner.get('cwd') != str(root) or runner.get('automatic_retry') is not False):
        return 'STOP', 'Preparation receipt identifies a different command/environment'
    if runner['status'] == 'RUNNING':
        return 'WAIT', 'Real preparation command is still running'
    if type(runner.get('exit_code')) is not int or runner['exit_code'] != 0:
        return 'STOP', 'Preparation PASS receipt lacks an exact zero exit code'
    if preparation is None or prep_status != COMPLETE_GATE:
        return 'STOP', 'Preparation command exited but its current inner completion gate is missing/wrong'
    required = {'new_reproduction': True, 'historical_receipts_reused': False,
        'wrapper_sha256': TRAINER_SHA256, 'package_manifest_sha256': PACKAGE_SHA256,
        'main_training_performed': False, 'test_data_opened': False,
        'jobs_prepared': 30, 'native_template_probes': 30, 'native_relocated_clone_probes': 20}
    if any(type(preparation.get(k)) is not type(v) or preparation[k] != v for k, v in required.items()):
        return 'STOP', 'Current preparation gate does not prove the required new-run identity/counts/boundaries'
    if campaign_hash is None or preparation.get('campaign_manifest_sha256') != campaign_hash:
        return 'STOP', 'Preparation gate is not bound to the current campaign manifest'
    return 'PASS', 'Both real preparation gates passed and bind the current campaign'


def driver_command(root, seal, export):
    python = str(root / '.venv/bin/python')
    return [python, '-I', '-B', str(root / 'portable/complete_reproduction.py'),
            '--run-root', str(root), '--adapter-seal', str(seal), '--export-dir', str(export)]


def snapshot_json(path):
    # The outer runner currently writes JSON directly. A transient incomplete write
    # is never interpreted as PASS; wait for its next readable snapshot.
    try:
        return read(path), None
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return None, type(exc).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--adapter-seal', type=Path)
    parser.add_argument('--max-wait-seconds', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.max_wait_seconds <= 8000:
        parser.error('--max-wait-seconds must be between 1 and 8000')
    root = args.run_root.resolve(strict=True)
    export = args.export_dir.resolve(strict=True)
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12) or os.geteuid() == 0:
        raise RuntimeError('Non-root Linux Python 3.12 required')
    if Path(os.path.abspath(sys.executable)) != root / '.venv/bin/python':
        raise RuntimeError('Use only this new run-root .venv/bin/python')
    if export.is_relative_to(root / 'source') or export.is_relative_to(root / 'campaign'):
        raise ValueError('Export directory must not overwrite source/campaign artifacts')
    import fcntl
    lock = (root / '.me06_waiting_launcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_dir = root / 'waiting_launch'
    state_dir.mkdir(exist_ok=False)
    windows_status = export / 'ME06_Launch_Status.json'
    if windows_status.exists():
        raise FileExistsError('Existing launch status requires explicit review; no launcher resume')
    driver, trainer = root / 'portable/complete_reproduction.py', root / 'portable/train_reproduction.py'
    seal = args.adapter_seal or root / 'provenance/portable_adapter_seal.json'
    seal_hash = sha(seal)
    expected = {driver: DRIVER_SHA256, trainer: TRAINER_SHA256}
    def frozen():
        for path, expected_hash in expected.items():
            if path.is_symlink() or sha(path) != expected_hash:
                raise ValueError('Pinned driver/trainer changed: ' + str(path))
        if sha(seal) != seal_hash or read(seal).get('status') != 'REVIEWED':
            raise ValueError('Reviewed adapter seal missing or changed during waiting')
        if (root / 'full_chain').exists() or (root / 'campaign/final_evaluation').exists() or (root / 'comparison').exists():
            raise FileExistsError('Chain/evaluation/comparison already exists; no duplicate launch')
    status = {'status': 'ME06_WAITING_FOR_REAL_PREPARATION', 'started_utc': utc(),
        'launcher_sha256': sha(__file__), 'driver_sha256': DRIVER_SHA256, 'trainer_sha256': TRAINER_SHA256,
        'adapter_seal_sha256': seal_hash, 'max_wait_seconds': args.max_wait_seconds,
        'poll_seconds': 20, 'automatic_retry': False, 'preparation_started_by_launcher': False,
        'root': str(root), 'driver_execution_requested': False}
    start = time.monotonic()
    last_printed = None
    runner_path = root / 'receipts/fresh_preparation.json'
    prep_path = root / 'campaign/prepare_reports/me06_preparation_status.json'
    try:
        while True:
            frozen()
            if sha(__file__) != status['launcher_sha256']:
                raise ValueError('Waiting launcher changed during execution')
            elapsed = time.monotonic() - start
            if elapsed >= args.max_wait_seconds:
                raise TimeoutError('Preparation did not pass within the explicit maximum waiting time')
            runner, runner_error = snapshot_json(runner_path)
            prep, prep_error = snapshot_json(prep_path)
            manifest = root / 'campaign/campaign_manifest.json'
            campaign_hash = sha(manifest) if manifest.is_file() else None
            decision, reason = evaluate_gates(runner, prep, campaign_hash, root)
            probes = root / 'campaign/prepare_reports/native_probes'
            clones = root / 'campaign/prepare_reports/native_clone_probes'
            status.update(updated_utc=utc(), elapsed_wait_seconds=elapsed,
                gate_decision=decision, gate_reason=reason,
                runner_status=runner.get('status') if runner else None,
                preparation_status=prep.get('status') if prep else None,
                preparation_stage=prep.get('stage') if prep else None,
                preparation_receipt_read_issue=runner_error, inner_gate_read_issue=prep_error,
                native_probe_receipts=sum(not p.name.endswith('.command.json') for p in probes.glob('*.json')),
                native_clone_probe_receipts=sum(not p.name.endswith('.command.json') for p in clones.glob('*.json')))
            save(state_dir / 'status.json', status)
            save(windows_status, status)
            marker = (status['runner_status'], status['preparation_status'],
                      status['preparation_stage'], status['native_probe_receipts'], status['native_clone_probe_receipts'])
            if marker != last_printed:
                print('ME06_WAITING_PREPARATION ' + json.dumps({
                    'runner': status['runner_status'], 'inner_gate': status['preparation_status'],
                    'stage': status['preparation_stage'], 'native_probes': status['native_probe_receipts'],
                    'native_clone_probes': status['native_clone_probe_receipts']}, ensure_ascii=False), flush=True)
                last_printed = marker
            if decision == 'STOP':
                raise RuntimeError(reason)
            if decision == 'PASS':
                frozen()
                # Re-read both finished receipts immediately before exec; no stale PASS.
                if evaluate_gates(read(runner_path), read(prep_path), sha(manifest), root)[0] != 'PASS':
                    raise ValueError('Preparation gate drift before driver launch')
                command = driver_command(root, seal, export)
                receipt = {'status': 'EXEC_REQUESTED_AFTER_BOTH_REAL_PREPARATION_GATES_PASS',
                    'created_utc': utc(), 'command': command, 'cwd': str(root),
                    'automatic_retry': False, 'launcher_sha256': status['launcher_sha256'],
                    'driver_sha256': DRIVER_SHA256, 'trainer_sha256': TRAINER_SHA256,
                    'adapter_seal_sha256': seal_hash, 'preparation_runner_receipt_sha256': sha(runner_path),
                    'preparation_gate_sha256': sha(prep_path), 'campaign_manifest_sha256': sha(manifest),
                    'actual_training_completion_claimed': False}
                save(root / 'receipts/full_chain_launch.json', receipt, exclusive=True)
                status.update(status='ME06_DRIVER_EXEC_REQUESTED', driver_execution_requested=True,
                              launch_receipt=str(root / 'receipts/full_chain_launch.json'),
                              follow_progress_at=str(export / 'ME06_Full_Execution_Status.json'))
                save(state_dir / 'status.json', status)
                save(windows_status, status)
                os.chdir(root)
                print('ME06_BOTH_REAL_PREPARATION_GATES_PASS_EXEC_REVIEWED_DRIVER', flush=True)
                os.execv(command[0], command)
                raise RuntimeError('execv unexpectedly returned')
            time.sleep(min(20., args.max_wait_seconds - elapsed))
    except BaseException as exc:
        status.update(status='ME06_LAUNCH_STOP_REVIEW_REQUIRED', completed_utc=utc(),
                      error_type=type(exc).__name__, message=str(exc), automatic_retry=False)
        save(state_dir / 'status.json', status)
        save(windows_status, status)
        (state_dir / 'failure_trace.txt').write_text(traceback.format_exc())
        print('STOP [ME06 waiting launcher]: ' + str(exc), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
