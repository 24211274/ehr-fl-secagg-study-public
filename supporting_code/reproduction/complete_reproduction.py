#!/usr/bin/env python3
"""One authorized fresh ME-06 campaign: 1-cell boundary, remaining 29, tests, comparison.

Run only after fresh preparation and independent adapter review have passed.
Never retry/resume a failed chain or overwrite a previous chain/evaluation.
No scientific implementation is contained here; all three reviewed adapters
and the original release remain frozen and are checked before each stage.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

BENCHMARK_SHA256 = 'b551bfdbe515ba40fc5b147163aad619788566062d5e571339aa9ef60d06700e'
ADAPTER_NAMES = ('train_bind_metadata.py', 'train_reproduction.py', 'evaluate_reproduction.py', 'compare_reproduction.py')
AUTHORIZATION_TEXT = 'هل ستقوم الان بحتى تنفيذ التشغيل والمقارنة.؟'


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
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def proc_identity(pid):
    """PID+start tick+uid prevents signaling an unrelated reused process ID."""
    try:
        base = Path('/proc') / str(pid)
        tail = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        uid = next(line.split()[1] for line in (base / 'status').read_text().splitlines()
                   if line.startswith('Uid:'))
        return {'pid': int(pid), 'ppid': int(tail[1]), 'pgid': int(tail[2]),
                'state': tail[0], 'start_ticks': int(tail[19]), 'uid': int(uid)}
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def collect_owned_descendants(parent_identity, known):
    current = {}
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            identity = proc_identity(int(path.name))
            if identity is not None:
                current[identity['pid']] = identity
    parents = set()
    # A stopped parent's numeric PID is not an ownership proof after reuse.
    if parent_identity is not None:
        actual_parent = current.get(parent_identity['pid'])
        if (actual_parent and actual_parent['uid'] == os.getuid()
                and actual_parent['start_ticks'] == parent_identity['start_ticks']):
            parents.add(parent_identity['pid'])
    for pid, identity in known.items():
        actual = current.get(pid)
        if actual and actual['start_ticks'] == identity['start_ticks'] and actual['uid'] == os.getuid():
            parents.add(pid)
    changed = True
    while changed:
        changed = False
        for pid, identity in current.items():
            if identity['ppid'] in parents and pid not in parents and identity['uid'] == os.getuid():
                known[pid] = identity
                parents.add(pid)
                changed = True


def same_live_process(identity):
    actual = proc_identity(identity['pid'])
    return bool(actual and actual['uid'] == os.getuid()
                and actual['start_ticks'] == identity['start_ticks']
                and actual['state'] not in ('Z', 'X'))


def cleanup_owned_failure(proc, parent_identity, known, grace=30., orphan_grace=5.):
    """Signal only our created process group and proven descendant identities."""
    actions = []
    collect_owned_descendants(parent_identity, known)
    if proc.poll() is None and parent_identity and same_live_process(parent_identity):
        if os.getpgid(proc.pid) != proc.pid:
            raise RuntimeError('Refuse process-group cleanup: child group identity changed')
        try:
            os.killpg(proc.pid, signal.SIGINT)
            actions.append({'signal': 'SIGINT', 'owned_process_group': proc.pid})
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            collect_owned_descendants(parent_identity, known)
    remaining = ([parent_identity] if parent_identity else []) + list(known.values())
    for identity in remaining:
        if same_live_process(identity):
            try:
                os.kill(identity['pid'], signal.SIGTERM)
                actions.append({'signal': 'SIGTERM', 'proven_identity': identity})
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + orphan_grace
    while time.monotonic() < deadline and any(same_live_process(i) for i in remaining):
        time.sleep(.1)
    for identity in remaining:
        if same_live_process(identity):
            try:
                os.kill(identity['pid'], signal.SIGKILL)
                actions.append({'signal': 'SIGKILL', 'proven_identity': identity})
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return {'actions': actions, 'only_owned_processes_signaled': True,
            'remaining_proven_live_processes': [i for i in remaining if same_live_process(i)]}


class StageFailure(RuntimeError):
    pass


def run_stage(name, command, directory, cwd, timeout, heartbeat=None, cleanup_grace=30.):
    """One subprocess attempt. Full logs preserved; no automatic retry of any command."""
    stage = Path(directory) / name
    stage.mkdir(parents=True, exist_ok=False)
    receipt = stage / 'command.json'
    log = stage / 'stdout_stderr.log'
    record = {'stage': name, 'status': 'RUNNING', 'command': [str(x) for x in command],
              'cwd': str(cwd), 'started_utc': utc(), 'timeout_seconds': timeout,
              'automatic_retry': False, 'resume_after_failure': False}
    save(receipt, record, exclusive=True)
    proc = None
    parent_identity = None
    known = {}
    messages = None
    reader = None
    start = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', FLWR_TELEMETRY_ENABLED='0')
    env.pop('PYTHONPATH', None)
    try:
        with log.open('x', encoding='utf-8') as output:
            proc = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace', bufsize=1, start_new_session=True)
            parent_identity = proc_identity(proc.pid)
            record.update(pid=proc.pid, process_identity=parent_identity)
            save(receipt, record)
            messages = queue.Queue()
            def pump():
                try:
                    for line in proc.stdout:
                        messages.put(line)
                finally:
                    messages.put(None)
            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            ended = False
            last_pulse = last_scan = -100.
            while not ended:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise TimeoutError('Explicit stage timeout exceeded: ' + name)
                if elapsed - last_scan >= 2.:
                    collect_owned_descendants(parent_identity, known)
                    last_scan = elapsed
                if heartbeat is not None and elapsed - last_pulse >= 20.:
                    heartbeat(name, elapsed)
                    last_pulse = elapsed
                try:
                    line = messages.get(timeout=min(.5, max(.01, timeout - elapsed)))
                except queue.Empty:
                    # A descendant inheriting stdout must not hide a nonzero parent exit.
                    if proc.poll() is not None and proc.returncode != 0:
                        raise StageFailure(f'{name} exited {proc.returncode}; inspect {log}')
                    continue
                if line is None:
                    ended = True
                else:
                    output.write(line)
                    output.flush()
                    if any(token in line for token in ('ME06_MAIN_RUN_START ', 'ME06_MAIN_RUN_COMPLETE ',
                        'ME06_INTENTIONAL_PAUSE_', 'ME06_ALL_30_', 'ME06_FINAL_EVALUATION_MODEL_',
                        'ME06_COMPARISON_COMPLETE', 'STOP [')):
                        print(line.rstrip(), flush=True)
            code = proc.wait(timeout=max(.1, timeout - (time.monotonic() - start)))
            record['exit_code'] = code
            if code != 0:
                raise StageFailure(f'{name} exited {code}; inspect {log}')
            record['status'] = 'PASS'
    except BaseException as exc:
        record.update(status='FAILED', error_type=type(exc).__name__, message=str(exc),
                      failure_trace=traceback.format_exc())
        if proc is not None:
            record['failure_cleanup'] = cleanup_owned_failure(proc, parent_identity, known, grace=cleanup_grace)
            record['exit_code'] = proc.returncode
        if reader is not None:
            reader.join(timeout=2.)
        if messages is not None:
            with log.open('a', encoding='utf-8') as output:
                while True:
                    try:
                        line = messages.get_nowait()
                    except queue.Empty:
                        break
                    if line is not None:
                        output.write(line)
        raise
    finally:
        record.update(completed_utc=utc(), elapsed_wall_seconds=time.monotonic() - start,
                      stdout_stderr_sha256=sha(log) if log.is_file() else None,
                      owned_descendant_identities=list(known.values()))
        save(receipt, record)
    return record


def check_adapter_seal(root, seal_path):
    seal = read(seal_path)
    if seal.get('status') != 'REVIEWED' or seal.get('scientific_modules_modified') is not False:
        raise ValueError('Reviewed portable adapter seal required before execution')
    rows = seal.get('adapters', [])
    expected = {'portable/' + n for n in ADAPTER_NAMES}
    if len(rows) != len(expected) or {r.get('path') for r in rows} != expected:
        raise ValueError('Adapter seal must cover the three substantive adapters and the reviewed metadata binder')
    for row in rows:
        path = root / row['path']
        if path.is_symlink() or not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('Frozen adapter changed: ' + row['path'])
    return seal


def check_first_boundary(progress, expected_first):
    if progress.get('status') != 'PAUSED_AT_JOB_BOUNDARY' or progress.get('completed_jobs') != [expected_first]:
        raise ValueError('First stage did not stop at the intentional single completed-job boundary')


def check_completed_grid(progress, expected_jobs):
    if progress.get('status') != 'COMPLETE' or progress.get('completed_jobs') != expected_jobs:
        raise ValueError('All 30 fresh training jobs did not complete in planned order')


def export_exclusive(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as f, target.open('xb') as g:
        shutil.copyfileobj(f, g)
    if sha(source) != sha(target):
        raise ValueError('Exported file checksum differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--adapter-seal', type=Path)
    parser.add_argument('--export-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    source, campaign = root / 'source', root / 'campaign'
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12) or os.geteuid() == 0:
        raise RuntimeError('Non-root Linux Python 3.12 required')
    if Path(os.path.abspath(sys.executable)) != root / '.venv/bin/python':
        raise RuntimeError('Use the new run-root .venv/bin/python')
    export = args.export_dir.resolve(strict=True)
    if export.is_relative_to(source) or export.is_relative_to(campaign):
        raise ValueError('Export directory cannot overwrite source/campaign artifacts')
    live_export = export / 'ME06_Full_Execution_Status.json'
    if live_export.exists():
        raise FileExistsError('Previous exported chain status exists; explicit review required')
    seal_path = args.adapter_seal or root / 'provenance/portable_adapter_seal.json'
    seal = check_adapter_seal(root, seal_path)
    seal_hash = sha(seal_path)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source), str(source / 'app'), str(source / 'validation_support')]
    from main_study.common import package_integrity, verify_campaign, require_authorisation
    from main_study.design import plan
    package_hash = package_integrity()
    verify_campaign(campaign)
    campaign_hash = sha(campaign / 'campaign_manifest.json')
    prep = read(campaign / 'prepare_reports/me06_preparation_status.json')
    if (prep.get('status') != 'ME06_FRESH_PREPARATION_COMPLETE_NO_FINAL_TESTS'
            or prep.get('new_reproduction') is not True or prep.get('historical_receipts_reused') is not False
            or prep.get('campaign_manifest_sha256') != campaign_hash
            or prep.get('package_manifest_sha256') != package_hash):
        raise ValueError('Fresh preparation gate does not bind this sealed campaign/source')
    train_hash = next(r['sha256'] for r in seal['adapters'] if r['path'].endswith('train_reproduction.py'))
    if prep.get('wrapper_sha256') != train_hash:
        raise ValueError('Prepared training wrapper differs from the reviewed adapter')
    progress_path = campaign / 'me06_train_progress.json'
    progress = read(progress_path)
    if progress.get('status') != 'READY' or progress.get('completed_jobs') != [] or progress.get('main_training_started') is not False:
        raise ValueError('Driver starts only from an untouched, ready main campaign; no failed-chain resume')
    if (campaign / 'selected_models_manifest.json').exists() or (campaign / 'final_evaluation').exists() or (root / 'comparison').exists():
        raise FileExistsError('Selected models/evaluation/comparison already exists; do not overwrite or rescore')
    if sha(root / 'assets/benchmark.zip') != BENCHMARK_SHA256:
        raise ValueError('Original benchmark archive changed')
    binding = campaign / 'frozen/development/ME06_Metadata_Binding.json'
    raw_binding = root / 'preprocessing/bound_trainval/ME06_Metadata_Binding.json'
    if not binding.is_file() or not raw_binding.is_file() or sha(binding) != sha(raw_binding):
        raise ValueError('Transparent canonical scaler metadata binding receipt missing/drifted')
    binding_record = read(binding)
    binder_hash = next(r['sha256'] for r in seal['adapters'] if r['path'].endswith('train_bind_metadata.py'))
    required_binding = {'status': 'ME06_CANONICAL_METADATA_BOUND_AFTER_EXACT_APPLIED_PARAMETER_AND_ARRAY_CHECKS',
        'new_reproduction': True, 'fresh_raw_outputs_preserved': True,
        'raw_scaler_json_byte_identity_reproduced': False, 'raw_current_replay_bitwise_identical': True,
        'applied_mean_and_scale_exact': True, 'array_count': 30,
        'scientific_parameters_rounded_or_retuned': False, 'original_source_modified': False,
        'old_learned_checkpoints_used': False, 'test_metrics_computed': False,
        'package_manifest_sha256': package_hash, 'wrapper_sha256': binder_hash,
        'canonical_reference_bytes_reused': ['relative_time_scaler.json']}
    if any(binding_record.get(k) != v for k, v in required_binding.items()):
        raise ValueError('Metadata binding receipt does not support the declared provenance scope')
    if (len(binding_record.get('arrays', [])) != 30 or any(
            r.get('reference_bytes_exact') is not True or r.get('reference_values_exact') is not True
            for r in binding_record['arrays']) or binding_record.get('diagnostic_difference', {}).get('float64_ulp_distance') != 1):
        raise ValueError('Metadata binding lacks the exact array/one-ULP diagnostic checks')
    state_dir = root / 'full_chain'
    state_dir.mkdir(exist_ok=False)
    stages = state_dir / 'stages'
    stages.mkdir()
    state_path = state_dir / 'status.json'
    status = {'status': 'ME06_FULL_CHAIN_STARTED', 'started_utc': utc(), 'active_stage': None,
              'new_reproduction': True, 'historical_receipts_reused': False,
              'automatic_retry': False, 'failed_chain_resume': False,
              'driver_sha256': sha(__file__), 'adapter_seal_sha256': seal_hash,
              'package_manifest_sha256': package_hash, 'campaign_manifest_sha256': campaign_hash,
              'benchmark_sha256': BENCHMARK_SHA256,
              'metadata_binding_receipt_sha256': sha(binding),
              'canonical_scaler_metadata_reused': True, 'raw_scaler_json_byte_identity': False,
              'preprocessing_scope': 'Fresh raw/replay outputs preserved; applied scaling parameters and all 30 numeric arrays verified exact. One-ULP diagnostic M2 difference explicitly documented; canonical scaler metadata bytes reused only for legacy hash binding.',
              'user_authorisation': {'date': '2026-10-11', 'source': 'Direct user instruction in this conversation',
                    'text': AUTHORIZATION_TEXT, 'scope': 'All 30 fresh training cells, final 60-cell evaluation, logging and comparison on the fixed supplied snapshot'},
              'completed_stages': [], 'completed_training_jobs': 0, 'planned_training_jobs': 30,
              'planned_final_metric_cells': 60, 'scope': 'Fresh environment on the same WSL host; not external-host reproduction or original data-generator recreation',
              'linux_run_root': str(root), 'windows_export_directory': str(export)}

    def update(stage=None, elapsed=None):
        if stage is not None:
            status['active_stage'] = stage
        if elapsed is not None:
            status['active_stage_elapsed_seconds'] = elapsed
        current = read(progress_path)
        status['completed_training_jobs'] = len(current['completed_jobs'])
        status['training_progress_status'] = current['status']
        status['updated_utc'] = utc()
        save(state_path, status)
        save(live_export, status)

    def recheck():
        if sha(__file__) != status['driver_sha256']:
            raise ValueError('Full-chain driver changed during execution')
        if sha(seal_path) != seal_hash:
            raise ValueError('Reviewed adapter seal changed during execution')
        check_adapter_seal(root, seal_path)
        if sha(campaign / 'campaign_manifest.json') != campaign_hash:
            raise ValueError('Campaign seal changed during execution')
        verify_campaign(campaign)

    def execute(name, command, timeout):
        recheck()
        update(name, 0.)
        print('ME06_FULL_CHAIN_STAGE_START ' + name, flush=True)
        result = run_stage(name, command, stages, root, timeout, update)
        status['completed_stages'].append({'stage': name, 'command_receipt': str(stages / name / 'command.json'),
                                           'exit_code': result['exit_code'], 'elapsed_seconds': result['elapsed_wall_seconds']})
        update(name, result['elapsed_wall_seconds'])
        recheck()
        print('ME06_FULL_CHAIN_STAGE_PASS ' + name, flush=True)

    update()
    rc = 1
    try:
        authorisations = campaign / 'authorisations'
        authorisations.mkdir(exist_ok=True)
        train_receipt = authorisations / 'ME06_FULL_CHAIN_TRAIN_20261011.json'
        common_receipt = {'approved': True, 'new_reproduction': True,
            'campaign_manifest_sha256': campaign_hash, 'package_manifest_sha256': package_hash,
            'authorisation_basis': status['user_authorisation'], 'created_utc': utc(),
            'historical_approval_reused': False, 'driver_sha256': status['driver_sha256']}
        save(train_receipt, {**common_receipt, 'action': 'TRAIN', 'test_data_authorised': False}, exclusive=True)
        require_authorisation(train_receipt, 'TRAIN', campaign)
        python = root / '.venv/bin/python'
        train_command = [python, '-I', '-B', root / 'portable/train_reproduction.py', 'train',
                         '--run-root', root, '--source', source, '--campaign', campaign,
                         '--authorisation', train_receipt]
        execute('01_first_fresh_training_cell', train_command + ['--max-jobs', '1'], 25200)
        progress = read(progress_path)
        expected_first = plan()[0]['planned_run_id']
        check_first_boundary(progress, expected_first)
        save(state_dir / 'first_cell_boundary_check.json', {'status': 'PASS', 'checked_utc': utc(),
             'completed_jobs': progress['completed_jobs'], 'continuation_is_not_a_failed_job_retry': True}, exclusive=True)
        execute('02_remaining_29_fresh_training_cells', train_command, 86400)
        progress = read(progress_path)
        check_completed_grid(progress, [r['planned_run_id'] for r in plan()])
        from main_study.final_evaluator import validate_models
        models = validate_models(campaign)
        completion = read(campaign / 'main_completion_reports/me06_completion_status.json')
        selected_hash = sha(campaign / 'selected_models_manifest.json')
        if (len(models) != 30 or completion.get('status') != 'ME06_ALL_30_FRESH_MODELS_CALIBRATED_FINAL_TESTS_CLOSED'
                or completion.get('selected_models_manifest_sha256') != selected_hash
                or completion.get('campaign_manifest_sha256') != campaign_hash
                or completion.get('test_data_opened') is not False):
            raise ValueError('Fresh full-model seal is incomplete or final tests already opened')
        evaluation_receipt = authorisations / 'ME06_FULL_CHAIN_FINAL_TESTS_20261011.json'
        save(evaluation_receipt, {**common_receipt, 'created_utc': utc(),
             'action': 'EVALUATE_FINAL_TESTS', 'test_data_authorised': True,
             'selected_models_manifest_sha256': selected_hash,
             'benchmark_sha256': BENCHMARK_SHA256, 'validation_only_models_calibration_locked_before_test': True}, exclusive=True)
        require_authorisation(evaluation_receipt, 'EVALUATE_FINAL_TESTS', campaign, expected_models_sha=selected_hash)
        execute('03_fresh_final_evaluation_60_cells',
            [python, '-I', '-B', root / 'portable/evaluate_reproduction.py', '--run-root', root,
             '--receipt', evaluation_receipt, '--expected-benchmark-sha256', BENCHMARK_SHA256], 14400)
        execute('04_full_precision_comparison',
            [python, '-I', '-B', root / 'portable/compare_reproduction.py', '--run-root', root,
             '--reference-assets', root / 'reference/original_assets'], 7200)
        comparison = read(root / 'comparison/comparison_status.json')
        if not comparison.get('status', '').startswith('ME06_COMPARISON_COMPLETE_'):
            raise ValueError('Comparison did not complete')
        expected_comparison = {'models': 30, 'metric_cells': 60, 'score_rows_compared': 90270,
            'selected_tensor_state_comparisons_available': 30, 'test_identity_and_labels_exact': True,
            'test_encoding_coverage_exact': True}
        if any(comparison.get(k) != v for k, v in expected_comparison.items()):
            raise ValueError('Comparison does not cover every required model, tensor, metric and score row')
        status.update(status='ME06_FULL_CHAIN_COMPLETE', active_stage=None,
            all_execution_stages_completed=True, final_metric_cells=60,
            comparison_status=comparison['status'], comparison_status_sha256=sha(root / 'comparison/comparison_status.json'),
            reproduction_conclusion='See exact differences and scope in the comparison report; execution completion is not clinical validity, global equivalence or external-host confirmation.',
            final_linux_comparison_report=str(root / 'comparison/Comparison_Report.md'))
        for local, filename in [(root / 'comparison/Comparison_Report.md', 'ME06_Comparison_Report.md'),
            (root / 'comparison/comparison_status.json', 'ME06_Comparison_Status.json'),
            (root / 'comparison/all_60_primary_metric_cells.csv', 'ME06_All_60_Primary_Metric_Cells.csv'),
            (root / 'comparison/all_seed_means_and_paired_contrasts.csv', 'ME06_Seed_Summaries_And_Paired_Contrasts.csv'),
            (binding, 'ME06_Metadata_Binding.json')]:
            export_exclusive(local, export / filename)
        status['exported_final_report'] = str(export / 'ME06_Comparison_Report.md')
        rc = 0
    except BaseException as exc:
        status.update(status='ME06_FULL_CHAIN_FAILED_REVIEW_REQUIRED', error_type=type(exc).__name__,
                      message=str(exc), failure_stage=status.get('active_stage'), all_execution_stages_completed=False,
                      automatic_retry=False, failed_chain_resume=False)
        (state_dir / 'failure_trace.txt').write_text(traceback.format_exc())
        print('STOP [ME06 full chain]: ' + str(exc), flush=True)
    finally:
        status.update(completed_utc=utc(), exit_code=rc)
        update()
        entries = [{'path': p.relative_to(state_dir).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in sorted(state_dir.rglob('*')) if p.is_file()]
        save(state_dir / 'full_chain_files.json', {'files': entries}, exclusive=True)
    print(status['status'], flush=True)
    return rc


if __name__ == '__main__':
    # A competing chain cannot acquire this lock; no other process is interrupted.
    import fcntl
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--run-root', type=Path)
    known, _ = preliminary.parse_known_args()
    if known.run_root is None:
        raise SystemExit(main())
    candidate = known.run_root.resolve(strict=True)
    with (candidate / '.me06_full_chain.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main())
