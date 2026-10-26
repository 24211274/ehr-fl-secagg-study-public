#!/usr/bin/env python3
"""Portable ME-06 orchestration around the unchanged scientific study release.

This is a NEW reproduction wrapper, not a historical preparation receipt.
No original module is patched, no old learned checkpoint is a training input,
and failed/interrupted jobs are never automatically retried. All model, loss,
optimiser, schedule, calibration, measurement and selection code comes from
the sealed source release. Only path management and evidence recording differ.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if exclusive:
        with p.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = p.with_name(p.name + '.tmp')
        with tmp.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)


def command(args, log, timeout, cwd, state=None):
    """Single attempt. No pip, retries, shell, or replacement of prior logs."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               FLWR_TELEMETRY_ENABLED='0')
    env['PATH'] = str(Path(os.path.abspath(sys.executable)).parent) + os.pathsep + env.get('PATH', '')
    env.pop('PYTHONPATH', None)
    started = utc()
    t0 = time.monotonic()
    receipt = log.with_suffix('.command.json')
    record = {'status': 'RUNNING', 'started_utc': started,
              'command': [str(a) for a in args], 'cwd': str(cwd),
              'timeout_seconds': timeout, 'new_reproduction': True,
              'automatic_retry': False}
    save(receipt, record, exclusive=True)
    with log.open('x', encoding='utf-8') as f:
        p = subprocess.Popen([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env, cwd=cwd)
        try:
            p.wait(timeout=timeout)
        except BaseException as e:
            # Permit the unchanged worker/owned-service finally blocks to clean up.
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            record.update(status='FAILED', error_type=type(e).__name__, message=str(e),
                          finished_utc=utc(), exit_code=p.returncode,
                          elapsed_seconds=time.monotonic() - t0, log_sha256=digest(log))
            save(receipt, record)
            raise
    record.update(status='COMPLETE' if p.returncode == 0 else 'FAILED',
                  finished_utc=utc(), exit_code=p.returncode,
                  elapsed_seconds=time.monotonic() - t0, log_sha256=digest(log))
    save(receipt, record)
    if p.returncode:
        raise RuntimeError(f'Worker exit {p.returncode}; inspect {log}')


def setup(args):
    if sys.platform != 'linux' or os.geteuid() == 0:
        raise RuntimeError('Reproduction must execute as the non-root Linux user')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Reproduction requires Python 3.12')
    args.source = args.source.resolve()
    args.campaign = args.campaign.resolve()
    args.run_root = args.run_root.resolve()
    if args.source != args.run_root / 'source':
        raise ValueError('Use the sealed new run-root/source copy, not an original checkout')
    if not args.campaign.is_relative_to(args.run_root) or args.campaign == args.run_root:
        raise ValueError('Campaign must be a child of the explicit run root')
    if args.campaign.is_relative_to(args.source) or args.source.is_relative_to(args.campaign):
        raise ValueError('Runnable source and new campaign must be disjoint')
    expected_python = args.run_root / '.venv/bin/python'
    if Path(os.path.abspath(sys.executable)) != expected_python:
        raise RuntimeError('Use the new run-root .venv/bin/python explicitly')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(args.source), str(args.source / 'app'),
                    str(args.source / 'validation_support')]
    from main_study.common import package_integrity
    return package_integrity()


def provenance(args, package_hash):
    return {'new_reproduction': True, 'historical_receipts_reused': False,
            'wrapper': str(Path(__file__).resolve()), 'wrapper_sha256': digest(__file__),
            'source': str(args.source), 'package_manifest_sha256': package_hash,
            'python_executable': sys.executable, 'python': platform.python_version(),
            'platform': platform.platform(), 'user_uid': os.geteuid(),
            'run_root': str(args.run_root), 'recorded_utc': utc(),
            'training_from_old_learned_checkpoints': False,
            'scientific_modules_modified': False,
            'scope': 'Fresh environment and rerun on the same host, not an external independent researcher or new dataset generation'}


def environment_identity(args):
    """Fingerprint the complete new venv, excluding incidental Python bytecode."""
    import importlib.metadata
    venv = args.run_root / '.venv'
    records = []
    for p in sorted(venv.rglob('*')):
        if '__pycache__' in p.parts or p.suffix in ('.pyc', '.pyo'):
            continue
        if p.is_file():
            row = {'path': p.relative_to(venv).as_posix(), 'bytes': p.stat().st_size,
                   'sha256': digest(p)}
            if p.is_symlink():
                row['symlink_target'] = os.readlink(p)
            records.append(row)
    if not records:
        raise ValueError('Empty new environment inventory')
    distributions = sorted([{'name': d.metadata['Name'], 'version': d.version}
                            for d in importlib.metadata.distributions()], key=lambda r: (r['name'].lower(), r['version']))
    return {'python_executable': sys.executable, 'python': platform.python_version(),
            'python_binary_sha256': digest(sys.executable),
            'distributions': distributions, 'files': records,
            'scope': 'Complete new venv file identities; pycache/pyc excluded; no historical environment receipt reused'}


def check_environment(args, baseline, receipt):
    current = environment_identity(args)
    equal = current == baseline
    save(receipt, {'status': 'PASS' if equal else 'FAIL', 'checked_utc': utc(),
                  'new_reproduction': True, 'complete_new_venv_equal_to_preparation': equal,
                  'files_checked': len(current['files']), 'distributions_checked': len(current['distributions'])}, exclusive=True)
    if not equal:
        save(Path(receipt).with_name(Path(receipt).stem + '_changed_inventory.json'), current, exclusive=True)
        raise ValueError('New environment changed since fresh preparation')


def verify_development(prepared, reports):
    """Read fresh encoded TRAIN/VALIDATION only; derive schedule inputs anew."""
    from main_study.common import ROOT, CLIENTS, read, read_csv, sha, write
    from main_study.design import department_rows
    from ehr_preprocessing.io_utils import verify_artefacts
    from ehr_preprocessing.dataset import SessionDataset
    if not prepared.is_dir() or prepared.is_symlink():
        raise ValueError('Fresh prepared trainval directory is required')
    if prepared.name == 'bound_trainval' and not (prepared / 'ME06_Metadata_Binding.json').is_file():
        raise ValueError('Bound development view requires its explicit metadata binding receipt')
    if any(p.is_symlink() for p in prepared.rglob('*')):
        raise ValueError('No symlinks in fresh preprocessing inputs')
    files = verify_artefacts(prepared)
    conf = read(ROOT / 'reference/fullsize_B.json')
    if sha(prepared / 'encoding_schema.json') != conf['schema_sha256']:
        raise ValueError('Fresh schema differs from frozen scientific design')
    if sha(prepared / 'relative_time_scaler.json') != conf['scaler_sha256']:
        raise ValueError('Fresh scaler differs from frozen scientific design')
    rows = read_csv(prepared / 'session_manifest.csv')
    members = read_csv(ROOT / 'reference/full_membership.csv')
    if [{k: r[k] for k in members[0]} for r in rows] != members:
        raise ValueError('Fresh development membership differs')
    if len(rows) != 14455 or len({r['session_id'] for r in rows}) != len(rows):
        raise ValueError('Development sessions count/uniqueness differs')
    if any(r['partition'] not in ('training', 'validation') for r in rows):
        raise ValueError('Final-test data appeared in development')
    schema = read(prepared / 'encoding_schema.json')
    field = schema['categorical_fields'][2]
    if field['name'] != 'department':
        raise ValueError('Department encoding position differs')
    fresh = []
    training_counts = {}
    for c in CLIENTS:
        dataset = SessionDataset(prepared, c, 'training', verify=False)
        training_counts[c] = len(dataset)
        for i in range(len(dataset)):
            s = dataset[i]
            if not s['categorical_valid'][:, 2].all():
                raise ValueError('Unknown training department')
            counts = Counter(field['categories'][int(x)] for x in s['categorical'][:, 2])
            mode = sorted(counts, key=lambda k: (-counts[k], k))[0]
            member = next(r for r in members if r['client'] == c and r['session_id'] == s['session_id'])
            fresh.append({'client': c, 'session_id': s['session_id'], 'department': mode,
                          'hospital_id': member['hospital_id']})
    fresh.sort(key=lambda r: (r['client'], r['session_id']))
    if fresh != department_rows():
        raise ValueError('Fresh training modal departments differ from original schedule design')
    if sum(training_counts.values()) != 10833:
        raise ValueError('Training session count differs')
    result = {'status': 'PASS', 'fresh_preprocessing_files_verified': len(files),
              'sessions': len(rows), 'training_counts': training_counts,
              'training_modes_rederived': len(fresh), 'membership_equal_reference': True,
              'schema_scaler_equal_reference': True, 'test_data_opened': False,
              'historical_preprocessing_receipts_reused': False}
    write(reports / 'development_audit.json', result, exclusive=True)
    return fresh


def compare_generated_reference(campaign, reference_zip, reports):
    """Frozen old initials/schedules are comparison evidence, never training input."""
    from main_study.common import SEEDS, read, sha, write
    import io
    import torch
    import numpy as np
    frozen = campaign / 'frozen'
    comparisons = []
    with zipfile.ZipFile(reference_zip) as z:
        original = json.loads(z.read('campaign/frozen/initial_states.json'))
        current = read(frozen / 'initial_states.json')
        for seed in SEEDS:
            key = str(seed)
            old_bytes = z.read(f'campaign/frozen/initials/seed_{seed}.pt')
            old_state = torch.load(io.BytesIO(old_bytes), map_location='cpu', weights_only=True)
            new_state = torch.load(frozen / 'initials' / f'seed_{seed}.pt', map_location='cpu', weights_only=True)
            equal = list(old_state) == list(new_state) and all(torch.equal(old_state[k], new_state[k]) for k in old_state)
            same_digest = current[key]['state_digest'] == original[key]['state_digest']
            row = {'kind': 'initial', 'seed': seed, 'state_tensors_exact': bool(equal),
                   'state_digest_exact': same_digest,
                   'serialized_file_sha256_exact': sha(frozen / 'initials' / f'seed_{seed}.pt') == hashlib.sha256(old_bytes).hexdigest()}
            comparisons.append(row)
            if not equal or not same_digest:
                raise ValueError(f'Fresh initial state differs: seed {seed}')
            del old_state, new_state
        for seed in SEEDS:
            for co in ('reference', 'noniid'):
                name = f'{seed}_{co}.csv'
                old_sha = hashlib.sha256(z.read('campaign/frozen/schedules/' + name)).hexdigest()
                new_sha = sha(frozen / 'schedules' / name)
                comparisons.append({'kind': 'schedule', 'seed': seed, 'condition': co,
                                    'sha256_exact': old_sha == new_sha, 'sha256': new_sha})
                if old_sha != new_sha:
                    raise ValueError('Fresh draw schedule differs: ' + name)
        # All 30 development arrays must equal the archived reference numerically;
        # other fresh metadata may legitimately contain new execution provenance.
        for p in sorted((frozen / 'development').rglob('*.npy')):
            name = 'campaign/frozen/development/' + p.relative_to(frozen / 'development').as_posix()
            old_bytes = z.read(name)
            old = np.load(io.BytesIO(old_bytes), allow_pickle=False)
            new = np.load(p, allow_pickle=False)
            equal = old.dtype == new.dtype and old.shape == new.shape and np.array_equal(old, new)
            comparisons.append({'kind': 'development_array', 'path': name,
                                'dtype_shape_values_exact': bool(equal),
                                'serialized_file_sha256_exact': sha(p) == hashlib.sha256(old_bytes).hexdigest()})
            if not equal:
                raise ValueError('Fresh development array differs: ' + name)
    binding_path = frozen / 'development/ME06_Metadata_Binding.json'
    binding = None
    if binding_path.exists():
        binding = read(binding_path)
        if (binding.get('status') != 'ME06_CANONICAL_METADATA_BOUND_AFTER_EXACT_APPLIED_PARAMETER_AND_ARRAY_CHECKS'
                or binding.get('new_reproduction') is not True
                or binding.get('raw_scaler_json_byte_identity_reproduced') is not False
                or binding.get('applied_mean_and_scale_exact') is not True
                or binding.get('all_other_scaler_json_fields_exact') is not True
                or binding.get('array_count') != 30
                or binding.get('canonical_reference_bytes_reused') != ['relative_time_scaler.json']
                or binding.get('canonical_scaler_sha256') != sha(frozen / 'development/relative_time_scaler.json')):
            raise ValueError('Canonical metadata binding receipt is invalid')
    result = {'status': 'PASS', 'reference_archive_sha256': digest(reference_zip),
              'reference_initials_schedules_arrays_used_as_training_inputs': False,
              'canonical_scaler_metadata_reused': binding is not None,
              'canonical_scaler_binding_receipt_sha256': sha(binding_path) if binding is not None else None,
              'raw_scaler_json_byte_identity_reproduced': False if binding is not None else True,
              'comparison_scope': '5 initial tensor states, 10 complete schedules, 30 fresh encoded development arrays; any canonical metadata exception is explicitly bound by its separate receipt',
              'comparisons': comparisons, 'new_reproduction': True}
    write(reports / 'fresh_vs_frozen_inputs.json', result, exclusive=True)
    return result


def prepare(args, package_hash):
    from main_study.common import (ROOT, SEEDS, inventory, file_record, read, write,
                                   sha, csv_write, verify_campaign)
    from main_study.design import (plan, decisions, make_draws, distribution_audit,
                                  cell_config, stage_cell, DRAW_FIELDS)
    from main_study.staging import stage_attempt_job
    campaign = args.campaign
    campaign.mkdir(parents=True, exist_ok=False)
    reports = campaign / 'prepare_reports'
    reports.mkdir()
    stamp = provenance(args, package_hash)
    save(reports / 'me06_provenance.json', stamp, exclusive=True)
    baseline = environment_identity(args)
    save(reports / 'me06_environment_baseline.json', baseline, exclusive=True)
    status = {**stamp, 'status': 'ME06_FRESH_PREPARATION_RUNNING',
              'stage': 'verify_development', 'main_training_performed': False,
              'test_data_opened': False, 'privacy_certification': False}
    save(reports / 'me06_preparation_status.json', status)
    try:
        if shutil.disk_usage(campaign).free < 8 * 1024**3:
            raise RuntimeError('At least 8 GiB free required')
        pre = args.prepared.resolve()
        if not pre.is_relative_to(args.run_root) or pre.is_relative_to(campaign):
            raise ValueError('Fresh preprocessing must be inside run root and outside campaign')
        if args.preprocessing_receipt is None or not args.preprocessing_receipt.is_file():
            raise ValueError('New raw-preprocessing receipt must be supplied explicitly')
        receipt_path = args.preprocessing_receipt.resolve()
        if not receipt_path.is_relative_to(args.run_root) or receipt_path.is_relative_to(args.source):
            raise ValueError('Preprocessing receipt must belong to the new run root, outside sealed source')
        fresh_receipt = load(receipt_path)
        if (fresh_receipt.get('status') != 'PREPROCESSING_GATE_PASS_NOT_FULL_PILOT'
                or fresh_receipt.get('mode') != 'raw_source_rebuild'
                or fresh_receipt.get('source_rebuilt_here') is not True):
            raise ValueError('Original preprocessing runner must report a successful raw rebuild here')
        if fresh_receipt.get('test_tensors_created') is not False or fresh_receipt.get('test_metrics_computed') is not False:
            raise ValueError('Fresh preprocessing must not create final-test tensors or metrics')
        replay_path = receipt_path.parent / 'replay.json'
        replay = load(replay_path)
        if replay.get('bitwise_identical') is not True:
            raise ValueError('Fresh preprocessing replay must be bitwise identical')
        save(reports / 'fresh_preprocessing_receipt_binding.json',
             {'path': str(args.preprocessing_receipt.resolve()), 'sha256': digest(args.preprocessing_receipt),
              'fresh_receipt': fresh_receipt, 'new_reproduction': True,
              'fresh_replay_sha256': digest(replay_path), 'fresh_replay': replay}, exclusive=True)
        departments = verify_development(pre, reports)
        frozen = campaign / 'frozen'
        frozen.mkdir()
        shutil.copytree(pre, frozen / 'development')
        shutil.copyfile(ROOT / 'protocol/decisions/config/main_run_order.csv', frozen / 'main_run_order.csv')
        write(frozen / 'provenance_limits.json', decisions()['data'], exclusive=True)
        write(campaign / 'origins.json', {'preprocessing_run': str(pre.parent),
              'fresh_environment': str(args.run_root / '.venv'),
              'source': str(args.source), 'new_reproduction': True,
              'historical_accepted_run_used_as_environment': False}, exclusive=True)
        status['stage'] = 'native_unit_suite'
        save(reports / 'me06_preparation_status.json', status)
        command([sys.executable, '-I', '-B', ROOT / 'scripts/run_main_tests.py', reports / 'unit_checks.json'],
                reports / 'unit_checks.log', 1800, args.run_root)
        unit = read(reports / 'unit_checks.json')
        if unit['status'] != 'PASS' or unit['skipped'] or not unit['tests_run']:
            raise ValueError('Native unit suite must pass with no skipped tests')
        print('ME06_NEW_NATIVE_UNIT_SUITE_PASS', flush=True)
        status['stage'] = 'candidate_public_toy_protocol'
        save(reports / 'me06_preparation_status.json', status)
        from prg_validation.public_protocol import run_oracle
        oracle = run_oracle('candidate')
        if oracle['status'] != 'PASS':
            raise ValueError('Fresh candidate public toy protocol failed')
        write(reports / 'public_protocol.json', oracle, exclusive=True)
        from ehr_fedavg.task import setup_torch, model_factory
        from pilot_core.model import set_training_seed, state_digest
        import torch
        setup_torch()
        initials = {}
        for seed in SEEDS:
            set_training_seed(seed)
            model = model_factory()
            path = frozen / 'initials' / f'seed_{seed}.pt'
            path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), path)
            initials[str(seed)] = {'state_digest': state_digest(model), 'sha256': sha(path)}
            second = model_factory()
            second.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            if state_digest(second) != initials[str(seed)]['state_digest']:
                raise ValueError('Fresh initial checkpoint reload failed')
        if len({r['state_digest'] for r in initials.values()}) != 5:
            raise ValueError('Training seeds do not create five distinct states')
        write(frozen / 'initial_states.json', initials, exclusive=True)
        audits = []
        schedules = {}
        for seed in SEEDS:
            nr = make_draws(seed, 'noniid', 30, departments)
            audit = distribution_audit(departments, nr)
            audit['seed'] = seed
            audits.append(audit)
            for co in ('reference', 'noniid'):
                rows = make_draws(seed, co, 30, departments) if co == 'reference' else nr
                p = frozen / 'schedules' / f'{seed}_{co}.csv'
                csv_write(p, rows, DRAW_FIELDS)
                schedules[(seed, co)] = p
        write(reports / 'sampling_audits.json', {'audits': audits, 'exact_occurrences': 10833*30*10,
              'test_data_used': False, 'fresh_training_modes_used': True}, exclusive=True)
        compare_generated_reference(campaign, args.frozen_reference, reports)
        print('ME06_FIVE_FRESH_INITIALS_AND_TEN_FRESH_SCHEDULES_EXACT_REFERENCE_PASS', flush=True)
        status['stage'] = 'native_job_template_and_clone_probes'
        save(reports / 'me06_preparation_status.json', status)
        for row in plan():
            seed = int(row['training_seed'])
            co, mode = row['condition'], row['configuration']
            conf = cell_config(seed, co, 30)
            schedule = schedules[(seed, co)]
            conf['schedule_sha256'] = sha(schedule)
            job = stage_cell(campaign, campaign / 'jobs' / row['planned_run_id'], conf,
                             schedule, initials[str(seed)]['state_digest'], mode)
            command([sys.executable, '-I', '-B', ROOT / 'scripts/job_worker.py', job, 'Probe'],
                    reports / 'native_probes' / f'{job.name}.log', 600, args.run_root)
            shutil.copyfile(job / 'reports/job_preflight.json', reports / 'native_probes' / f'{job.name}.json')
            if mode in ('B', 'C'):
                clone = stage_attempt_job(job, campaign / 'attempt_clone_probes' / job.name)
                command([sys.executable, '-I', '-B', ROOT / 'scripts/job_worker.py', clone, 'Probe'],
                        reports / 'native_clone_probes' / f'{job.name}.log', 600, args.run_root)
                shutil.copyfile(clone / 'reports/job_preflight.json', reports / 'native_clone_probes' / f'{job.name}.json')
            print('ME06_FRESH_JOB_PROBES_PASS ' + row['planned_run_id'], flush=True)
        # Same original engineering compatibility epoch, not one of the 30 outcomes.
        status['stage'] = 'central_fullsize_engineering_epoch'
        save(reports / 'me06_preparation_status.json', status)
        seed = 20260906
        set_training_seed(seed)
        expected = state_digest(model_factory())
        schedule = campaign / 'engineering_draws.csv'
        csv_write(schedule, make_draws(seed, 'reference', 1, departments), DRAW_FIELDS)
        conf = cell_config(seed, 'reference', 1, engineering=True)
        conf['schedule_sha256'] = sha(schedule)
        job = stage_cell(campaign, campaign / 'engineering_A', conf, schedule, expected, 'A')
        command([sys.executable, '-I', '-B', ROOT / 'scripts/job_worker.py', job, 'Probe'],
                reports / 'engineering_A_probe.log', 600, args.run_root)
        command([sys.executable, '-I', '-B', ROOT / 'scripts/job_worker.py', job, 'Train'],
                reports / 'engineering_A_train.log', 4000, args.run_root)
        result = read(job / 'central_results/result.json')
        if result['local_updates'] != 170 or result['partial_last_batch_sessions'] != 17:
            raise ValueError('Full-size A engineering epoch update accounting differs')
        shutil.copyfile(job / 'central_results/result.json', reports / 'engineering_A_summary.json')
        files = [file_record(p, campaign) for p in sorted(frozen.rglob('*')) if p.is_file()]
        for row in plan():
            j = campaign / 'jobs' / row['planned_run_id']
            for directory in ('apps', 'client_data'):
                files.extend(file_record(p, campaign) for p in sorted((j / directory).rglob('*'))
                             if p.is_file() and p.suffix not in ('.pyc', '.pyo', '.fab') and '__pycache__' not in p.parts)
            files.extend(file_record(j / n, campaign) for n in ('runtime.json', 'job.json'))
        files.append(file_record(campaign / 'origins.json', campaign))
        files.append(file_record(reports / 'me06_provenance.json', campaign))
        files.append(file_record(reports / 'me06_environment_baseline.json', campaign))
        write(campaign / 'campaign_manifest.json', {'version': 'ME06-portable-wrapper-v1',
              'package_manifest_sha256': package_hash,
              'decision_sha256': sha(ROOT / 'protocol/decisions/config/main_protocol_decisions.json'),
              'files': files, 'new_reproduction': True, 'historical_receipts_reused': False,
              'training_authorised': True, 'test_authorised': False,
              'authorisation_basis': 'User explicitly requested fresh full execution and comparison in this chat on 2026-10-11',
              'public_availability_verified': False, 'licence_verified': False}, exclusive=True)
        verify_campaign(campaign)
        check_environment(args, baseline, reports / 'me06_environment_after_preparation.json')
        save(campaign / 'me06_train_progress.json', {'status': 'READY', 'completed_jobs': [],
             'completed_blocks': [], 'blocks': {}, 'main_training_started': False,
             'new_reproduction': True, 'automatic_retry': False}, exclusive=True)
        status.update(status='ME06_FRESH_PREPARATION_COMPLETE_NO_FINAL_TESTS', stage='prepared',
                      jobs_prepared=30, native_template_probes=30, native_relocated_clone_probes=20,
                      main_training_performed=False, engineering_updates=170,
                      campaign_manifest_sha256=sha(campaign / 'campaign_manifest.json'), finished_utc=utc())
        save(reports / 'me06_preparation_status.json', status)
        print(status['status'], flush=True)
    except BaseException as e:
        status.update(status='ME06_FRESH_PREPARATION_FAILED_NO_AUTORETRY',
                      error_type=type(e).__name__, message=str(e), finished_utc=utc())
        save(reports / 'me06_preparation_status.json', status)
        (reports / 'me06_preparation_failure.txt').write_text(traceback.format_exc())
        raise


def validate_finished(job):
    from main_study.campaign import validate_finished_job
    from main_study.common import read, sha
    s = validate_finished_job(job)
    meta = read(job / 'job.json')
    scores = job / 'LOCAL_ONLY/selected_validation_scores.json'
    if sha(scores) != s['validation_scores_sha256']:
        raise ValueError('Finished validation scores altered')
    if s['rounds'] != 30 or s['seed'] != meta['training_seed'] or s['test_data_opened'] is not False:
        raise ValueError('Finished cell identity/budget/test boundary differs')
    return s


def seal(args, package_hash, progress):
    from main_study.common import read, write, sha, verify_campaign
    from main_study.design import plan, ordered_blocks
    from main_study.costs import campaign_summary
    from main_study.final_evaluator import validate_models
    campaign = args.campaign
    if progress['completed_jobs'] != [r['planned_run_id'] for r in plan()] or progress['completed_blocks'] != ordered_blocks():
        raise ValueError('All 30 jobs and ten matched blocks must complete in order')
    verify_campaign(campaign)
    models = []
    for row in plan():
        entry = progress['blocks'][row['block_id']]
        job = campaign / entry['path'] / row['configuration']
        s = validate_finished(job)
        for key, value in [('configuration', row['configuration']), ('condition', row['condition']), ('seed', int(row['training_seed']))]:
            if s[key] != value:
                raise ValueError('Final model identity differs')
        models.append({**s, 'job': job.relative_to(campaign).as_posix(),
                      'checkpoint': (job / 'checkpoints' / row['condition'] / 'selected_state.pt').relative_to(campaign).as_posix(),
                      'calibration': (job / 'calibration.json').relative_to(campaign).as_posix()})
    selected = {'models': models, 'test_data_authorised': False,
                'campaign_manifest_sha256': sha(campaign / 'campaign_manifest.json'),
                'package_manifest_sha256': package_hash, 'new_reproduction': True,
                'historical_selected_models_reused': False}
    target = campaign / 'selected_models_manifest.json'
    if target.exists():
        if read(target) != selected:
            raise ValueError('Existing selected-model seal differs; no overwrite')
    else:
        write(target, selected, exclusive=True)
    validate_models(campaign)
    reports = campaign / 'main_completion_reports'
    reports.mkdir(exist_ok=True)
    cost = reports / 'system_cost_analysis.json'
    if not cost.exists():
        write(cost, campaign_summary(campaign, models), exclusive=True)
    write(reports / 'me06_completion_status.json', {**provenance(args, package_hash),
          'status': 'ME06_ALL_30_FRESH_MODELS_CALIBRATED_FINAL_TESTS_CLOSED',
          'main_training_complete': True, 'models': 30, 'test_data_opened': False,
          'selected_models_manifest_sha256': sha(target),
          'campaign_manifest_sha256': sha(campaign / 'campaign_manifest.json'),
          'privacy_certification': False})
    progress['status'] = 'COMPLETE'
    save(campaign / 'me06_train_progress.json', progress)
    print('ME06_ALL_30_FRESH_MODELS_CALIBRATED_FINAL_TESTS_CLOSED', flush=True)


def train(args, package_hash):
    from main_study.common import ROOT, read, sha, verify_campaign, require_authorisation
    from main_study.design import plan, ordered_blocks
    from main_study.staging import stage_attempt_job
    campaign = args.campaign
    verify_campaign(campaign)
    prep = read(campaign / 'prepare_reports/me06_preparation_status.json')
    if prep['status'] != 'ME06_FRESH_PREPARATION_COMPLETE_NO_FINAL_TESTS':
        raise ValueError('Fresh preparation is not complete')
    if prep['wrapper_sha256'] != digest(__file__):
        raise ValueError('Wrapper changed after preparation; review required')
    if prep['campaign_manifest_sha256'] != sha(campaign / 'campaign_manifest.json'):
        raise ValueError('Campaign seal differs from fresh preparation')
    if args.authorisation is None:
        raise PermissionError('A new run-specific TRAIN receipt must be supplied')
    approval_path = args.authorisation.resolve()
    if not approval_path.is_relative_to(campaign / 'authorisations'):
        raise PermissionError('Training receipt must be in this new campaign/authorisations')
    approval = require_authorisation(approval_path, 'TRAIN', campaign)
    if approval.get('new_reproduction') is not True:
        raise PermissionError('Historical training receipts are not accepted')
    baseline = read(campaign / 'prepare_reports/me06_environment_baseline.json')
    path = campaign / 'me06_train_progress.json'
    progress = read(path)
    rows = plan()
    ids = [r['planned_run_id'] for r in rows]
    if progress['completed_jobs'] != ids[:len(progress['completed_jobs'])]:
        raise ValueError('Completed job order differs')
    if progress['status'] in ('FAILED', 'RUNNING', 'STAGING'):
        raise RuntimeError('Previous failed/interrupted/running attempt needs explicit review; no automatic retry')
    start_token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    check_environment(args, baseline, campaign / 'environment_checks' / f'train_entry_{start_token}.json')
    save(campaign / 'training_authorisation_checks' / f'TRAIN_{start_token}.json',
         {'status': 'PASS', 'new_reproduction': True, 'checked_utc': utc(),
          'receipt': str(approval_path), 'receipt_sha256': digest(approval_path),
          'campaign_manifest_sha256': sha(campaign / 'campaign_manifest.json'),
          'package_manifest_sha256': package_hash,
          'historical_training_receipt_reused': False}, exclusive=True)
    for row in rows[:len(progress['completed_jobs'])]:
        validate_finished(campaign / progress['blocks'][row['block_id']]['path'] / row['configuration'])
    if len(progress['completed_jobs']) == 30:
        seal(args, package_hash, progress)
        return
    executed = 0
    current_job = None
    try:
        for block in ordered_blocks():
            block_rows = [r for r in rows if r['block_id'] == block]
            if block in progress['completed_blocks']:
                continue
            # A clean boundary resume gets a new audit receipt, not a replacement.
            audit_token = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            check_environment(args, baseline, campaign / 'environment_checks' / f'{block}_before_{audit_token}.json')
            if block not in progress['blocks']:
                if shutil.disk_usage(campaign).free < 8 * 1024**3:
                    raise RuntimeError('At least 8 GiB free required before a matched block')
                verify_campaign(campaign)
                out = campaign / 'attempts' / block / 'attempt_1'
                out.mkdir(parents=True, exist_ok=False)
                entry = {'block_id': block, 'attempt': 1, 'status': 'STAGING',
                         'path': out.relative_to(campaign).as_posix(), 'completed_jobs': []}
                progress['blocks'][block] = entry
                progress['status'] = 'STAGING'
                save(path, progress)
                for row in block_rows:
                    current_job = stage_attempt_job(campaign / 'jobs' / row['planned_run_id'], out / row['configuration'])
                    command([sys.executable, '-I', '-B', ROOT / 'scripts/job_worker.py', current_job, 'Probe'],
                            out / 'reports' / (row['configuration'] + '_probe.log'), 600, args.run_root)
                entry['status'] = 'READY'
                progress['status'] = 'READY'
                save(path, progress)
                print('ME06_ALL_THREE_BLOCK_PREFLIGHTS_PASS_BEFORE_ANY_TRAINING ' + block, flush=True)
            entry = progress['blocks'][block]
            if entry['status'] not in ('READY', 'PAUSED_AT_JOB_BOUNDARY'):
                raise RuntimeError('Block is not at a clean resumable boundary')
            out = campaign / entry['path']
            for row in block_rows:
                if row['planned_run_id'] in progress['completed_jobs']:
                    continue
                current_job = out / row['configuration']
                verify_campaign(campaign)
                entry.update(status='RUNNING', active_job=row['planned_run_id'], started_utc=utc())
                progress.update(status='RUNNING', main_training_started=True)
                save(path, progress)
                print('ME06_MAIN_RUN_START ' + row['planned_run_id'], flush=True)
                command([sys.executable, '-I', '-B', ROOT / 'scripts/job_worker.py', current_job, 'Train'],
                        out / 'reports' / (row['configuration'] + '_train.log'), 22200, args.run_root)
                s = validate_finished(current_job)
                report_dir = out / 'reports' / row['configuration']
                shutil.copytree(current_job / 'reports', report_dir)
                shutil.copyfile(current_job / 'calibration.json', report_dir / 'calibration.json')
                entry['completed_jobs'].append(row['planned_run_id'])
                entry['status'] = 'PAUSED_AT_JOB_BOUNDARY'
                entry.pop('active_job', None)
                progress['completed_jobs'].append(row['planned_run_id'])
                progress['status'] = 'PAUSED_AT_JOB_BOUNDARY'
                if len(entry['completed_jobs']) == 3:
                    completed = [validate_finished(out / r['configuration']) for r in block_rows]
                    if len({c['initial_state_digest'] for c in completed}) != 1:
                        raise ValueError('A/B/C block initial state mismatch')
                    check_environment(args, baseline, campaign / 'environment_checks' / f'{block}_after_{audit_token}.json')
                    entry.update(status='COMPLETE', finished_utc=utc())
                    progress['completed_blocks'].append(block)
                save(path, progress)
                print('ME06_MAIN_RUN_COMPLETE ' + row['planned_run_id'] + f' ({len(progress["completed_jobs"])}/30)', flush=True)
                executed += 1
                if args.max_jobs and executed >= args.max_jobs and len(progress['completed_jobs']) < 30:
                    check_environment(args, baseline, campaign / 'environment_checks' / f'{block}_intentional_pause_{audit_token}.json')
                    print('ME06_INTENTIONAL_PAUSE_AT_COMPLETED_JOB_BOUNDARY', flush=True)
                    return
        seal(args, package_hash, progress)
    except BaseException as e:
        progress.update(status='FAILED', error_type=type(e).__name__, message=str(e),
                        failed_utc=utc(), automatic_retry=False)
        if current_job is not None:
            progress['failure_job'] = str(current_job)
            worker_failure = current_job / 'reports/worker_failure.json'
            if worker_failure.is_file():
                progress['worker_failure'] = read(worker_failure)
        save(path, progress)
        (campaign / 'me06_training_failure.txt').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'verify'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--frozen-reference', type=Path)
    parser.add_argument('--preprocessing-receipt', type=Path)
    parser.add_argument('--authorisation', type=Path)
    parser.add_argument('--max-jobs', type=int, default=0,
                        help='Intentional stop after this many newly completed jobs; 0 means all remaining')
    args = parser.parse_args()
    if args.max_jobs < 0:
        parser.error('--max-jobs must be nonnegative')
    if args.action == 'prepare' and (args.prepared is None or args.frozen_reference is None):
        parser.error('prepare requires --prepared and --frozen-reference')
    package_hash = setup(args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / '.me06_training_wrapper.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare':
            prepare(args, package_hash)
        elif args.action == 'train':
            train(args, package_hash)
        else:
            from main_study.common import verify_campaign
            verify_campaign(args.campaign)
            progress = load(args.campaign / 'me06_train_progress.json')
            from main_study.design import plan
            for row in plan()[:len(progress['completed_jobs'])]:
                validate_finished(args.campaign / progress['blocks'][row['block_id']]['path'] / row['configuration'])
            print(json.dumps({'status': 'PASS', 'new_reproduction': True,
                              'completed_jobs_verified': len(progress['completed_jobs']),
                              'progress_status': progress['status'], 'package_manifest_sha256': package_hash}))


if __name__ == '__main__':
    main()
