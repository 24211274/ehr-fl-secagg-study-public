#!/usr/bin/env python3
"""Portable orchestration of the preserved final-test calculation, not new science.

This adapter intentionally does not call the historical environment/origins hooks.
It requires a fresh 30-model completion gate and a new campaign-bound receipt;
it then imports the unmodified scientific implementation from run_root/source.
An existing final_evaluation directory always stops execution, even after failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def environment():
    """Stable identity fields only; timestamps and live load are separate records."""
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
    return {'python': sys.version, 'prefix': sys.prefix, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'packages': [{'name': n, 'version': v} for n, v in packages],
            'thread_environment': {k: os.environ.get(k) for k in
                                   ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'NUMEXPR_NUM_THREADS', 'PYTHONHASHSEED')},
            'user_site_enabled': bool(__import__('site').ENABLE_USER_SITE)}


def require_fresh_gate(path, expected):
    value = json.loads(path.read_text())
    if (value.get('status') != expected or value.get('new_reproduction') is not True
            or value.get('historical_receipts_reused') is not False):
        raise ValueError('Fresh reproduction gate missing or invalid: ' + str(path))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--expected-benchmark-sha256', required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    source = (root / 'source').resolve(strict=True)
    campaign = (root / 'campaign').resolve(strict=True)
    if sys.prefix == sys.base_prefix or Path(sys.prefix).absolute() != root / '.venv':
        raise ValueError('Use only this new run root .venv, not the historical environment')
    if sys.version_info[:2] != (3, 12) or sys.platform != 'linux' or os.geteuid() == 0:
        raise ValueError('Prespecified non-root Linux Python 3.12 required')
    if not campaign.is_relative_to(root) or not source.is_relative_to(root):
        raise ValueError('Source/campaign escaped reproduction root')
    sys.path[:0] = [str(source), str(source / 'app'), str(source / 'validation_support')]
    from main_study.common import (read, write, sha, csv_write, safe_child, package_integrity,
                                   verify_campaign, require_authorisation, file_record, TESTS, CLIENTS)
    from main_study.final_evaluator import validate_models, score
    from main_study.test_inputs import load_final_inputs
    from main_study.evaluation import (join_scores_labels, score_endpoints, complete_analysis,
                                       calibrate_validation)
    from ehr_preprocessing.io_utils import BenchmarkSource
    from ehr_fedavg.task import setup_torch, model_factory
    from pilot_core.model import state_digest
    import torch

    package_hash = package_integrity()
    verify_campaign(campaign)
    preparation = require_fresh_gate(campaign / 'prepare_reports/me06_preparation_status.json',
                                    'ME06_FRESH_PREPARATION_COMPLETE_NO_FINAL_TESTS')
    completion = require_fresh_gate(campaign / 'main_completion_reports/me06_completion_status.json',
                                   'ME06_ALL_30_FRESH_MODELS_CALIBRATED_FINAL_TESTS_CLOSED')
    for gate in (preparation, completion):
        if gate.get('package_manifest_sha256') != package_hash:
            raise ValueError('Fresh gate does not bind the current sealed source')
    models = validate_models(campaign)
    if (preparation.get('campaign_manifest_sha256') != sha(campaign / 'campaign_manifest.json')
            or completion.get('campaign_manifest_sha256') != sha(campaign / 'campaign_manifest.json')
            or completion.get('selected_models_manifest_sha256') != sha(campaign / 'selected_models_manifest.json')):
        raise ValueError('Fresh gates do not bind this campaign and its locked selected models')
    if any(r.get('main_training_performed') is not True for r in models):
        raise ValueError('Historical/reused selected checkpoints cannot stand for fresh training')
    receipt = require_authorisation(args.receipt, 'EVALUATE_FINAL_TESTS', campaign,
                                    expected_models_sha=sha(campaign / 'selected_models_manifest.json'))
    benchmark = args.benchmark or root / 'assets/benchmark.zip'
    if sha(benchmark) != args.expected_benchmark_sha256:
        raise ValueError('Raw benchmark archive differs from frozen execution plan')
    setup_torch()
    before = environment()
    selected_hash = sha(campaign / 'selected_models_manifest.json')
    campaign_hash = sha(campaign / 'campaign_manifest.json')
    target = campaign / 'final_evaluation'
    target.mkdir(exist_ok=False)  # Never rescore, resume or overwrite a viewed final test.
    reports = target / 'reports'
    reports.mkdir()
    status = {'status': 'ME06_FINAL_EVALUATION_STARTED', 'started_utc': utc(),
              'new_reproduction': True, 'historical_receipts_reused': False,
              'test_data_opened': False, 'test_metrics_computed': False,
              'no_test_threshold_search': True, 'score_direction_unchanged': True,
              'global_score': 'validation_client_calibrated', 'per_client_score': 'raw',
              'package_manifest_sha256': package_hash,
              'campaign_manifest_sha256': campaign_hash,
              'selected_models_manifest_sha256': selected_hash,
              'benchmark_sha256': args.expected_benchmark_sha256,
              'receipt_sha256': sha(args.receipt), 'adapter_sha256': digest(__file__),
              'scientific_functions': ['main_study.test_inputs.load_final_inputs',
                  'main_study.final_evaluator.validate_models', 'main_study.final_evaluator.score',
                  'main_study.evaluation.join_scores_labels', 'main_study.evaluation.score_endpoints',
                  'main_study.evaluation.complete_analysis'],
              'adaptation': 'Only portable path, fresh-run gates and environment/provenance orchestration replaced; scientific functions unchanged.'}
    write(reports / 'stage_status.json', status)
    write(reports / 'environment_before.json', before, exclusive=True)
    write(reports / 'reproduction_receipt.json', receipt, exclusive=True)
    start = time.monotonic()
    rc = 1
    try:
        # Validation provenance is checked before first final-test access; no fitting on test.
        expected_members = __import__('csv')
        with (campaign / 'frozen/development/session_manifest.csv').open(newline='') as f:
            members = list(expected_members.DictReader(f))
        expected_validation = {r['session_id'] for r in members if r['partition'] == 'validation'}
        validation_identity = {r['session_id']: (r['client'], r['partition']) for r in members
                               if r['partition'] == 'validation'}
        if len(expected_validation) != 3622:
            raise ValueError('Validation session membership count changed')
        calibration_checks = []
        for row in models:
            job = safe_child(campaign, row['job'])
            path = job / 'LOCAL_ONLY/selected_validation_scores.json'
            if sha(path) != row['validation_scores_sha256']:
                raise ValueError('Selected validation score file changed')
            validation_artifact = read(path)
            values = validation_artifact.get('rows') if isinstance(validation_artifact, dict) else None
            if not isinstance(values, list):
                raise ValueError('Validation score schema differs')
            if any(validation_identity.get(v['session_id']) != (v['client'], v['partition']) for v in values):
                raise ValueError('Validation score client/partition identity changed')
            cal = read(safe_child(campaign, row['calibration']))
            regenerated = calibrate_validation(values, expected_validation)
            if any(cal.get(k) != v for k, v in regenerated.items()):
                raise ValueError('Saved calibration is not exact validation-only recipe output')
            calibration_checks.append({'configuration': row['configuration'], 'condition': row['condition'],
                'seed': row['seed'], 'validation_scores_sha256': sha(path),
                'validation_sessions': len(values), 'calibration_recipe_exact': True})
        write(reports / 'calibration_provenance_checks.json', calibration_checks, exclusive=True)
        status['test_data_opened'] = True
        write(reports / 'stage_status.json', status)
        with BenchmarkSource(benchmark, source / 'protocol/decisions/evidence/S02_source_file_manifest.csv') as raw:
            write(reports / 'source_integrity.json', {'files': raw.verify()}, exclusive=True)
            samples, labels, coverage = load_final_inputs(raw, campaign)
        write(reports / 'encoding_coverage.json', coverage, exclusive=True)
        for part in TESTS:
            if len(samples[part]) != {'test_held_out': 2364, 'test_known': 645}[part]:
                raise ValueError('Final-test membership count differs')
        cells = []
        model_digests = []
        score_timings = []
        for row in models:
            cp = safe_child(campaign, row['checkpoint'])
            model = model_factory()
            model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True), strict=True)
            cal = read(safe_child(campaign, row['calibration']))
            model_digests.append({'configuration': row['configuration'], 'condition': row['condition'],
                                  'seed': row['seed'], 'state_digest': state_digest(model),
                                  'checkpoint_file_sha256': sha(cp), 'selected_round': row['selected_round']})
            for part in TESTS:
                tick = time.perf_counter_ns()
                scores = score(model, samples[part])
                elapsed = time.perf_counter_ns() - tick
                expected_ids = {s['session_id'] for s in samples[part]}
                joined = join_scores_labels(scores, labels[part], part, expected_ids)
                cell = {'configuration': row['configuration'], 'condition': row['condition'],
                        'seed': row['seed'], 'test': part,
                        'checkpoint_sha256': row['selected_checkpoint_sha256'],
                        'calibration_sha256': row['calibration_sha256'],
                        'metrics': score_endpoints(joined, cal)}
                cells.append(cell)
                name = f"{row['configuration']}_{row['condition']}_{row['seed']}_{part}"
                csv_write(target / 'LOCAL_ONLY' / f'{name}_scores.csv', joined)
                write(reports / 'cells' / f'{name}.json', cell, exclusive=True)
                score_timings.append({'cell': name, 'sessions': len(joined), 'score_wall_ns': elapsed,
                                      'scope': 'batched model loss scoring only, encoding and metrics excluded'})
            print(f"ME06_FINAL_EVALUATION_MODEL_SCORED {row['configuration']}/{row['condition']}/{row['seed']}", flush=True)
        if len(cells) != 60:
            raise ValueError('Exactly all 60 final metric cells required')
        analysis = complete_analysis(cells)
        if analysis['status'] != 'COMPLETE':
            raise ValueError('Incomplete paired-seed analysis')
        write(reports / 'paired_seed_analysis.json', analysis, exclusive=True)
        write(reports / 'selected_model_tensor_digests.json', model_digests, exclusive=True)
        write(reports / 'new_final_scoring_times.json', score_timings, exclusive=True)
        status.update(status='ME06_FINAL_EVALUATION_COMPLETE_LOCKED_MODELS_NO_RETUNING',
                      test_metrics_computed=True, metric_cells=60,
                      saved_score_rows=sum(c['metrics']['quantiles']['0.99']['global']['sessions'] for c in cells))
        rc = 0
    except BaseException as exc:
        status.update(status='ME06_FINAL_EVALUATION_STOP_REVIEW_REQUIRED',
                      error_type=type(exc).__name__, message=str(exc))
        (reports / 'failure_trace.txt').write_text(traceback.format_exc())
        print('STOP [ME06 final evaluator]: ' + str(exc), flush=True)
    finally:
        try:
            after = environment()
            write(reports / 'environment_after.json', after, exclusive=True)
            if after != before:
                raise ValueError('New isolated environment identity changed during final evaluation')
            verify_campaign(campaign)
            validate_models(campaign)
            if sha(campaign / 'selected_models_manifest.json') != selected_hash or sha(campaign / 'campaign_manifest.json') != campaign_hash:
                raise ValueError('Campaign/model selection changed during final evaluation')
            if sha(benchmark) != args.expected_benchmark_sha256:
                raise ValueError('Raw benchmark changed during final evaluation')
        except BaseException as exc:
            rc = 1
            status['postcheck_error'] = str(exc)
            status['status'] = 'ME06_FINAL_EVALUATION_STOP_REVIEW_REQUIRED'
        status.update(completed_utc=utc(), elapsed_wall_seconds=time.monotonic() - start,
                      exit_code=rc, environment_identity_unchanged=('postcheck_error' not in status))
        write(reports / 'stage_status.json', status)
        members = [file_record(p, target) for p in sorted(target.rglob('*')) if p.is_file()]
        write(target / 'me06_evaluation_files.json', {'status': status['status'], 'files': members}, exclusive=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
