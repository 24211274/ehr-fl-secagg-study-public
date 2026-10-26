#!/usr/bin/env python3
"""Full-precision, per-session comparison of a locked fresh 30-model reproduction.

All thresholds/metrics retain the original direction and validation calibration.
Exact discrepancies are counted. The 1e-12 column is an arithmetic diagnostic,
never a generic tolerance for training, inference or scientific equivalence.
No data/model selection, calibration fitting, optimisation or test tuning occurs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import traceback
import zipfile


EXPECTED_SCORES_SHA256 = '6bd4d752c12730d6d93143c2e8bd184622b5e5c2778d244a1327ddc1657d6dcf'
EXPECTED_EVALUATION_SHA256 = 'fe66ff79196f2936605ce873689821a0716fa7bf7c7c99323ad5af26e8497ecd'
EXPECTED_SELECTED_SHA256 = '26519dd2e95f55667eac71ae5a5fc84529a25a14e39a5cd9a1b26648bd13510c'
SEEDS = (11, 29, 47, 71, 97)
QUANTILES = ('0.99', '0.98', '0.995')


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


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def checked_zip(path, expected_hash, manifest_name):
    if sha(path) != expected_hash:
        raise ValueError('Reference archive identity changed: ' + str(path))
    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive members')
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:
            raise ValueError('Unsafe reference member')
    manifest = json.loads(archive.read(manifest_name))
    rows = manifest['files']
    if not rows or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Invalid reference file manifest')
    for row in rows:
        raw = archive.read(row['path'])
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Reference manifest drift: ' + row['path'])
    return archive, manifest


def compare_tree(reference, fresh, prefix=''):
    """Yield every unequal leaf/schema, count all visited leaves in a side counter."""
    stats = {'leaves_compared': 0, 'numeric_leaves_compared': 0, 'differences': 0,
             'integer_or_boolean_differences': 0, 'float_differences': 0,
             'float_differences_exceeding_1e12_diagnostic': 0, 'schema_or_text_differences': 0}
    differences = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    different(path + '/' + str(key), a.get(key), b.get(key), 'schema')
                else:
                    walk(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                different(path + '/length', len(a), len(b), 'schema')
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + '/' + str(index))
        else:
            stats['leaves_compared'] += 1
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                stats['numeric_leaves_compared'] += 1
            # The original metric schema is fixed; bool/int confusion is not accepted.
            if type(a) is not type(b) or a != b:
                if type(a) is float and type(b) is float:
                    kind = 'float'
                elif type(a) in (int, bool) and type(a) is type(b):
                    kind = 'integer_or_boolean'
                else:
                    kind = 'schema_or_text'
                different(path, a, b, kind)

    def different(path, a, b, kind):
        stats['differences'] += 1
        row = {'path': path, 'reference': a, 'fresh': b, 'kind': kind,
               'fresh_minus_reference': None, 'absolute_difference': None,
               'within_1e12_arithmetic_diagnostic': None}
        if kind == 'float' and math.isfinite(a) and math.isfinite(b):
            d = b - a
            row.update(fresh_minus_reference=d, absolute_difference=abs(d),
                       within_1e12_arithmetic_diagnostic=abs(d) <= 1e-12)
            stats['float_differences'] += 1
            stats['float_differences_exceeding_1e12_diagnostic'] += abs(d) > 1e-12
        elif kind == 'integer_or_boolean':
            stats['integer_or_boolean_differences'] += 1
        else:
            stats['schema_or_text_differences'] += 1
        differences.append(row)

    walk(reference, fresh, prefix)
    return stats, differences


def score_rows(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if not rows or len({r['session_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate saved score IDs')
    for row in rows:
        if row['label'] not in ('0', '1'):
            raise ValueError('Invalid binary label')
        row['label'] = int(row['label'])
        row['loss'] = float(row['loss'])
        if not math.isfinite(row['loss']) or row['loss'] < 0:
            raise ValueError('Invalid score')
    return rows


def ordered_score_pair(reference, fresh, identity_fields):
    old = {r['session_id']: r for r in reference}
    new = {r['session_id']: r for r in fresh}
    if len(old) != len(reference) or len(new) != len(fresh) or set(old) != set(new):
        raise ValueError('Reference/fresh score IDs differ; comparison cannot inner join')
    keys = sorted(old)
    for sid in keys:
        for name in identity_fields:
            if old[sid].get(name) != new[sid].get(name):
                raise ValueError(f'Identity/label mismatch for {sid}: {name}')
    return keys, [old[s] for s in keys], [new[s] for s in keys]


def numerical_errors(reference, fresh):
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(fresh, dtype=np.float64)
    d = b - a
    nonzero = np.abs(a) > 0
    relative = np.abs(d[nonzero] / a[nonzero])
    return {'n': len(a), 'exactly_equal': int((a == b).sum()),
            'unequal': int((a != b).sum()), 'max_absolute_error': float(np.abs(d).max()),
            'median_absolute_error': float(np.median(np.abs(d))),
            'mean_absolute_error': float(np.mean(np.abs(d))),
            'root_mean_squared_error': float(np.sqrt(np.mean(d * d))),
            'max_relative_error_excluding_zero_reference': float(relative.max()) if len(relative) else None,
            'reference_zero_count': int((~nonzero).sum())}


def ranking_errors(reference, fresh):
    import numpy as np
    from scipy.stats import rankdata, kendalltau
    a, b = np.asarray(reference), np.asarray(fresh)
    ra, rb = rankdata(a, method='average'), rankdata(b, method='average')
    d = rb - ra
    # Exact pair ordering/tie transitions, counted once for every pair of session IDs.
    pair_order_changes = strict_reversals = old_tie_broken = new_tie_created = 0
    for i in range(len(a) - 1):
        oa = np.sign(a[i] - a[i + 1:])
        ob = np.sign(b[i] - b[i + 1:])
        pair_order_changes += int(np.count_nonzero(oa != ob))
        strict_reversals += int(np.count_nonzero(oa * ob < 0))
        old_tie_broken += int(np.count_nonzero((oa == 0) & (ob != 0)))
        new_tie_created += int(np.count_nonzero((oa != 0) & (ob == 0)))
    tau = float(kendalltau(a, b, variant='b').statistic)
    summary = {'average_rank_changed_sessions': int((ra != rb).sum()),
               'max_absolute_average_rank_change': float(np.abs(d).max()),
               'pair_count': len(a) * (len(a) - 1) // 2,
               'pair_order_or_tie_changes': pair_order_changes,
               'strict_pair_reversals': strict_reversals,
               'reference_ties_broken': old_tie_broken, 'fresh_ties_created': new_tie_created,
               'kendall_tau_b_descriptive': tau if math.isfinite(tau) else None,
               'significance_test_performed': False}
    return summary, ra, rb


def compare_sessions(old_rows, new_rows, old_cal, new_cal, output):
    import numpy as np
    ids, old, new = ordered_score_pair(old_rows, new_rows,
        ('client', 'partition', 'label', 'scenario', 'primary_class'))
    old_raw = np.array([r['loss'] for r in old])
    new_raw = np.array([r['loss'] for r in new])
    def calibrated(rows, cal):
        return np.array([(r['loss'] - cal['clients'][r['client']]['median']) /
                         cal['clients'][r['client']]['scale'] for r in rows])
    old_z, new_z = calibrated(old, old_cal), calibrated(new, new_cal)
    raw_ranks, orr, nrr = ranking_errors(old_raw, new_raw)
    z_ranks, ozr, nzr = ranking_errors(old_z, new_z)
    per_id = [{'session_id': sid, 'client': a['client'], 'partition': a['partition'],
        'label': a['label'], 'scenario': a['scenario'], 'primary_class': a['primary_class'],
        'reference_raw': float(old_raw[i]), 'fresh_raw': float(new_raw[i]),
        'raw_fresh_minus_reference': float(new_raw[i] - old_raw[i]),
        'reference_calibrated': float(old_z[i]), 'fresh_calibrated': float(new_z[i]),
        'calibrated_fresh_minus_reference': float(new_z[i] - old_z[i]),
        'reference_raw_average_rank': float(orr[i]), 'fresh_raw_average_rank': float(nrr[i]),
        'raw_average_rank_change': float(nrr[i] - orr[i]),
        'reference_calibrated_average_rank': float(ozr[i]), 'fresh_calibrated_average_rank': float(nzr[i]),
        'calibrated_average_rank_change': float(nzr[i] - ozr[i])} for i, (sid, a) in enumerate(zip(ids, old))]
    flips = {}
    for q in QUANTILES:
        old_thresholds = np.array([old_cal['clients'][r['client']]['thresholds'][q] for r in old])
        new_thresholds = np.array([new_cal['clients'][r['client']]['thresholds'][q] for r in new])
        op, npred = old_raw > old_thresholds, new_raw > new_thresholds
        fixed_old_cal = new_raw > old_thresholds
        old_scores_new_cal = old_raw > new_thresholds
        y = np.array([r['label'] for r in old])
        changed = op != npred
        flips[q] = {'threshold_operator': '>', 'changed_decisions': int(changed.sum()),
                    'reference_alert_to_fresh_no_alert': int((op & ~npred).sum()),
                    'reference_no_alert_to_fresh_alert': int((~op & npred).sum()),
                    'positive_label_decision_flips': int((changed & (y == 1)).sum()),
                    'negative_label_decision_flips': int((changed & (y == 0)).sum()),
                    'fresh_raw_fixed_reference_threshold_flips': int((op != fixed_old_cal).sum()),
                    'reference_raw_fresh_threshold_flips': int((op != old_scores_new_cal).sum()),
                    'component_diagnostics_are_not_additive_causal_effects': True}
        for i, row in enumerate(per_id):
            row.update({f'reference_alert_q{q}': int(op[i]), f'fresh_alert_q{q}': int(npred[i]),
                        f'alert_flip_q{q}': int(changed[i]),
                        f'fresh_raw_reference_threshold_alert_q{q}': int(fixed_old_cal[i]),
                        f'reference_raw_fresh_threshold_alert_q{q}': int(old_scores_new_cal[i])})
    write_csv(output, per_id)
    return {'sessions': len(ids), 'identity_label_join_exact': True,
            'raw_score_errors': numerical_errors(old_raw, new_raw),
            'calibrated_score_errors': numerical_errors(old_z, new_z),
            'raw_ranking': raw_ranks, 'calibrated_ranking': z_ranks,
            'threshold_flips': flips,
            'per_client': {c: {'raw': numerical_errors(old_raw[[r['client'] == c for r in old]],
                                                     new_raw[[r['client'] == c for r in old]]),
                              'calibrated': numerical_errors(old_z[[r['client'] == c for r in old]],
                                                            new_z[[r['client'] == c for r in old]])}
                           for c in ('C1', 'C2', 'C3')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--reference-scores', type=Path)
    parser.add_argument('--reference-evaluation', type=Path)
    parser.add_argument('--reference-review', type=Path)
    parser.add_argument('--reference-selected', type=Path)
    parser.add_argument('--reference-assets', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    source, campaign = root / 'source', root / 'campaign'
    sys.path[:0] = [str(source), str(source / 'app'), str(source / 'validation_support')]
    from main_study.common import verify_campaign, safe_child, verify_rows
    from main_study.final_evaluator import validate_models
    from main_study.evaluation import score_endpoints, complete_analysis
    from main_study.costs import campaign_summary
    from ehr_fedavg.task import model_factory, setup_torch
    from pilot_core.model import state_digest
    import torch
    import numpy as np
    setup_torch()
    verify_campaign(campaign)
    models = validate_models(campaign)
    model_map = {(r['configuration'], r['condition'], r['seed']): r for r in models}
    final = campaign / 'final_evaluation'
    evaluation_manifest = read(final / 'me06_evaluation_files.json')
    verify_rows(final, evaluation_manifest['files'])
    stage = read(final / 'reports/stage_status.json')
    if stage.get('status') != 'ME06_FINAL_EVALUATION_COMPLETE_LOCKED_MODELS_NO_RETUNING' or stage.get('metric_cells') != 60:
        raise ValueError('Fresh final evaluation incomplete')
    if stage['selected_models_manifest_sha256'] != sha(campaign / 'selected_models_manifest.json'):
        raise ValueError('Fresh model manifest changed after scoring')
    assets = root / 'assets'
    scores_path = args.reference_scores or assets / 'saved_score_reference.zip'
    evaluation_path = args.reference_evaluation or assets / 'final_evaluation_reference.zip'
    review_path = args.reference_review or assets / 'reference_targets.json'
    selected_path = args.reference_selected or assets / 'selected_models_reference.json'
    if sha(selected_path) != EXPECTED_SELECTED_SHA256:
        raise ValueError('Reference selected-model registry changed')
    old_selected = read(selected_path)
    old_models = {(r['configuration'], r['condition'], r['seed']): r for r in old_selected['models']}
    if set(model_map) != set(old_models) or len(model_map) != 30:
        raise ValueError('Reference/fresh model grid mismatch')
    old_scores, scores_manifest = checked_zip(scores_path, EXPECTED_SCORES_SHA256, 'audit_export_manifest.json')
    old_evaluation, _ = checked_zip(evaluation_path, EXPECTED_EVALUATION_SHA256, 'report_files.json')
    review = read(review_path)
    if review.get('campaign') != '20260909T142009Z_2ad79bf6' or review['grid']['evaluation_cells'] != 60:
        raise ValueError('Reference review identifies a different campaign')
    out = args.output or root / 'comparison'
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'ME06_COMPARISON_STARTED', 'started_utc': utc(),
              'comparison_does_not_change_models_scores_calibration_or_thresholds': True,
              'selected_manifest_sha256': sha(campaign / 'selected_models_manifest.json'),
              'fresh_evaluation_manifest_sha256': sha(final / 'me06_evaluation_files.json'),
              'reference_scores_sha256': sha(scores_path), 'reference_evaluation_sha256': sha(evaluation_path),
              'reference_review_sha256': sha(review_path), 'adapter_sha256': sha(__file__),
              'numeric_policy': 'Report exact equality and all differences. abs(delta)<=1e-12 is only an arithmetic diagnostic, not a model reproduction/equivalence acceptance criterion.',
              'cause_of_any_difference': 'UNATTRIBUTED unless separately established by controlled diagnostic evidence',
              'timing_policy': 'New timing is descriptive; historical/new hardware and load differ. No exact timing equality or generic percent tolerance.'}
    write(out / 'started.json', status)
    all_diff, metric_summary, per_session, cal_summary, model_summary = [], [], [], [], []
    fresh_cells, primary_rows = [], []
    try:
        for key in sorted(model_map):
            m, co, seed = key
            row, oldrow = model_map[key], old_models[key]
            name = f'{m}_{co}_{seed}'
            new_cal = read(safe_child(campaign, row['calibration']))
            old_cal_raw = old_scores.read(f'calibrations/{name}.json')
            if hashlib.sha256(old_cal_raw).hexdigest() != oldrow['calibration_sha256']:
                raise ValueError('Reference calibration/registry mismatch')
            old_cal = json.loads(old_cal_raw)
            # File provenance hashes can differ while numerical calibrations remain equal.
            cal_keys = set(old_cal) - {'selected_checkpoint_sha256', 'validation_scores_sha256'}
            if cal_keys != set(new_cal) - {'selected_checkpoint_sha256', 'validation_scores_sha256'}:
                raise ValueError('Calibration schema differs')
            calstats, caldiff = compare_tree({k: old_cal[k] for k in cal_keys},
                                             {k: new_cal[k] for k in cal_keys}, name + '/calibration')
            cal_summary.append({'model': name, **calstats})
            all_diff.extend(caldiff)
            summary = {'model': name, 'reference_selected_round': oldrow['selected_round'],
                'fresh_selected_round': row['selected_round'],
                'selected_round_equal': oldrow['selected_round'] == row['selected_round'],
                'initial_tensor_digest_equal': oldrow['initial_state_digest'] == row['initial_state_digest'],
                'checkpoint_serialization_file_sha256_equal': oldrow['selected_checkpoint_sha256'] == row['selected_checkpoint_sha256'],
                'calibration_file_sha256_equal': oldrow['calibration_sha256'] == row['calibration_sha256'],
                'validation_score_file_sha256_equal': oldrow['validation_scores_sha256'] == row['validation_scores_sha256'],
                'selected_tensor_comparison': 'NOT_AVAILABLE_WITHOUT_REFERENCE_ASSETS'}
            if args.reference_assets is not None:
                old_cp = safe_child(args.reference_assets, oldrow['checkpoint'])
                if sha(old_cp) != oldrow['selected_checkpoint_sha256']:
                    raise ValueError('Reference checkpoint identity differs')
                fresh_cp = safe_child(campaign, row['checkpoint'])
                old_state = torch.load(old_cp, map_location='cpu', weights_only=True)
                new_state = torch.load(fresh_cp, map_location='cpu', weights_only=True)
                if set(old_state) != set(new_state):
                    raise ValueError('Selected tensor architecture differs')
                tensor_errors = []
                for parameter in sorted(old_state):
                    a, b = old_state[parameter], new_state[parameter]
                    if a.shape != b.shape or a.dtype != b.dtype:
                        raise ValueError('Selected tensor shape/dtype differs: ' + parameter)
                    tensor_errors.append({'tensor': parameter, 'shape': list(a.shape), 'dtype': str(a.dtype),
                        **numerical_errors(a.detach().numpy().reshape(-1).astype(np.float64),
                                           b.detach().numpy().reshape(-1).astype(np.float64))})
                old_model, new_model = model_factory(), model_factory()
                old_model.load_state_dict(old_state, strict=True)
                new_model.load_state_dict(new_state, strict=True)
                summary.update(reference_tensor_state_digest=state_digest(old_model),
                               fresh_tensor_state_digest=state_digest(new_model),
                               selected_tensor_comparison='EXACT' if state_digest(old_model) == state_digest(new_model) else 'DIFFERENT',
                               tensor_errors=tensor_errors)
                old_val_path = safe_child(args.reference_assets, oldrow['job']) / 'LOCAL_ONLY/selected_validation_scores.json'
                new_val_path = safe_child(campaign, row['job']) / 'LOCAL_ONLY/selected_validation_scores.json'
                if sha(old_val_path) != oldrow['validation_scores_sha256'] or sha(new_val_path) != row['validation_scores_sha256']:
                    raise ValueError('Validation score provenance differs')
                ids, a, b = ordered_score_pair(read(old_val_path)['rows'], read(new_val_path)['rows'],
                                               ('client', 'partition', 'events'))
                va, vb = np.array([r['loss'] for r in a]), np.array([r['loss'] for r in b])
                vrank, ar, br = ranking_errors(va, vb)
                summary['validation_raw_errors'] = numerical_errors(va, vb)
                summary['validation_raw_ranking'] = vrank
                write_csv(out / 'validation_per_session' / f'{name}.csv',
                    [{'session_id': sid, 'client': x['client'], 'events': x['events'],
                      'reference_raw': float(va[i]), 'fresh_raw': float(vb[i]),
                      'fresh_minus_reference': float(vb[i] - va[i]),
                      'reference_average_rank': float(ar[i]), 'fresh_average_rank': float(br[i]),
                      'average_rank_change': float(br[i] - ar[i])} for i, (sid, x) in enumerate(zip(ids, a))])
            model_summary.append(summary)
            for test in ('test_held_out', 'test_known'):
                cellname = name + '_' + test
                old_cell = json.loads(old_evaluation.read(f'cells/{cellname}.json'))
                fresh_cell = read(final / 'reports/cells' / f'{cellname}.json')
                identity = {'configuration': m, 'condition': co, 'seed': seed, 'test': test}
                if any(old_cell[k] != v or fresh_cell[k] != v for k, v in identity.items()):
                    raise ValueError('Cell filename/identity mismatch')
                old_rows = score_rows(old_scores.read(f'scores/{cellname}_scores.csv'))
                new_rows = score_rows((final / 'LOCAL_ONLY' / f'{cellname}_scores.csv').read_bytes())
                # Recompute arithmetic from both locked score sets; do not rescore/recalibrate.
                for tag, cellscores, cal, original in [('reference', old_rows, old_cal, old_cell),
                                                      ('fresh', new_rows, new_cal, fresh_cell)]:
                    proofstats, proofdiff = compare_tree(original['metrics'], score_endpoints(cellscores, cal))
                    if (proofstats['integer_or_boolean_differences'] or proofstats['schema_or_text_differences']
                            or proofstats['float_differences_exceeding_1e12_diagnostic']):
                        write(out / f'INVALID_{tag}_arithmetic_{cellname}.json', proofdiff)
                        raise ValueError(tag + ' metrics fail saved-score arithmetic check: ' + cellname)
                stats, diffs = compare_tree(old_cell['metrics'], fresh_cell['metrics'], cellname + '/metrics')
                metric_summary.append({'cell': cellname, **identity, **stats})
                all_diff.extend(diffs)
                sessionstats = compare_sessions(old_rows, new_rows, old_cal, new_cal,
                                                 out / 'test_per_session' / f'{cellname}.csv')
                per_session.append({'cell': cellname, **identity, **sessionstats})
                fresh_cells.append(fresh_cell)
                old_primary = old_cell['metrics']['quantiles']['0.99']['global']
                new_primary = fresh_cell['metrics']['quantiles']['0.99']['global']
                primary = {**identity, 'reference_selected_round': oldrow['selected_round'],
                           'fresh_selected_round': row['selected_round']}
                for endpoint in ('AP', 'AUROC', 'precision', 'recall', 'F1', 'FPR', 'FNR',
                                 'false_alerts_per_1000_legitimate', 'TP', 'TN', 'FP', 'FN'):
                    x, y = old_primary[endpoint], new_primary[endpoint]
                    primary.update({f'reference_{endpoint}': x, f'fresh_{endpoint}': y,
                                    f'delta_{endpoint}': y - x if x is not None and y is not None else None})
                primary['q0.99_decision_flips'] = sessionstats['threshold_flips']['0.99']['changed_decisions']
                primary_rows.append(primary)
                print('ME06_COMPARISON_CELL ' + cellname, flush=True)
        if len(fresh_cells) != 60 or sum(r['sessions'] for r in per_session) != 90270:
            raise ValueError('Incomplete comparison matrix')
        old_paired = json.loads(old_evaluation.read('paired_seed_analysis.json'))
        new_paired = read(final / 'reports/paired_seed_analysis.json')
        if new_paired != complete_analysis(fresh_cells):
            raise ValueError('Fresh paired summary fails recomputation')
        if old_paired != review['reference_paired_seed_analysis']:
            raise ValueError('Reference review/full-precision paired data disagree')
        pairedstats, paireddiff = compare_tree(old_paired, new_paired, 'paired_seed_analysis')
        all_diff.extend(paireddiff)
        covstats, covdiff = compare_tree(json.loads(old_evaluation.read('encoding_coverage.json')),
                                        read(final / 'reports/encoding_coverage.json'), 'encoding_coverage')
        if covstats['differences']:
            write(out / 'encoding_coverage_differences.json', covdiff)
            raise ValueError('Raw test encoding coverage differs despite the same frozen inputs')
        new_costs = campaign_summary(campaign, models)
        write(out / 'new_system_cost_analysis.json', new_costs)
        old_costs = review['reference_system_costs']
        old_cost_map = {(r['configuration'], r['condition'], r['seed']): r for r in old_costs['cells']}
        cost_rows = []
        for row in new_costs['cells']:
            old = old_cost_map[(row['configuration'], row['condition'], row['seed'])]
            result = {k: row[k] for k in ('configuration', 'condition', 'seed')}
            for field in row:
                if field in result:
                    continue
                a, b = old[field], row[field]
                result.update({f'reference_{field}': a, f'fresh_{field}': b,
                               f'delta_{field}': b - a if a is not None and b is not None else None})
            cost_rows.append(result)
        write_csv(out / 'new_vs_historical_system_costs_descriptive.csv', cost_rows)
        summary_rows = []
        for group, endpoints in old_paired['analysis'].items():
            for endpoint, old in endpoints.items():
                new = new_paired['analysis'][group][endpoint]
                for method in 'ABC':
                    a, b = old['by_method'][method], new['by_method'][method]
                    summary_rows.append({'group': group, 'endpoint': endpoint, 'method_or_contrast': method,
                        'reference_mean': a['mean'], 'fresh_mean': b['mean'],
                        'delta_mean': b['mean'] - a['mean'] if a['mean'] is not None and b['mean'] is not None else None,
                        'reference_sample_sd': a['sample_sd'], 'fresh_sample_sd': b['sample_sd'],
                        'reference_95_interval': None, 'fresh_95_interval': None})
                for contrast in ('B_minus_A', 'C_minus_B'):
                    a, b = old[contrast], new[contrast]
                    summary_rows.append({'group': group, 'endpoint': endpoint, 'method_or_contrast': contrast,
                        'reference_mean': a['mean'], 'fresh_mean': b['mean'],
                        'delta_mean': b['mean'] - a['mean'] if a['mean'] is not None and b['mean'] is not None else None,
                        'reference_sample_sd': a['sample_sd'], 'fresh_sample_sd': b['sample_sd'],
                        'reference_95_interval': a['interval_95'], 'fresh_95_interval': b['interval_95']})
        write_csv(out / 'all_seed_means_and_paired_contrasts.csv', summary_rows)
        write_csv(out / 'all_60_primary_metric_cells.csv', primary_rows)
        write_csv(out / 'metric_and_calibration_unequal_leaves.csv', all_diff,
                  ['path', 'reference', 'fresh', 'kind', 'fresh_minus_reference',
                   'absolute_difference', 'within_1e12_arithmetic_diagnostic'])
        write(out / 'all_metric_comparison_counts.json', metric_summary)
        write(out / 'all_calibration_comparison_counts.json', cal_summary)
        write(out / 'all_test_score_comparisons.json', per_session)
        write(out / 'all_model_and_validation_comparisons.json', model_summary)
        write(out / 'paired_analysis_comparison_counts.json', pairedstats)
        write(out / 'word_table_mapping.json', review['word_table_mapping'])
        numerical_differences = sum(r['differences'] for r in metric_summary + cal_summary) + pairedstats['differences']
        flips = sum(r['threshold_flips']['0.99']['changed_decisions'] for r in per_session)
        raw_unequal = sum(r['raw_score_errors']['unequal'] for r in per_session)
        calibrated_unequal = sum(r['calibrated_score_errors']['unequal'] for r in per_session)
        round_mismatches = sum(not r['selected_round_equal'] for r in model_summary)
        initial_mismatches = sum(not r['initial_tensor_digest_equal'] for r in model_summary)
        tensor_mismatches = sum(r['selected_tensor_comparison'] == 'DIFFERENT' for r in model_summary)
        val_unequal = sum(r.get('validation_raw_errors', {}).get('unequal', 0) for r in model_summary)
        exact_observed = not any((numerical_differences, raw_unequal, calibrated_unequal,
                                  round_mismatches, initial_mismatches, tensor_mismatches, val_unequal))
        status.update(status='ME06_COMPARISON_COMPLETE_EXACT_REPORTED_VALUES' if exact_observed else 'ME06_COMPARISON_COMPLETE_WITH_DIFFERENCES',
            models=30, metric_cells=60, score_rows_compared=90270,
            metric_unequal_leaves=sum(r['differences'] for r in metric_summary),
            calibration_unequal_leaves=sum(r['differences'] for r in cal_summary),
            paired_analysis_unequal_leaves=pairedstats['differences'],
            raw_test_score_unequal_rows=raw_unequal,
            calibrated_test_score_unequal_rows=calibrated_unequal,
            validation_raw_score_unequal_rows=val_unequal,
            primary_threshold_decision_flips=flips,
            test_identity_and_labels_exact=True, test_encoding_coverage_exact=True,
            selected_round_mismatches=round_mismatches,
            initial_tensor_state_mismatches=initial_mismatches,
            selected_tensor_state_mismatches=tensor_mismatches,
            selected_tensor_state_comparisons_available=sum(r['selected_tensor_comparison'] != 'NOT_AVAILABLE_WITHOUT_REFERENCE_ASSETS' for r in model_summary),
            equivalence_or_clinical_validity_proven=False, scope='Same-machine fresh Python environment and fixed raw snapshot; not independent external researcher/host or original data-generator reproduction.')
        verify_campaign(campaign)
        validate_models(campaign)
        verify_rows(final, evaluation_manifest['files'])
        status['completed_utc'] = utc()
        write(out / 'comparison_status.json', status)
        def display(value):
            return 'undefined' if value is None else format(value, '.17g')
        report = ['# ME-06 fresh-run comparison', '',
            '**Status:** ' + status['status'], '',
            'The comparison covers 30 freshly trained selected models, all 60 final metric cells, '
            'all three fixed validation quantiles, 90,270 test score rows, calibration values, '
            'and every prespecified paired seed endpoint. Raw labels and session identities are '
            'joined exactly; no inner join, score-direction reversal or test retuning was used.', '',
            '| Check | Finding |', '|---|---:|',
            f'| Unequal test metric leaves | {status["metric_unequal_leaves"]} |',
            f'| Unequal calibration leaves | {status["calibration_unequal_leaves"]} |',
            f'| Unequal paired-summary leaves | {status["paired_analysis_unequal_leaves"]} |',
            f'| Unequal raw test score rows | {raw_unequal} |',
            f'| Unequal calibrated test score rows | {calibrated_unequal} |',
            f'| q=.99 alert decision flips | {flips} |',
            f'| Changed selected rounds | {round_mismatches} |',
            f'| Changed initial tensor states | {initial_mismatches} |',
            f'| Selected tensor comparisons available | {status["selected_tensor_state_comparisons_available"]}/30 |',
            f'| Changed selected tensor states | {tensor_mismatches} |', '',
            'Exact equality is distinguished from the 1e-12 arithmetic diagnostic. '
            'No generic percentage tolerance, equivalence conclusion, or explanation by randomness '
            'is inferred from these diagnostics. Checkpoint serialization SHA is separate from '
            'the numerical tensor state digest. New timing is descriptive, not an equality target.', '',
            '## Global Average Precision (all five original seeds)', '',
            '| Condition/test | Method | Reference mean | Fresh mean | Difference |', '|---|---|---:|---:|---:|']
        for row in summary_rows:
            if row['endpoint'] == 'global_AP' and row['method_or_contrast'] in 'ABC':
                report.append('| ' + row['group'] + ' | ' + row['method_or_contrast'] + ' | '
                    + display(row['reference_mean']) + ' | ' + display(row['fresh_mean']) + ' | '
                    + display(row['delta_mean']) + ' |')
        report.extend(['', '## Evidence and interpretation limits', '',
            '- `all_60_primary_metric_cells.csv`: per-seed primary metrics and exact count/decision changes.',
            '- `all_seed_means_and_paired_contrasts.csv`: all endpoints, SDs and the original exploratory intervals.',
            '- `metric_and_calibration_unequal_leaves.csv`: every unequal metric/calibration/paired field.',
            '- `test_per_session/`: raw/calibrated scores, ranks and three threshold decisions for every test row.',
            '- `all_model_and_validation_comparisons.json`: selected rounds, serialization hashes, tensor errors and validation diagnostics.',
            '- `new_vs_historical_system_costs_descriptive.csv`: newly measured costs compared descriptively with the historical run.', '',
            'This tests the fixed raw snapshot in a newly created Python environment on the same machine. '
            'It does not establish a new independent clinical/security validation of labels, public redistribution rights, '
            'the unavailable original data generator, external-data generalization, or independent external-host reproducibility. '
            'Any observed numeric difference needs separately evidenced investigation; it is not automatically attributed '
            'to a random seed or environment.', ''])
        (out / 'Comparison_Report.md').write_text('\n'.join(report), encoding='utf-8')
        entries = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in sorted(out.rglob('*')) if p.is_file()]
        write(out / 'comparison_files.json', {'files': entries})
        print(status['status'], flush=True)
        return 0
    except BaseException as exc:
        status.update(status='ME06_COMPARISON_STOP_REVIEW_REQUIRED', completed_utc=utc(),
                      error_type=type(exc).__name__, message=str(exc))
        write(out / 'comparison_failure.json', status)
        (out / 'failure_trace.txt').write_text(traceback.format_exc())
        print('STOP [ME06 comparison]: ' + str(exc), flush=True)
        raise
    finally:
        old_scores.close()
        old_evaluation.close()


if __name__ == '__main__':
    raise SystemExit(main())
