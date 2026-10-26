#!/usr/bin/env python3
"""Explicit canonical metadata binding for one verified diagnostic-only ULP drift.

The fresh raw rebuild remains untouched. A new sibling trainval directory keeps
every fresh array. Only the canonical scaler JSON representation and its hash
links are bound to the fixed original runtime contract, after exact evidence.
This does NOT assert byte-identical fresh scaler reproduction.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def inventory(root):
    return [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
             'sha256': sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]


def differences(a, b, path=()):
    if type(a) is not type(b):
        return [{'path': list(path), 'reference': a, 'fresh': b, 'type_mismatch': True}]
    if isinstance(a, dict):
        if list(a) != list(b):
            return [{'path': list(path), 'reference_keys': list(a), 'fresh_keys': list(b)}]
        return [r for k in a for r in differences(a[k], b[k], path + (k,))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [{'path': list(path), 'reference_length': len(a), 'fresh_length': len(b)}]
        return [r for i in range(len(a)) for r in differences(a[i], b[i], path + (i,))]
    return [] if a == b else [{'path': list(path), 'reference': a, 'fresh': b}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--fresh', type=Path, required=True)
    p.add_argument('--bound', type=Path, required=True)
    p.add_argument('--frozen-reference', type=Path, required=True)
    args = p.parse_args()
    if sys.platform != 'linux' or os.geteuid() == 0 or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use non-root Linux Python 3.12')
    run = args.run_root.resolve()
    source, fresh, bound, reference = (getattr(args, n).resolve()
                                       for n in ('source', 'fresh', 'bound', 'frozen_reference'))
    if Path(os.path.abspath(sys.executable)) != run / '.venv/bin/python':
        raise RuntimeError('Use the new run-root venv explicitly')
    if source != run / 'source' or fresh != run / 'preprocessing/trainval' or bound != run / 'preprocessing/bound_trainval':
        raise ValueError('Require explicit new source/fresh/bound paths within this run root')
    if reference != run / 'assets/frozen_reference.zip':
        raise ValueError('Require the local reference comparison archive')
    if bound.exists() or bound.is_symlink():
        raise FileExistsError('A new binding directory is required; no overwrite/retry')
    if fresh.is_symlink() or any(x.is_symlink() for x in fresh.rglob('*')):
        raise ValueError('Fresh preprocessing cannot contain symlinks')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(source), str(source / 'app')]
    from main_study.common import package_integrity
    from ehr_preprocessing.io_utils import verify_artefacts, seal_artefacts, write_json
    import numpy as np
    package_hash = package_integrity()
    verify_artefacts(fresh)
    before = inventory(fresh)
    replay_root = fresh.parent / 'trainval_replay'
    verify_artefacts(replay_root)
    replay_inventory = inventory(replay_root)
    if replay_inventory != before:
        raise ValueError('All fresh/replay file identities must match independently')
    stage = json.loads((fresh.parent / 'reports/stage_status.json').read_text())
    replay = json.loads((fresh.parent / 'reports/replay.json').read_text())
    if (stage.get('status') != 'PREPROCESSING_GATE_PASS_NOT_FULL_PILOT'
            or stage.get('mode') != 'raw_source_rebuild'
            or stage.get('source_rebuilt_here') is not True
            or replay.get('bitwise_identical') is not True):
        raise ValueError('Fresh raw rebuild and deterministic current replay are prerequisites')
    fresh_scaler_bytes = (fresh / 'relative_time_scaler.json').read_bytes()
    current = json.loads(fresh_scaler_bytes)
    with zipfile.ZipFile(reference) as z:
        old_bytes = z.read('campaign/frozen/development/relative_time_scaler.json')
        old = json.loads(old_bytes)
        delta = differences(old, current)
        if len(delta) != 1 or delta[0]['path'] != ['client_summaries', 2, 'M2', 1]:
            raise ValueError('Only the documented single C3 local-M2 diagnostic difference is permitted')
        if old['client_summaries'][2]['client'] != 'C3' or current['client_summaries'][2]['client'] != 'C3':
            raise ValueError('Diagnostic does not identify C3')
        a, b = delta[0]['reference'], delta[0]['fresh']
        if type(a) is not float or type(b) is not float or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Expected finite float64 diagnostic values')
        if math.nextafter(a, b) != b:
            raise ValueError('Diagnostic difference is not exactly one adjacent float64 ULP')
        if a.hex() != '0x1.cbf14cb1f49e1p+16' or b.hex() != '0x1.cbf14cb1f49e2p+16':
            raise ValueError('Diagnostic is not the specifically observed and reviewed pair')
        # Every top-level merged statistic and every applied transform parameter
        # must be EXACT, not within a numerical tolerance.
        top_old = {k: v for k, v in old.items() if k != 'client_summaries'}
        top_current = {k: v for k, v in current.items() if k != 'client_summaries'}
        if differences(top_old, top_current):
            raise ValueError('An applied transform parameter or merged statistic differs')
        old_sha = hashlib.sha256(old_bytes).hexdigest()
        fresh_sha = hashlib.sha256(fresh_scaler_bytes).hexdigest()
        if fresh_sha != 'f3ee0f61048cf5ebad1e2020af645006c6fdd9b8e04f6ec2ce56d6b7a7a7c22c':
            raise ValueError('Fresh scaler is not the exact reviewed raw-rebuild artifact')
        config = json.loads((source / 'reference/fullsize_B.json').read_text())
        if old_sha != config['scaler_sha256']:
            raise ValueError('Reference representation is not the fixed runtime scaler identity')
        fresh_summary = json.loads((fresh / 'preprocessing_summary.json').read_text())
        reference_summary = json.loads(z.read('campaign/frozen/development/preprocessing_summary.json'))
        summary_reference_deltas = differences(reference_summary, fresh_summary)
        expected_hash_paths = [['scaler_sha256']] + [['groups', i, 'scaler_sha256'] for i in range(6)]
        if [r['path'] for r in summary_reference_deltas] != expected_hash_paths:
            raise ValueError('Raw summary differs from historical summary beyond seven scaler hash links')
        if any(r['reference'] != old_sha or r['fresh'] != fresh_sha for r in summary_reference_deltas):
            raise ValueError('Raw summary scaler hash differences are unexpected')
        arrays = []
        npy_paths = sorted(fresh.rglob('*.npy'))
        ref_npys = sorted(n for n in z.namelist()
                          if n.startswith('campaign/frozen/development/') and n.endswith('.npy'))
        if len(npy_paths) != 30 or len(ref_npys) != 30:
            raise ValueError('Exactly 30 encoded development arrays are required')
        expected_names = ['campaign/frozen/development/' + n.relative_to(fresh).as_posix() for n in npy_paths]
        if sorted(expected_names) != ref_npys:
            raise ValueError('Reference/fresh array identities differ')
        for path, refname in zip(npy_paths, expected_names):
            raw = z.read(refname)
            old_array = np.load(io.BytesIO(raw), allow_pickle=False)
            new_array = np.load(path, allow_pickle=False)
            byte_equal = sha(path) == hashlib.sha256(raw).hexdigest()
            exact = (old_array.dtype == new_array.dtype and old_array.shape == new_array.shape
                     and np.array_equal(old_array, new_array))
            if not byte_equal or not exact:
                raise ValueError('Fresh encoded array is not exactly identical: ' + refname)
            arrays.append({'path': path.relative_to(fresh).as_posix(), 'sha256': sha(path),
                           'dtype': new_array.dtype.str, 'shape': list(new_array.shape),
                           'reference_bytes_exact': True, 'reference_values_exact': True})
    # No source/raw output is written. Metadata changes exist only in this copy.
    shutil.copytree(fresh, bound, copy_function=shutil.copy2)
    (bound / 'relative_time_scaler.json').write_bytes(old_bytes)
    summary_path = bound / 'preprocessing_summary.json'
    summary = json.loads(summary_path.read_text())
    changed_hash_links = []

    def rebind(obj, path=()):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'scaler_sha256':
                    if value != fresh_sha:
                        raise ValueError('Unexpected scaler hash link at ' + repr(path + (key,)))
                    obj[key] = old_sha
                    changed_hash_links.append(list(path + (key,)))
                else:
                    rebind(value, path + (key,))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                rebind(value, path + (i,))

    original_summary = json.loads((fresh / 'preprocessing_summary.json').read_text())
    rebind(summary)
    expected_links = [['scaler_sha256']] + [['groups', i, 'scaler_sha256'] for i in range(6)]
    if changed_hash_links != expected_links:
        raise ValueError('Expected exactly one global and six group hash links')
    observed_summary_delta = differences(original_summary, summary)
    if [r['path'] for r in observed_summary_delta] != expected_links:
        raise ValueError('Unexpected summary modification')
    write_json(summary_path, summary)
    if inventory(fresh) != before:
        raise ValueError('Fresh raw preprocessing changed before binding seal')
    copied = inventory(bound)
    allowed = {'relative_time_scaler.json', 'preprocessing_summary.json'}
    original_by_path = {r['path']: r for r in before}
    changed_files = []
    for row in copied:
        original = original_by_path[row['path']]
        if row != original:
            changed_files.append(row['path'])
            if row['path'] not in allowed:
                raise ValueError('Unexpected copied-file change: ' + row['path'])
    if set(changed_files) != allowed:
        raise ValueError('Exactly the two documented metadata files should change before resealing')
    receipt = {
        'status': 'ME06_CANONICAL_METADATA_BOUND_AFTER_EXACT_APPLIED_PARAMETER_AND_ARRAY_CHECKS',
        'new_reproduction': True,
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'wrapper_sha256': sha(__file__), 'package_manifest_sha256': package_hash,
        'fresh_raw_directory': str(fresh), 'bound_directory': str(bound),
        'reference_archive_sha256': sha(reference),
        'fresh_raw_outputs_preserved': True,
        'raw_scaler_json_byte_identity_reproduced': False,
        'raw_current_replay_bitwise_identical': True,
        'replay_independently_verified_files': len(replay_inventory),
        'fresh_scaler_sha256': fresh_sha, 'canonical_scaler_sha256': old_sha,
        'diagnostic_difference': {**delta[0], 'float64_ulp_distance': 1,
                                  'absolute_difference': abs(b-a),
                                  'reference_float_hex': a.hex(), 'fresh_float_hex': b.hex()},
        'all_other_scaler_json_fields_exact': True,
        'all_top_level_merged_statistics_exact': True,
        'applied_mean_and_scale_exact': True,
        'arrays': arrays, 'array_count': len(arrays),
        'array_source': 'Fresh raw preprocessing outputs; no archived array copied into training input',
        'canonical_reference_bytes_reused': ['relative_time_scaler.json'],
        'summary_hash_links_rebound': changed_hash_links,
        'additional_changes': ['artefact_manifest.csv and SHA256SUMS.txt regenerated with unchanged original seal_artefacts'],
        'scientific_parameters_rounded_or_retuned': False,
        'original_source_modified': False, 'old_learned_checkpoints_used': False,
        'test_metrics_computed': False,
        'fresh_original_file_inventory': before,
        'fresh_original_manifested_file_count': len(before) - 2,
        'fresh_original_total_files_including_two_seals': len(before),
        'limits': [
            'The historical scaler JSON diagnostic was not regenerated byte for byte from raw data.',
            'One local C3 M2 component differs by one adjacent float64 value; its origin is not established by this receipt.',
            'Applied scaling parameters and all 30 fresh arrays are exactly equal; canonical metadata is reused solely to satisfy the unchanged pinned runtime identity.',
            'This binding is an explicitly documented reproduction step, not a hidden full raw-byte reproducibility claim.',
            'The initial comparison checker expected 40 total files; there are 40 manifested payloads plus two sealing files, making 42 total. This is a validator counting correction, not a missing-data repair.'
        ]
    }
    write_json(bound / 'ME06_Metadata_Binding.json', receipt)
    manifest = seal_artefacts(bound)
    verify_artefacts(bound)
    if inventory(fresh) != before or inventory(replay_root) != replay_inventory or package_integrity() != package_hash:
        raise ValueError('Original raw/replay outputs or sealed source changed during binding')
    # Final receipt outside the directory can identify the sealed inside receipt
    # without a self-referential hash cycle.
    final = {'status': 'PASS', 'new_reproduction': True,
             'binding_receipt_sha256': sha(bound / 'ME06_Metadata_Binding.json'),
             'binding_manifest_sha256': sha(bound / 'artefact_manifest.csv'),
             'binding_sha256sums_sha256': sha(bound / 'SHA256SUMS.txt'),
             'manifested_files': len(manifest), 'total_files_including_seals': len(inventory(bound)),
             'fresh_raw_unchanged': True, 'fresh_replay_unchanged': True, 'package_unchanged': True,
             'raw_scaler_json_byte_identity_reproduced': False,
             'applied_parameters_and_30_fresh_arrays_exact': True}
    write_json(bound.parent / 'ME06_Metadata_Binding_Seal.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
