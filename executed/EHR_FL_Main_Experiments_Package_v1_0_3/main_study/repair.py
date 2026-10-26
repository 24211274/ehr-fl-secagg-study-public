"""A reviewed code correction creates a successor, never edits the failed campaign.

The known completed A cell remains archived. No result-quality comparison selects
between old/new results; primary data collection will restart the complete first
block after a separately reviewed successor training release.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from .common import ROOT, read, sha, write, verify_rows, file_record
from fullsize_support.integrity import check_report
from central_smoke.identity import verify_manifest


def repair_policy():
    return read(ROOT/'reference/measurement_scope_fix/repair_policy.json')


def verify_repair_parent(parent: Path, successor: Path) -> dict:
    parent, successor = Path(parent).resolve(), Path(successor).resolve()
    p = repair_policy()
    if str(parent) != p['parent_campaign'] or parent == successor or successor.is_relative_to(parent):
        raise ValueError('Repair parent/destination does not match the reviewed campaign')
    if sha(parent/'campaign_manifest.json') != p['parent_campaign_manifest_sha256']:
        raise ValueError('Parent campaign manifest changed')
    m = read(parent/'campaign_manifest.json')
    if m['package_manifest_sha256'] != p['parent_source_manifest_sha256']:
        raise ValueError('Unexpected parent source release')
    if m['decision_sha256'] != sha(ROOT/'protocol/decisions/config/main_protocol_decisions.json'):
        raise ValueError('Scientific decisions changed; not this code repair')
    verify_rows(parent, m['files'])
    source = parent/'source'
    original_files = verify_manifest(source, source/'PACKAGE_SHA256SUMS.txt', p['parent_source_manifest_sha256'])
    expected = {r['path'] for r in original_files}|{'PACKAGE_SHA256SUMS.txt'}
    extras = [x.relative_to(source).as_posix() for x in source.rglob('*') if x.is_file()
              and '__pycache__' not in x.parts and x.suffix not in ('.pyc','.pyo')
              and not (x.parent == source and 'Report_' in x.name and x.suffix == '.zip')
              and x.relative_to(source).as_posix() not in expected]
    if extras:
        raise ValueError('Unlisted parent source artefacts')
    reports = parent/p['failure_report_relative']
    evidence = check_report(reports, p['failure_report_manifest_sha256'])
    if sha(reports/'block_status.json') != p['failure_block_status_sha256']:
        raise ValueError('Failed block is not the reviewed attempt')
    status = read(reports/'block_status.json')
    progress = read(parent/'progress.json')
    expected_attempt = {k:v for k,v in status.items() if k not in ('test_data_opened','privacy_claim_validated')}
    if (progress.get('completed_blocks') != [] or progress.get('attempts') != {p['block']:expected_attempt}
        or progress.get('main_training_started') is not True or progress.get('archived_attempts')):
        raise ValueError('Parent has progressed or contains another attempt; review required')
    if status['root_error_type'] != p.get('expected_wrapper_error_type', 'FileExistsError') or status['status'] != 'FAILED':
        raise ValueError('Unexpected failure category')
    if (parent/'final_evaluation').exists() or (parent/'selected_models_manifest.json').exists():
        raise ValueError('Parent is beyond the reviewed training-only boundary')
    preparation_evidence=check_report(parent/'prepare_reports',p['parent_preparation_report_manifest_sha256'])
    if read(parent/'prepare_reports/stage_status.json')['status'] != 'MAIN_PREPARATION_PASS_NOT_MAIN_TRAINING_OR_TEST':
        raise ValueError('Parent preparation not accepted')
    a = parent/status['path']/'A'
    complete = read(a/'reports/cell_complete.json')
    if sha(a/'reports/cell_complete.json') != p['old_complete_cell_sha256']:
        raise ValueError('Archived A completion record changed')
    for location, key in [('checkpoints/reference/selected_state.pt','selected_checkpoint_sha256'),
                          ('calibration.json','calibration_sha256'),
                          ('LOCAL_ONLY/selected_validation_scores.json','validation_scores_sha256')]:
        if sha(a/location) != complete[key]:
            raise ValueError('Archived A artefact changed: '+location)
    # In the latest repair, require the precise local ServerApp failure supplied
    # by the researcher, not merely a generic worker exit or performance result.
    partial_b = None
    if p.get('expected_server_failure'):
        b = parent/status['path']/'B'
        failure = read(b/'server_outputs/reference/server_result.json')
        if any(failure.get(k) != v for k,v in p['expected_server_failure'].items()):
            raise ValueError('Internal ServerApp failure is not the reviewed cause')
        if (b/'reports/cell_complete.json').exists():
            raise ValueError('Reviewed partial B has become complete')
        if (parent/status['path']/'C/reports/cell_complete.json').exists():
            raise ValueError('C progressed beyond the reviewed attempt')
        error = (b/'error_reference_LOCAL_ONLY.txt').read_text(encoding='utf-8')
        if ('mai_measurement.py' not in error or 'in descriptor' not in error
                or p['expected_server_failure']['message'] not in error):
            raise ValueError('Expected observer traceback absent')
        history = read(b/'server_outputs/reference/history.json')['history']
        rounds = [row['round'] for row in history]
        if not rounds or rounds != list(range(rounds[-1]+1)) or rounds[-1] not in (5,6):
            raise ValueError('Unexpected partial-B progress; separate review required')
        partial_b = {'status':'ARCHIVED_PARTIAL_NOT_PRIMARY',
                     'last_recorded_evaluation_round':rounds[-1],
                     'server_result_sha256':sha(b/'server_outputs/reference/server_result.json'),
                     'traceback_sha256':sha(b/'error_reference_LOCAL_ONLY.txt'),
                     'history_sha256':sha(b/'server_outputs/reference/history.json'),
                     'no_model_or_performance_selection':True}
    # Existing prepared probe directories must remain untouched in the parent.
    workdirs = {mode: (parent/'jobs'/f'MAIN-11-reference-{mode}'/'bundle_preflight/reference').is_dir()
                for mode in ('B','C')}
    if not all(workdirs.values()):
        raise ValueError('Expected parent FAB diagnostic directories are absent; do not delete them')
    return {'status':'PASS','repair_id':p['repair_id'],'parent_campaign':str(parent),
            'parent_campaign_manifest_sha256':sha(parent/'campaign_manifest.json'),
            'parent_source_manifest_sha256':p['parent_source_manifest_sha256'],
            'parent_progress_sha256':sha(parent/'progress.json'),
            'failure_evidence':evidence,'parent_preparation_evidence':preparation_evidence, 'prepared_probe_directories_present':workdirs,
            'parent_sealed_files_verified':len(m['files']),
            'archived_completed_cells':[complete],'archived_partial_B':partial_b,
            'old_A_selected_for_primary_analysis':False,
            'frozen_rows':[r for r in m['files'] if r['path'].startswith('frozen/')],
            'parent_origins_sha256':sha(parent/'origins.json'),
            'test_data_read_by_repair':False,'training_authorised':False}


def copy_parent_frozen(parent: Path, successor: Path, audit: dict) -> dict:
    """No new sampling, initialisation or fitted preprocessing during repair."""
    parent, successor = Path(parent), Path(successor)
    rows = audit['frozen_rows']
    if not rows:
        raise ValueError('No frozen artefacts in reviewed campaign')
    verify_rows(parent, rows)
    frozen = parent/'frozen'
    expected = {r['path'][len('frozen/'):] for r in rows}
    actual = {x.relative_to(frozen).as_posix() for x in frozen.rglob('*') if x.is_file()}
    if actual != expected or any(x.is_symlink() for x in frozen.rglob('*')):
        raise ValueError('Unexpected frozen input/symlink in parent')
    shutil.copytree(frozen, successor/'frozen')
    verify_rows(successor, rows)
    source = read(parent/'origins.json')
    write(successor/'origins.json',source,exclusive=True)
    if sha(successor/'origins.json') != audit['parent_origins_sha256']:
        raise ValueError('Origins changed during successor preparation')
    return {'status':'PASS','copied_files':len(rows),'all_sha256_identical':True,
            'initial_states_regenerated':False,'sampling_regenerated':False,
            'preprocessing_refitted':False,'trained_checkpoints_reused':False,
            'shared_inodes_with_parent':sum((parent/r['path']).stat().st_ino == (successor/r['path']).stat().st_ino
                                           for r in rows),
            'test_data_opened':False}
