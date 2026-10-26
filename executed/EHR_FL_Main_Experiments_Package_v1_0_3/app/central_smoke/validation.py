"""Acceptance is numerical/structural, not a target detection score."""
from __future__ import annotations
import json
from pathlib import Path
from .identity import sha256_file


def validate_config(cfg,inherited,matrix):
    if cfg['configuration']!='A' or cfg['conditions']!=['reference','noniid'] or cfg['run_ids']!=['SMOKE-reference-A','SMOKE-noniid-A']:
        raise ValueError('Only the two planned centralised smoke cells are permitted')
    checks={'selection_seed':inherited['pilot']['selection_seed'],'training_seed':inherited['pilot']['selection_seed'],
        'outer_epochs':inherited['pilot']['smoke_rounds'],'training_sessions_per_client':inherited['pilot']['smoke_training_sessions_per_client'],
        'validation_sessions_per_client':inherited['pilot']['smoke_validation_sessions_per_client'],
        'input_dim':inherited['schema']['input_dim'],'schema_sha256':inherited['schema']['sha256'],
        'batch_sessions':inherited['training']['batch_sessions'],'learning_rate':inherited['training']['learning_rate'],
        'weight_decay':inherited['training']['weight_decay'],'betas':inherited['training']['betas'],'epsilon':inherited['training']['epsilon'],
        'gradient_clip_global_l2':inherited['training']['gradient_clip_global_l2'],'model_dropout_during_training':inherited['model']['decoder_output_dropout'],
        'device':inherited['training']['device'],'foreach':False,'fused':False,'torch_threads':1,'torch_interop_threads':1}
    if any(cfg[k]!=v for k,v in checks.items()):raise ValueError('Stage controls differ from the inherited protocol')
    if (cfg['outer_epochs'],cfg['schedule_audit_epochs'],cfg['training_sessions_per_client'],cfg['validation_sessions_per_client'])!=(3,5,256,128):
        raise ValueError('Unexpected smoke budget')
    if cfg['checkpoint_tie_tolerance']!=1e-12 or cfg['dominant_departments']!=inherited['sampling']['dominant_departments']:
        raise ValueError('Checkpoint/sampling policy changed')
    for co,run_id in zip(cfg['conditions'],cfg['run_ids']):
        r=[r for r in matrix if r['run_id']==run_id]
        if len(r)!=1 or r[0]['configuration']!='A' or r[0]['distribution']!=co or int(r[0]['training_seed'])!=cfg['training_seed'] or int(r[0]['outer_rounds_or_epochs'])!=3:
            raise ValueError('Cell disagrees with original run matrix')


def check_preprocessing_gate(path,cfg):
    s=json.loads(path.read_text())
    if s.get('status')!='PREPROCESSING_GATE_PASS_NOT_FULL_PILOT' or s.get('model_interface_checked') is not True:
        raise ValueError('Previous preprocessing gate did not pass')
    for k in ['input_dim','schema_sha256','scaler_sha256']:
        if s.get(k)!=cfg[k]:raise ValueError('Previous preprocessing identity differs: '+k)
    return s


def check_tiny_reports(reportdir,cfg):
    status=json.loads((reportdir/'stage_status.json').read_text());result=json.loads((reportdir/'tiny_training_result.json').read_text())
    ident=json.loads((reportdir/'input_identity.json').read_text())
    if status.get('status')!='TINY_TRAINING_GATE_PASS_NOT_FULL_PILOT' or result.get('pass') is not True or status.get('optimiser_steps_completed')!=100:
        raise ValueError('Previous tiny-training gate did not pass')
    for k in ['base_release_manifest_sha256','schema_sha256','scaler_sha256']:
        if ident.get(k)!=cfg[k]:raise ValueError('Tiny stage used different inputs: '+k)
    if status.get('input_dim')!=119 or status.get('test_metrics_computed') is not False:
        raise ValueError('Unexpected tiny gate scope')
    entry=json.loads((reportdir/'report_files.json').read_text())['files']
    seen=set()
    for r in entry:
        p=Path(r['path'])
        if p.is_absolute() or '..' in p.parts or str(p) in seen:raise ValueError('Invalid prior report manifest')
        seen.add(str(p));f=reportdir/p
        if sha256_file(f)!=r['sha256'] or f.stat().st_size!=r['bytes']:raise ValueError('Prior tiny report integrity failure')
    return {'status':'PASS','previous_tiny_status':status['status'],'files_verified':len(entry),
        'stage_status_sha256':sha256_file(reportdir/'stage_status.json'),
        'note':'Prior report checked; previous trained weights are neither opened nor reused.'}
