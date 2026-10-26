"""No final test is opened without a receipt bound to all 30 trained checkpoints/calibrations."""
from __future__ import annotations
import math,time
from pathlib import Path
import torch
from .common import *
from .evaluation import join_scores_labels,score_endpoints,complete_analysis
from .campaign import publish,environment,original_snapshot
from .test_inputs import load_final_inputs
from ehr_preprocessing.io_utils import BenchmarkSource
from ehr_preprocessing.dataset import collate_sessions
from ehr_fedavg.task import setup_torch,model_factory
from pilot_core.model import state_digest,session_losses
from central_smoke.engine import checked_losses


def validate_models(campaign:Path):
    path=campaign/'selected_models_manifest.json';v=read(path)
    if v['campaign_manifest_sha256']!=sha(campaign/'campaign_manifest.json'):raise ValueError('Selection manifest campaign drift')
    if v['package_manifest_sha256']!=package_integrity():raise ValueError('Selection source drift')
    models=v['models'];expected={(m,co,s) for m in 'ABC' for co in ('reference','noniid') for s in SEEDS}
    keys=[(r['configuration'],r['condition'],r['seed']) for r in models]
    if len(keys)!=30 or set(keys)!=expected:raise ValueError('All 30 planned selected models required')
    for r in models:
        if r['status']!='PASS' or r['rounds']!=30 or r['test_data_opened'] is not False:raise ValueError('Incomplete main cell')
        cp=safe_child(campaign,r['checkpoint']);cal=safe_child(campaign,r['calibration'])
        if sha(cp)!=r['selected_checkpoint_sha256'] or sha(cal)!=r['calibration_sha256']:raise ValueError('Selected weights/calibration altered')
        c=read(cal)
        if c['fit_partition']!='validation' or c['test_data_used'] is not False or c['selected_checkpoint_sha256']!=sha(cp):raise ValueError('Calibration provenance')
    return models


def score(model,samples):
    model.eval();before=state_digest(model);out=[]
    with torch.no_grad():
        for i in range(0,len(samples),64):
            group=samples[i:i+64];batch=collate_sessions(group);values=checked_losses(model,batch,session_losses)
            out.extend({'session_id':s['session_id'],'client':s['client'],'partition':s['partition'],'loss':float(v)} for s,v in zip(group,values))
    if before!=state_digest(model) or any(p.grad is not None for p in model.parameters()):raise ValueError('Final scoring altered model')
    return out


def evaluate(campaign:Path,receipt:Path,source_path:Path):
    verify_campaign(campaign);models=validate_models(campaign)
    require_authorisation(receipt,'EVALUATE_FINAL_TESTS',campaign,
                          expected_models_sha=sha(campaign/'selected_models_manifest.json'))
    target=campaign/'final_evaluation'
    target.mkdir(exist_ok=False)  # No silent rescore after viewing results.
    results=target/'reports';results.mkdir();limits=read(ROOT/'protocol/decisions/config/main_protocol_decisions.json')['data']
    status={'status':'FINAL_EVALUATION_STOP_REVIEW_REQUIRED','test_data_opened':False,'test_metrics_computed':False,
            'models_complete':30,'test_sets_separate':True,'no_test_threshold_search':True,'publication_and_exposure_limits':limits}
    prior=Path(read(campaign/'origins.json')['accepted_run']);before=original=None;rc=1
    try:
        before=environment(prior,results/'environment_before.json');original=original_snapshot(results/'original_environment_before.json')
        # Mark access prospectively; a failed evaluation never resets this fact.
        status['test_data_opened']=True;write(results/'stage_status.json',status)
        with BenchmarkSource(source_path,ROOT/'protocol/decisions/evidence/S02_source_file_manifest.csv') as source:
            write(results/'source_integrity.json',{'files':source.verify()})
            samples,labels,coverage=load_final_inputs(source,campaign)
        write(results/'encoding_coverage.json',coverage)
        setup_torch();cells=[]
        for row in models:
            m=model_factory();m.load_state_dict(torch.load(safe_child(campaign,row['checkpoint']),map_location='cpu',weights_only=True),strict=True)
            cal=read(safe_child(campaign,row['calibration']))
            for part in TESTS:
                scores=score(m,samples[part]);expected={s['session_id'] for s in samples[part]}
                joined=join_scores_labels(scores,labels[part],part,expected)
                cell={'configuration':row['configuration'],'condition':row['condition'],'seed':row['seed'],'test':part,
                      'checkpoint_sha256':row['selected_checkpoint_sha256'],'calibration_sha256':row['calibration_sha256'],
                      'metrics':score_endpoints(joined,cal)}
                cells.append(cell);name=f"{row['configuration']}_{row['condition']}_{row['seed']}_{part}"
                # Researcher-only scores; model/server training never reads these.
                csv_write(target/'LOCAL_ONLY'/f'{name}_scores.csv',joined)
                write(results/'cells'/f'{name}.json',cell,exclusive=True)
            print(f"FINAL_EVALUATION_CELL_SCORED {row['configuration']}/{row['condition']}/{row['seed']}",flush=True)
        write(results/'paired_seed_analysis.json',complete_analysis(cells))
        status.update(status='FINAL_EVALUATION_COMPLETE_LOCKED_MODELS_NO_RETUNING',test_metrics_computed=True,metric_cells=60)
        validate_models(campaign);rc=0
    except BaseException as e:
        import traceback
        status.update(error_type=type(e).__name__,message=str(e));(results/'failure_trace.txt').write_text(traceback.format_exc())
        print('STOP [final_evaluation]: '+str(e),flush=True)
    finally:
        try:
            if before is not None and environment(prior,results/'environment_after.json')!=before:raise ValueError('Environment changed')
            if original is not None and original_snapshot(results/'original_environment_after.json')!=original:raise ValueError('Original environment changed')
            verify_campaign(campaign)
        except BaseException as e:rc=1;status['postcheck_error']=str(e)
        if rc:status['status']='FINAL_EVALUATION_STOP_REVIEW_REQUIRED'
        write(results/'stage_status.json',status);publish(results,campaign,'Final_Evaluation_Report')
    return rc
