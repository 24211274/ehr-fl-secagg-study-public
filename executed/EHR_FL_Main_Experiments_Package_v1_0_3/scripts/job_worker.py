#!/usr/bin/env python3
"""Run one prepared job in a fresh process, so fixed FAB resources cannot leak between seeds."""
from __future__ import annotations
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('job',type=Path);a.add_argument('action',choices=['Probe','Train'])
o=a.parse_args();job=o.job.resolve()
meta=json.loads((job/'job.json').read_text());co=meta['condition'];mode=meta['configuration']
sys.path[:0]=[str(job/'apps'/co),str(ROOT),str(ROOT/'validation_support')]
from main_study.common import read,write,sha,CLIENTS
from ehr_fedavg.task import cfg,model_factory,load_client,fresh_global
from pilot_core.model import state_digest
from main_study.evaluation import calibrate_validation


def main():
    conf=cfg();runtime=read(job/'runtime.json')
    if conf['training_seed']!=meta['training_seed'] or conf['rounds']!=meta['rounds'] or conf['conditions']!=[co]:raise ValueError('Job config binding')
    if state_digest(fresh_global())!=runtime['expected_initial_digest']:raise ValueError('Job initial-state mismatch')
    if o.action=='Probe':
        measurement_gate=None
        if mode in ('B','C'):
            from main_study.measurement_gate import check_measurement_horizon
            measurement_gate=check_measurement_horizon(job,mode,co)
            write(job/'reports/measurement_contract_preflight.json',measurement_gate)
            print('MAIN_MEASUREMENT_HORIZON_CHECK_PASS '+mode+'/'+co,flush=True)
        for c in CLIENTS:load_client(str(job),c)
        if mode in ('B','C'):
            from ehr_fedavg.host import local_environment
            if mode=='B':from ehr_fedavg.bundle_checks import preflight_native_fab
            else:from ehr_secagg.bundle_checks import preflight_native_fab
            preflight_native_fab(job/'apps'/co,job,co,job/'reports'/co/'app_bundle_preflight.json',
                                 local_environment(job/'probe_flower_home'))
        write(job/'reports/job_preflight.json',{'status':'PASS','configuration':mode,'seed':meta['training_seed'],
                 'condition':co,'rounds':meta['rounds'],'native_fab_built':mode in ('B','C'),'measurement_envelope_gate':measurement_gate,'test_data_opened':False})
        print('MAIN_JOB_PREFLIGHT_PASS '+job.name,flush=True);return
    if read(job/'reports/job_preflight.json')['status']!='PASS':raise ValueError('Missing native job gate')
    if (job/'reports/cell_complete.json').exists():raise FileExistsError('Completed job cannot be rerun')
    if mode=='A':
        from main_study.central import run_central
        s=run_central(job);rows=read(job/'LOCAL_ONLY/selected_validation_scores.json')['rows']
    else:
        from fullsize_support.preflight import preflight_flower
        from timing_support.deployment import run_condition
        paths=preflight_flower(conf,job)
        s=run_condition(job,co,paths,conf,mode)
        from main_study.costs import extract
        extract(job,co,mode,conf['rounds'])
        rows=[]
        for c in CLIENTS:
            x=read(job/f'client_outputs/{co}/{c}/evaluate_selected_{s["selected_round"]:02d}.json')
            rows.extend(r for r in x['session_losses'] if r['partition']=='validation')
    expected=set()
    for c in CLIENTS:
        _,samples,_,_=load_client(str(job),c);expected.update(r['session_id'] for r in samples['validation'])
    cp=job/'checkpoints'/co/'selected_state.pt'
    calibration=calibrate_validation(rows,expected)
    score_path=job/'LOCAL_ONLY/selected_validation_scores.json'
    if not score_path.exists():write(score_path,{'rows':rows},exclusive=True)
    calibration.update(selected_checkpoint_sha256=sha(cp),validation_scores_sha256=sha(score_path),
                       configuration=mode,condition=co,training_seed=meta['training_seed'])
    write(job/'calibration.json',calibration,exclusive=True)
    from main_study.inference import measure
    measure(job,cp)
    complete={'status':'PASS','configuration':mode,'condition':co,'seed':meta['training_seed'],
              'phase':conf['stage'],'main_training_performed':conf['stage']=='MAIN_CONTROLLED_COMPARISON',
              'inherited_pilot_freeze_flags_do_not_grant_campaign_authorisation':True,
              'rounds':conf['rounds'],'selected_round':s['selected_round'],
              'initial_state_digest':s['initial_state_digest'],'selected_checkpoint_sha256':sha(cp),
              'calibration_sha256':sha(job/'calibration.json'),'validation_scores_sha256':sha(score_path),
              'test_data_opened':False,'final_test_authorised':False,'privacy_certification':False}
    write(job/'reports/cell_complete.json',complete,exclusive=True)
    print('MAIN_CELL_COMPLETE '+mode+'/'+co+'/'+str(meta['training_seed']),flush=True)

try: main()
except BaseException as e:
    write(job/'reports/worker_failure.json',{'status':'FAIL','action':o.action,'exception':type(e).__name__,
        'message':str(e),'frames':[{'file':x.filename,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)],
        'locals_exported':False,'test_data_opened':False})
    print('STOP [main_job_worker]: '+str(e),file=sys.stderr,flush=True);raise
