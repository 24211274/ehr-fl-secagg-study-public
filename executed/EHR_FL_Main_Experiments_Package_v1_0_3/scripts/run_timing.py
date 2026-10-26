#!/usr/bin/env python3
"""Predeclared Reference-only timing/payload repeats in the accepted environment.

No installers, new ML hyperparameters, final-test partitions or earlier trained
weights. Six fresh runs B-C / C-B / B-C; no automatic retry or warm-up selection.
"""
from __future__ import annotations
import argparse,datetime,fcntl,json,os,platform,shutil,subprocess,sys,time,traceback,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'app'),str(ROOT/'validation_support')]
from central_smoke.identity import sha256_file as sha,write_json,verify_manifest
from fullsize_support.integrity import read,check_report,package_report,copy_exclusive
from fullsize_support.data import verify_staged
from timing_support.data import verify_timing_inputs,stage_timing_data
from timing_support.analysis import extract_cell,summarise,write_csv
from ehr_fedavg.task import cfg
from ehr_fedavg.host import local_environment,write_project
from ehr_secagg.settings import SHA256 as C_CONFIG_SHA


def command(args,log,env,timeout):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    try:
        r=subprocess.run([str(x) for x in args],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        log.write_text(str(exc.stdout or '')+'\nTIMED OUT\n');raise
    log.write_text(r.stdout,encoding='utf-8')
    if r.returncode:raise RuntimeError(f'{Path(args[0]).name} command failed ({r.returncode}); see {log.name}')
    return r.stdout


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('package','fullsize-run','accepted-run','preprocessing-run'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--run-base',type=Path,default=Path.home()/'EHR_FL_Timing_BC_v1/runs')
    a=ap.parse_args();source=Path(a.package).resolve();full=Path(a.fullsize_run).resolve()
    prior=Path(a.accepted_run).resolve();pre=Path(a.preprocessing_run).resolve()
    if source!=ROOT:raise ValueError('Package path differs from running script')
    rid=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    runbase=a.run_base.expanduser().resolve();runbase.mkdir(parents=True,exist_ok=True)
    run=runbase/rid;run.mkdir(exist_ok=False);reports=run/'results';reports.mkdir();logs=run/'logs_LOCAL_ONLY';logs.mkdir()
    env=local_environment(run/'preflight_flower_home');env['PYTHONPATH']=str(ROOT/'app');env['PYTHONNOUSERSITE']='1'
    status={'status':'TIMING_BC_STOP_REVIEW_REQUIRED','stage':'initialisation','completed_runs':[],
            'deployment_engine_may_have_started':False,'test_data_opened':False,'test_metrics_computed':False,
            'model_tuning_performed':False,'main_experiments_frozen':False,'privacy_claim_validated':False,
            'new_environment_created':False,'pip_or_download_executed':False,'automatic_retry':False,
            'wire_bytes_measured':False,'inference_microbenchmark_executed':False}
    started=time.perf_counter_ns();rc=1;locks=[];before=None;original_before=None;staged={};outputs={};cells=[]
    full_identity=None;prior_identity=None;data_checked=False
    try:
        status['stage']='package_and_prior_evidence'
        verify_manifest(ROOT,ROOT/'PACKAGE_SHA256SUMS.txt')
        shutil.copytree(ROOT,run/'package_snapshot',ignore=shutil.ignore_patterns('__pycache__','Timing_BC_Report_*.zip'))
        policy=read(ROOT/'config/timing_policy.json');conf=cfg()
        for src,dst in [('timing_policy.json','timing_policy.json'),('timing_B.json','B_configuration.json'),('timing_C.json','C_configuration.json')]:
            shutil.copyfile(ROOT/'config'/src,reports/dst)
        full_identity=check_report(full/'results',policy['accepted_fullsize_manifest_sha256'])
        fs=read(full/'results/stage_status.json')
        if full.name!=policy['accepted_fullsize_run_id'] or fs.get('status')!='FULLSIZE_PATCHED_BC_PILOT_PASS_NOT_MAIN_OR_PRIVACY_CERTIFICATION' or fs.get('configuration_cells_completed')!=['B/reference','B/noniid','C/reference','C/noniid']:
            raise ValueError('The accepted full-size gate is required')
        prior_identity=check_report(prior/'results',policy['accepted_patched_manifest_sha256'])
        if prior.name!=policy['accepted_patched_run_id']:raise ValueError('Wrong accepted patched environment source')
        write_json(reports/'prior_evidence.json',{'fullsize':full_identity,'patched':prior_identity})
        if platform.system()!='Linux' or sys.version_info[:2]!=(3,12) or os.geteuid()==0:
            raise RuntimeError('Use accepted non-root Linux Python 3.12, no environment fallback')
        if Path(os.path.abspath(sys.executable)).parent!=prior/'.venv_full_candidate/bin':
            raise ValueError('Use accepted full candidate venv in place; no copied environment')
        # Also claim the predecessor full-size lock to avoid simultaneous workloads.
        for lockpath in [Path.home()/'EHR_FL_FullSize_BC_v1/fullsize.lock',runbase.parent/'timing.lock']:
            lockpath.parent.mkdir(parents=True,exist_ok=True);lock=lockpath.open('a+')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();raise RuntimeError('Another full-size/timing workload is running')
            locks.append(lock)
        status['stage']='environment_before'
        command([sys.executable,'-I','-B',ROOT/'scripts/environment_probe.py',prior,reports/'environment_before.json'],logs/'environment_before.log',env,180)
        before=read(reports/'environment_before.json')
        original=Path.home()/'EHR_FL_Pilot_Host_v1/.venv/bin/python'
        command([original,'-I','-B',ROOT/'scripts/host_snapshot.py',reports/'original_environment_before.json'],logs/'original_before.log',env,120)
        original_before=read(reports/'original_environment_before.json')
        if original_before!=read(ROOT/'reference/accepted_bc/original_environment_after.json'):
            raise ValueError('Original environment differs from accepted record')
        print('ACCEPTED_TIMING_ENVIRONMENT_CHECK_PASS_NO_INSTALL',flush=True)
        free={'linux_free_GiB':shutil.disk_usage(run).free/1024**3}
        if Path('/mnt/c').exists():free['windows_C_free_GiB']=shutil.disk_usage('/mnt/c').free/1024**3
        write_json(reports/'disk_preflight.json',free)
        if min(free.values())<policy['required_free_GiB']:raise RuntimeError('Insufficient free disk space')
        import psutil
        write_json(reports/'measurement_host.json',{'python':platform.python_version(),'platform':platform.platform(),
            'logical_cpus':os.cpu_count(),'cpu_affinity':psutil.Process().cpu_affinity(),
            'memory_total_bytes':psutil.virtual_memory().total,'rss_interval_seconds':conf['resource_sample_seconds'],
            'wall_clock':vars(time.get_clock_info('perf_counter')),'windows_power_policy_read':False,
            'background_load_not_controlled_by_script':True,'new_affinity_or_thread_policy_imposed':False})
        status['stage']='timing_input_replay'
        audit=verify_timing_inputs(ROOT,pre);data_checked=True;write_json(reports/'timing_data_audit.json',audit)
        for n in ['timing_membership.csv','timing_draws_six_rounds.csv']:shutil.copyfile(ROOT/'config'/n,reports/n)
        print('TIMING_SUBSETS_AND_SIX_ROUND_DRAWS_CHECK_PASS',flush=True)
        expected=read(prior/'results/B/reference/server/server_result.json')['initial_state_digest']
        status['stage']='all_unit_checks'
        command([sys.executable,'-B',ROOT/'scripts/run_unit_checks.py',reports/'unit_checks.json'],reports/'unit_checks.log',env,900)
        unit=read(reports/'unit_checks.json')
        if unit.get('status')!='PASS' or unit.get('skipped') or not unit.get('tests_run'):raise RuntimeError('All tests must execute with zero skips/failures')
        print('TIMING_UNIT_CHECKS_PASS',flush=True)
        status['stage']='public_protocol_oracle'
        from prg_validation.public_protocol import run_oracle
        write_json(reports/'public_protocol_oracle.json',run_oracle('candidate'))
        from fullsize_support.preflight import preflight_flower
        probe=run/'cli_preflight';(probe/'reports').mkdir(parents=True)
        paths=preflight_flower(conf,probe);shutil.copytree(probe/'reports',reports/'cli_preflight')
        from ehr_fedavg.resources import resource_sha256,verify_source_resources
        verify_source_resources(ROOT/'app')
        from ehr_fedavg.bundle_checks import preflight_native_fab as bf
        from ehr_secagg.bundle_checks import preflight_native_fab as cf
        # Stage and test all six FABs before measured services; never tune from output.
        for cell in policy['plan']:
            label=cell['run_id'];mode=cell['configuration'];out=run/'runs'/label
            (out/'reports').mkdir(parents=True,exist_ok=False);outputs[label]=out
            status['stage']='staging_'+label
            staged[label]=stage_timing_data(ROOT,pre,out)
            runtime={'fixed_config_sha256':resource_sha256('fixed_config.json'),'expected_initial_digest':expected,
                     'run_token':uuid.uuid4().hex,'client_files':staged[label]}
            if mode=='C':runtime.update(security_config_sha256=C_CONFIG_SHA,toy_oracle_passed=True)
            write_json(out/'runtime.json',runtime)
            write_json(out/'reports/cell_plan.json',cell)
            write_json(out/'reports/client_staging.json',{'files':staged[label],'logical_only_not_os_isolation':True})
            appdir=out/'apps/reference';shutil.copytree(ROOT/'app',appdir,ignore=shutil.ignore_patterns('__pycache__'))
            project=appdir/'pyproject.toml';write_project(project,out,'reference',conf['ports']['control'])
            if mode=='C':project.write_text(project.read_text().replace('name = "ehr-fedavg-fullsize"','name = "ehr-secagg-fullsize"').replace('ehr_fedavg.server_app:app','ehr_secagg.server_app:app').replace('ehr_fedavg.client_app:app','ehr_secagg.client_app:app'))
            status['stage']='native_fab_'+label
            (bf if mode=='B' else cf)(appdir,out,'reference',out/'reports/reference/app_bundle_preflight.json',local_environment(out/'bundle_flower_home'))
        from timing_support.deployment import run_condition
        for cell in policy['plan']:
            label=cell['run_id'];mode=cell['configuration'];out=outputs[label]
            status['stage']='deployment_'+label;status['deployment_engine_may_have_started']=True
            verify_manifest(ROOT,ROOT/'PACKAGE_SHA256SUMS.txt');verify_staged(out,staged[label])
            print(f"TIMING_RUN_START {label}: 6 rounds; round 1 retained as warm-up; measured rounds 2-6",flush=True)
            result=run_condition(out,'reference',paths,conf,mode)
            if result['initial_state_digest']!=expected:raise ValueError('Initial state differs from prior plan')
            measured=extract_cell(out,mode,cell['pair']);cells.append(measured)
            write_json(out/'reports/measurement_summary.json',measured)
            write_csv(out/'reports/round_measurements.csv',measured['rounds'])
            status['completed_runs'].append(label)
            print('TIMING_RUN_PASS '+label,flush=True)
        status['stage']='paired_measurement_audit'
        summary=summarise(cells,policy['plan'])
        write_json(reports/'paired_timing_summary.json',summary)
        write_csv(reports/'paired_contrasts.csv',summary['contrasts'])
        write_csv(reports/'all_round_measurements.csv',[r for c in cells for r in c['rounds']])
        status.update(local_optimizer_updates=432,global_aggregations=36,measured_fit_rounds=30,paired_repetitions=3)
        rc=0
    except (Exception,KeyboardInterrupt) as exc:
        status.update(failed_stage=status['stage'],error_type=type(exc).__name__,message=str(exc))
        (logs/'failure_trace_LOCAL_ONLY.txt').write_text(traceback.format_exc())
        write_json(reports/'failure_context.json',{'stage':status['stage'],'error_type':type(exc).__name__,
            'message':str(exc),'frames':[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)],
            'traceback_locals_exported':False})
        print(f"STOP [{status['stage']}]: {exc}",file=sys.stderr,flush=True)
    finally:
        try:
            if before is not None:
                command([sys.executable,'-I','-B',ROOT/'scripts/environment_probe.py',prior,reports/'environment_after.json'],logs/'environment_after.log',env,180)
                if before!=read(reports/'environment_after.json'):raise ValueError('Accepted environment changed')
                status['accepted_environment_unchanged_verified']=True
            if original_before is not None:
                command([Path.home()/'EHR_FL_Pilot_Host_v1/.venv/bin/python','-I','-B',ROOT/'scripts/host_snapshot.py',reports/'original_environment_after.json'],logs/'original_after.log',env,120)
                if original_before!=read(reports/'original_environment_after.json'):raise ValueError('Original environment changed')
                status['original_environment_unchanged_verified']=True
            if prior_identity is not None:check_report(prior/'results',prior_identity['manifest_sha256'])
            if full_identity is not None:check_report(full/'results',full_identity['manifest_sha256'])
            if data_checked:
                verify_timing_inputs(ROOT,pre)
                for label,files in staged.items():verify_staged(outputs[label],files)
                status['prepared_and_staged_data_unchanged_verified']=True
            verify_manifest(ROOT,ROOT/'PACKAGE_SHA256SUMS.txt')
        except BaseException as exc:rc=1;status['postcheck_error']=str(exc);print('STOP [postcheck]: '+str(exc),flush=True)
        for label,out in outputs.items():
            if (out/'reports').exists():shutil.copytree(out/'reports',reports/'runs'/label)
        status['elapsed_wall_ns_including_all_audits']=time.perf_counter_ns()-started
        if rc==0 and status['completed_runs']!=['TIMING-1-B','TIMING-1-C','TIMING-2-C','TIMING-2-B','TIMING-3-B','TIMING-3-C']:
            rc=1;status['postcheck_error']='Incomplete predeclared run order'
        if rc==0:status.update(status='TIMING_PATCHED_BC_PILOT_PASS_NOT_MAIN_OR_PRIVACY_CERTIFICATION',stage='completed')
        write_json(reports/'stage_status.json',status)
        target=run/('Timing_BC_Report_'+rid+'.zip')
        try:
            package_report(reports,target);copy_exclusive(target,source/target.name)
            print('WINDOWS_REPORT_FILENAME='+target.name,flush=True)
        except BaseException as exc:rc=1;print('STOP [report]: '+str(exc),flush=True)
        print('Report: '+str(target),flush=True);print('Local Linux run directory: '+str(run),flush=True)
        print(status['status'] if rc==0 else 'TIMING_BC_STOP_REVIEW_REQUIRED',flush=True)
        for lock in reversed(locks):fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return rc
if __name__=='__main__':raise SystemExit(main())
