"""Preparation/ten matched blocks/test release are separate, fail-closed states."""
from __future__ import annotations
import datetime,fcntl,hashlib,json,os,platform,shutil,subprocess,sys,time,traceback,uuid
from pathlib import Path
from .common import *
from .design import (plan,ordered_blocks,department_rows,make_draws,distribution_audit,
                     cell_config,stage_cell,decisions,DRAW_FIELDS)
from .staging import stage_attempt_job
from central_smoke.identity import verify_manifest
from fullsize_support.integrity import check_report,package_report,copy_exclusive


def command(args,log:Path,timeout:int=1800,stream:bool=False):
    """Preserve an operational log; no automatic retries or pip calls."""
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
    env['PATH']=str(Path(os.path.abspath(sys.executable)).parent)+os.pathsep+env.get('PATH','')
    env.pop('PYTHONPATH',None)
    start=time.monotonic()
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen([str(a) for a in args],stdout=f,stderr=subprocess.STDOUT,env=env)
        offset=0
        try:
            while p.poll() is None:
                if stream:
                    with log.open(errors='replace') as r:r.seek(offset);text=r.read();offset=r.tell()
                    for line in text.splitlines():
                        if any(x in line for x in ('[A/','[B/','[C/','MAIN_','STOP','FLOWER_PRG_NETWORK')):print(line,flush=True)
                if time.monotonic()-start>timeout:raise TimeoutError('Subprocess watchdog: '+log.name)
                time.sleep(.25)
        except BaseException:
            # SIGINT permits the existing owned-service finally blocks to run.
            import signal
            if p.poll() is None:p.send_signal(signal.SIGINT)
            try:p.wait(timeout=25)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            raise
    if p.returncode:raise RuntimeError(f'Worker exit {p.returncode}; inspect {log.name}')
    if stream:
        with log.open(errors='replace') as r:r.seek(offset);text=r.read()
        for line in text.splitlines():
            if any(x in line for x in ('[A/','[B/','[C/','MAIN_','STOP')):print(line,flush=True)


def environment(prior:Path,out:Path):
    command([sys.executable,'-I','-B',ROOT/'scripts/environment_probe.py',prior,out],out.with_suffix('.log'),240)
    return read(out)


def original_snapshot(out:Path):
    python=Path.home()/'EHR_FL_Pilot_Host_v1/.venv/bin/python'
    command([python,'-I','-B',ROOT/'scripts/host_snapshot.py',out],out.with_suffix('.log'),180)
    result=read(out)
    if result!=read(ROOT/'reference/accepted_bc/original_environment_after.json'):raise ValueError('Original environment drift')
    return result


def verify_development(pre:Path):
    from ehr_preprocessing.io_utils import verify_artefacts
    from ehr_preprocessing.dataset import SessionDataset
    from collections import Counter
    base=Path(pre)/'package';data=base/'prepared_trainval';conf=read(ROOT/'reference/fullsize_B.json')
    if read(Path(pre)/'results/reports/stage_status.json')['status']!='PREPROCESSING_GATE_PASS_NOT_FULL_PILOT':raise ValueError('Preprocessing gate missing')
    verify_manifest(base,base/'PACKAGE_SHA256SUMS.txt',conf['base_release_manifest_sha256'])
    files=verify_artefacts(data)
    if sha(data/'encoding_schema.json')!=conf['schema_sha256'] or sha(data/'relative_time_scaler.json')!=conf['scaler_sha256']:
        raise ValueError('Approved schema/scaler changed')
    rows=read_csv(data/'session_manifest.csv');members=read_csv(ROOT/'reference/full_membership.csv')
    if [{k:r[k] for k in members[0]} for r in rows]!=members:raise ValueError('Full membership changed')
    if len({r['session_id'] for r in rows})!=len(rows) or any(r['partition'] not in ('training','validation') for r in rows):raise ValueError('Development membership invalid')
    # Cross-check recorded modal training departments against encoded training only.
    schema=read(data/'encoding_schema.json');field=schema['categorical_fields'][2]
    if field['name']!='department':raise ValueError('Department field changed')
    expected={(r['client'],r['session_id']):r['department'] for r in department_rows()};count=0
    for c in CLIENTS:
        ds=SessionDataset(data,c,'training',verify=False)
        for i in range(len(ds)):
            s=ds[i]
            if not s['categorical_valid'][:,2].all():raise ValueError('Unknown training department')
            counts=Counter(field['categories'][int(x)] for x in s['categorical'][:,2])
            mode=sorted(counts,key=lambda k:(-counts[k],k))[0]
            if mode!=expected[(c,s['session_id'])]:raise ValueError('Training modal department mismatch')
            count+=1
    return {'status':'PASS','files':len(files),'training_modes_checked':count,'sessions':len(rows),'test_data_opened':False}


def run_tests(reports:Path):
    command([sys.executable,'-B',ROOT/'scripts/run_main_tests.py',reports/'unit_checks.json'],reports/'unit_checks.log',1800)
    unit=read(reports/'unit_checks.json')
    if unit['status']!='PASS' or unit['skipped'] or not unit['tests_run']:raise ValueError('All native tests must execute; no skips permitted on target')
    print('MAIN_UNIT_AND_EVALUATOR_FIXTURE_CHECKS_PASS',flush=True)


def prepare(campaign:Path,inference:Path,prior:Path,pre:Path,*,repair_parent:Path|None=None):
    reports=campaign/'prepare_reports';reports.mkdir(parents=True,exist_ok=False)
    status={'status':'MAIN_PREPARATION_STOP_REVIEW_REQUIRED','stage':'environment','main_training_performed':False,
        'test_data_opened':False,'test_metrics_computed':False,'privacy_claim_validated':False,
        'main_experiments_frozen':False,'training_authorised':False,'main_evaluation_authorised':False}
    before=original=None;rc=1;parent_audit=None
    try:
        package_hash=package_integrity()
        if repair_parent is not None:
            from .repair import verify_repair_parent, copy_parent_frozen
            parent_audit=verify_repair_parent(repair_parent,campaign)
            status.update(code_repair_preparation=True,historical_main_training_occurred_in_parent=True)
            expected_origins=read(repair_parent/'origins.json')
            if expected_origins!={'inference_run':str(inference),'accepted_run':str(prior),'preprocessing_run':str(pre)}:
                raise ValueError('Successor must use the original accepted run paths')
            write(reports/'parent_campaign_review.json',parent_audit)
            print('REVIEWED_FAILED_CAMPAIGN_CHECK_PASS_NO_MUTATION',flush=True)
        expected_hash=sha(ROOT/'reference/accepted_inference_seed/report_files.json')
        previous=check_report(Path(inference)/'results',expected_hash)
        if read(Path(inference)/'results/stage_status.json')['status']!='INFERENCE_SEED_PILOT_PASS_NOT_MAIN_OR_PRIVACY_CERTIFICATION':raise ValueError('Prior gate not passed')
        write(reports/'predecessor_gate.json',previous)
        before=environment(prior,reports/'environment_before.json');original=original_snapshot(reports/'original_environment_before.json')
        print('ACCEPTED_MAIN_ENVIRONMENT_CHECK_PASS_NO_INSTALL',flush=True)
        if shutil.disk_usage(campaign).free<4*1024**3:raise RuntimeError('At least 4 GiB free required for preparation')
        status['stage']='development';write(reports/'development_audit.json',verify_development(pre))
        frozen=campaign/'frozen'
        if repair_parent is not None:
            reuse=copy_parent_frozen(repair_parent,campaign,parent_audit)
            write(reports/'frozen_input_reuse.json',reuse)
            initials=read(frozen/'initial_states.json')
            if set(initials)!={str(seed) for seed in SEEDS}:
                raise ValueError('Frozen initial state inventory differs')
            schedules={(seed,co):frozen/'schedules'/f'{seed}_{co}.csv'
                       for seed in SEEDS for co in ('reference','noniid')}
            for seed in SEEDS:
                if sha(frozen/'initials'/f'seed_{seed}.pt')!=initials[str(seed)]['sha256']:
                    raise ValueError('Frozen initial checkpoint changed')
            shutil.copyfile(repair_parent/'prepare_reports/sampling_audits.json',reports/'sampling_audits.json')
            run_tests(reports)
            status['stage']='public_protocol'
            from prg_validation.public_protocol import run_oracle
            write(reports/'public_protocol.json',run_oracle('candidate'))
            print('PARENT_FROZEN_INPUTS_INITIALS_AND_SCHEDULES_REUSED_BYTE_IDENTICAL',flush=True)
        else:
            frozen=campaign/'frozen';frozen.mkdir(exist_ok=False)
            shutil.copytree(Path(pre)/'package/prepared_trainval',frozen/'development')
            shutil.copyfile(ROOT/'protocol/decisions/config/main_run_order.csv',frozen/'main_run_order.csv')
            write(frozen/'provenance_limits.json',decisions()['data'])
            write(campaign/'origins.json',{'inference_run':str(inference),'accepted_run':str(prior),'preprocessing_run':str(pre)})
            run_tests(reports)
            status['stage']='public_protocol'
            from prg_validation.public_protocol import run_oracle
            write(reports/'public_protocol.json',run_oracle('candidate'))
            from ehr_fedavg.task import setup_torch,model_factory
            from pilot_core.model import set_training_seed,state_digest
            import torch
            setup_torch();initials={}
            for seed in SEEDS:
                set_training_seed(seed);model=model_factory();p=frozen/'initials'/f'seed_{seed}.pt';p.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(),p);initials[str(seed)]={'state_digest':state_digest(model),'sha256':sha(p)}
                second=model_factory();second.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
                if state_digest(second)!=initials[str(seed)]['state_digest']:raise ValueError('Initial checkpoint reload')
            if len({r['state_digest'] for r in initials.values()})!=5:raise ValueError('Main initial seeds not distinct')
            write(frozen/'initial_states.json',initials)
            ds=department_rows();schedules={};audits=[]
            for seed in SEEDS:
                nr=make_draws(seed,'noniid',30,ds);audit=distribution_audit(ds,nr);audit['seed']=seed;audits.append(audit)
                for co in ('reference','noniid'):
                    rows=make_draws(seed,co,30,ds) if co=='reference' else nr
                    p=frozen/'schedules'/f'{seed}_{co}.csv';csv_write(p,rows,DRAW_FIELDS);schedules[(seed,co)]=p
            write(reports/'sampling_audits.json',{'audits':audits,'exact_occurrences':10833*30*10,'test_data_used':False})
            print('FIVE_MAIN_INITIAL_STATES_AND_30_ROUND_SCHEDULES_PASS',flush=True)
        status['stage']='job_materialisation_and_native_FAB'
        for row in plan():
            seed=int(row['training_seed']);co=row['condition'];mode=row['configuration'];conf=cell_config(seed,co,30)
            schedule=schedules[(seed,co)];conf['schedule_sha256']=sha(schedule)
            job=campaign/'jobs'/row['planned_run_id']
            stage_cell(campaign,job,conf,schedule,initials[str(seed)]['state_digest'],mode)
            command([sys.executable,'-I','-B',ROOT/'scripts/job_worker.py',job,'Probe'],reports/'native_probes'/f'{job.name}.log',600)
            shutil.copyfile(job/'reports/job_preflight.json',reports/'native_probes'/f'{job.name}.json')
            if mode in ('B','C'):
                # Reproduce the failing lifecycle: successful template Probe -> copy -> Probe.
                # Use the identical staging helper called by execute_block, no special cleanup.
                clone=stage_attempt_job(job,campaign/'attempt_clone_probes'/job.name)
                command([sys.executable,'-I','-B',ROOT/'scripts/job_worker.py',clone,'Probe'],
                        reports/'native_clone_probes'/f'{job.name}.log',600)
                shutil.copyfile(clone/'reports/job_preflight.json',reports/'native_clone_probes'/f'{job.name}.json')
        print('ALL_20_MEASUREMENT_HORIZON_AND_CONDITION_PROBES_PASS',flush=True)
        print('ALL_20_PREPARED_TEMPLATE_COPY_AND_NATIVE_REPROBE_CHECKS_PASS',flush=True)
        print('ALL_30_JOB_CONFIGURATIONS_AND_20_NATIVE_FABS_PASS',flush=True)
        if repair_parent is not None:
            # Numerical A implementation is unchanged; preserve its accepted engineering gate.
            # Do not repeat main A or mix the partial failed block into the successor.
            summary=read(repair_parent/'prepare_reports/engineering_A_summary.json')
            if summary['status']!='PASS' or summary['rounds_completed']!=1 or summary['local_updates']!=170:
                raise ValueError('Inherited A compatibility evidence not as expected')
            write(reports/'engineering_A_summary.json',summary)
            write(reports/'engineering_A_evidence_scope.json',{
                'inherited_from_parent':True,'executed_in_this_preparation':False,
                'main_A_old_result_archived_not_selected':True})
            status['engineering_updates_this_preparation']=0
            print('UNCHANGED_A_ENGINEERING_GATE_RETAINED_NO_TRAINING_REPEATED',flush=True)
        else:
            # Previously untested full-size A in the pinned environment: one fixed engineering epoch, not a main run.
            status['stage']='central_fullsize_engineering_epoch'
            seed=20260906;set_training_seed(seed);expected=state_digest(model_factory())
            schedule=campaign/'engineering_draws.csv';csv_write(schedule,make_draws(seed,'reference',1,ds),DRAW_FIELDS)
            conf=cell_config(seed,'reference',1,engineering=True);conf['schedule_sha256']=sha(schedule)
            job=stage_cell(campaign,campaign/'engineering_A',conf,schedule,expected,'A')
            command([sys.executable,'-I','-B',ROOT/'scripts/job_worker.py',job,'Probe'],reports/'engineering_A_probe.log',600)
            command([sys.executable,'-I','-B',ROOT/'scripts/job_worker.py',job,'Train'],reports/'engineering_A.log',4000,True)
            result=read(job/'central_results/result.json')
            if result['local_updates']!=170 or result['partial_last_batch_sessions']!=17:raise ValueError('Full-size A epoch accounting')
            shutil.copyfile(job/'central_results/result.json',reports/'engineering_A_summary.json')
            print('FULLSIZE_A_ONE_EPOCH_COMPATIBILITY_PASS_NOT_MAIN',flush=True)
            status['engineering_updates']=170
        status['stage']='seal_campaign'
        files=inventory(frozen)
        for row in plan():
            j=campaign/'jobs'/row['planned_run_id']
            for directory in ('apps','client_data'):
                files.extend(file_record(p,campaign) for p in sorted((j/directory).rglob('*'))
                    if p.is_file() and p.suffix not in ('.pyc','.pyo','.fab') and '__pycache__' not in p.parts)
            files.extend(file_record(j/n,campaign) for n in ('runtime.json','job.json'))
        # Frozen inventory paths above are rooted at frozen; convert to the campaign root.
        prefix='frozen/'
        n=len(inventory(frozen))
        for r in files[:n]:r['path']=prefix+r['path']
        files.append(file_record(campaign/'origins.json',campaign))
        if repair_parent is not None:
            from .repair import repair_policy
            amendment={**repair_policy(),'successor_campaign':str(campaign),
                       'new_package_manifest_sha256':package_hash,
                       'inherited_frozen_input_files':len(parent_audit['frozen_rows']),
                       'repaired_source_before_new_training':True,
                       'historical_main_training_occurred_in_parent':True,
                       'successor_training_authorised':False,
                       'old_A_not_a_candidate_for_primary_selection':True}
            write(campaign/'code_repair_amendment.json',amendment,exclusive=True)
            files.append(file_record(campaign/'code_repair_amendment.json',campaign))
            write(reports/'code_repair_amendment.json',amendment)
        write(campaign/'campaign_manifest.json',{'version':'1.0.1','package_manifest_sha256':package_hash,
            'decision_sha256':sha(ROOT/'protocol/decisions/config/main_protocol_decisions.json'),
            'files':files,'training_authorised':False,'test_authorised':False,
            'public_availability_verified':False,'licence_verified':False},exclusive=True)
        verify_campaign(campaign)
        write(reports/'ready_receipt.json',{'status':'READY_FOR_TRAINING_RELEASE_REVIEW',
            'campaign':str(campaign),'campaign_manifest_sha256':sha(campaign/'campaign_manifest.json'),
            'package_manifest_sha256':package_hash,'jobs_prepared':30,'main_initial_states':initials,
            'main_training_performed':False,'final_test_authorised':False})
        write(campaign/'progress.json',{'completed_blocks':[],'attempts':{},'main_training_started':False},exclusive=True)
        rc=0
    except BaseException as e:
        status.update(error_type=type(e).__name__,message=str(e),failed_stage=status['stage'])
        (reports/'failure_trace.txt').write_text(traceback.format_exc());print(f"STOP [{status['stage']}]: {e}",flush=True)
    finally:
        try:
            if before is not None and environment(prior,reports/'environment_after.json')!=before:raise ValueError('Accepted environment changed')
            if original is not None and original_snapshot(reports/'original_environment_after.json')!=original:raise ValueError('Original environment changed')
            package_integrity()
            if parent_audit is not None:
                if verify_repair_parent(repair_parent,campaign)!=parent_audit:
                    raise ValueError('Parent campaign evidence changed during repair preparation')
                write(reports/'parent_postcheck.json',{'status':'PASS','checked_parent_unchanged':True})
        except BaseException as e:rc=1;status['postcheck_error']=str(e)
        if rc==0:status.update(status='MAIN_PREPARATION_PASS_NOT_MAIN_TRAINING_OR_TEST',stage='prepared')
        write(reports/'stage_status.json',status)
        publish(reports,campaign,'Main_Preparation_Report')
        print(status['status'],flush=True)
    return rc


def publish(reports:Path,campaign:Path,prefix:str):
    p=Path(campaign)/(prefix+'_'+Path(campaign).name+'_'+uuid.uuid4().hex[:6]+'.zip')
    package_report(reports,p);copy_exclusive(p,ROOT/p.name)
    print('WINDOWS_REPORT_FILENAME='+p.name,flush=True);print('Local Linux campaign: '+str(campaign),flush=True)
    print('Report: '+str(p),flush=True)


def validate_finished_job(job:Path):
    s=read(job/'reports/cell_complete.json');m=read(job/'job.json');cp=job/'checkpoints'/m['condition']/'selected_state.pt'
    if s['status']!='PASS' or sha(cp)!=s['selected_checkpoint_sha256'] or sha(job/'calibration.json')!=s['calibration_sha256']:
        raise ValueError('Finished job selection/calibration altered')
    return s


def train_next(campaign:Path,approval:Path):
    m=verify_campaign(campaign);require_authorisation(approval,'TRAIN',campaign)
    if read(campaign/'prepare_reports/stage_status.json')['status']!='MAIN_PREPARATION_PASS_NOT_MAIN_TRAINING_OR_TEST':raise ValueError('Preparation did not pass')
    progress=read(campaign/'progress.json');blocks=ordered_blocks()
    if progress['completed_blocks']!=blocks[:len(progress['completed_blocks'])]:raise ValueError('Changed block order')
    if len(progress['completed_blocks'])==len(blocks):raise ValueError('All main blocks already completed')
    block=blocks[len(progress['completed_blocks'])]
    if block in progress['attempts']:raise RuntimeError('Existing block attempt: review failure; no automatic resume/retry')
    return execute_block(campaign,block,progress,attempt=1)


def transient_type_eligible(entry:dict)->bool:
    # Type alone is NOT sufficient: the external receipt must also document a
    # reviewed transient cause. Unknown wrapper RuntimeError is never sufficient.
    kind=entry.get('root_error_type',entry.get('error_type'))
    return kind in ('TimeoutError','ConnectionError','ConnectionResetError',
                    'BrokenPipeError','OSError','InterruptedError','KeyboardInterrupt')


def retry_block(campaign:Path,approval:Path):
    verify_campaign(campaign);v=require_authorisation(approval,'RETRY_TRANSIENT_BLOCK',campaign)
    progress=read(campaign/'progress.json');block=ordered_blocks()[len(progress['completed_blocks'])]
    previous=progress['attempts'].get(block)
    if not previous or previous['attempt']!=1 or previous['status']!='FAILED':raise PermissionError('Only one reviewed retry of a failed first attempt')
    if not transient_type_eligible(previous):raise PermissionError('Unclassified, code, security or numerical fault requires review/repair; not an infrastructure retry')
    if v.get('fault_class')!='TRANSIENT_INFRASTRUCTURE' or v.get('block_id')!=block or len(v.get('reason',''))<20:
        raise PermissionError('Documented transient cause and exact block required')
    # Do not relabel slow or unfavourable completed outcomes as faults.
    progress.setdefault('archived_attempts',[]).append(previous)
    return execute_block(campaign,block,progress,attempt=2)


def execute_block(campaign:Path,block:str,progress:dict,attempt:int):
    origins=read(campaign/'origins.json');prior=Path(origins['accepted_run'])
    out=campaign/'attempts'/block/f'attempt_{attempt}';out.mkdir(parents=True,exist_ok=False)
    reports=out/'reports';reports.mkdir();before=original=None;rc=1;done=[];job=None
    entry={'block_id':block,'attempt':attempt,'status':'RUNNING','path':out.relative_to(campaign).as_posix()}
    progress['attempts'][block]=entry;progress['main_training_started']=True;write(campaign/'progress.json',progress)
    try:
        if shutil.disk_usage(campaign).free<8*1024**3:raise RuntimeError('At least 8 GiB free required before a main block')
        before=environment(prior,reports/'environment_before.json');original=original_snapshot(reports/'original_environment_before.json')
        rows=[r for r in plan() if r['block_id']==block]
        staged=[]
        for r in rows:
            verify_campaign(campaign)
            template=campaign/'jobs'/r['planned_run_id'];job=out/r['configuration']
            stage_attempt_job(template,job)
            command([sys.executable,'-I','-B',ROOT/'scripts/job_worker.py',job,'Probe'],reports/(r['configuration']+'_probe.log'),600)
            staged.append((r,job))
        print('ALL_THREE_BLOCK_PREFLIGHTS_PASS_BEFORE_ANY_TRAINING '+block,flush=True)
        for r,job in staged:
            verify_campaign(campaign)
            print('MAIN_RUN_START '+r['planned_run_id'],flush=True)
            command([sys.executable,'-I','-B',ROOT/'scripts/job_worker.py',job,'Train'],reports/(r['configuration']+'_train.log'),22200,True)
            s=validate_finished_job(job);done.append(s)
            # Copy only aggregate reports, not individual scores, weights or protocol stores.
            shutil.copytree(job/'reports',reports/r['configuration'])
            shutil.copyfile(job/'calibration.json',reports/r['configuration']/'calibration.json')
        if len(done)!=3 or len({x['initial_state_digest'] for x in done})!=1:raise ValueError('Incomplete/unmatched A/B/C block')
        rc=0
    except BaseException as e:
        entry.update(status='FAILED',error_type=type(e).__name__,message=str(e))
        if job is not None and (job/'reports/worker_failure.json').is_file():
            root=read(job/'reports/worker_failure.json');entry['root_error_type']=root.get('exception','UNKNOWN')
            write(reports/'worker_failure.json',root)
        # Carry a safe, bounded diagnostic even when the worker failed before
        # its success reports could be copied. Do not export full local logs.
        if job is not None and (job/'job.json').is_file():
            try:
                from .failure_evidence import collect_failure_capsule
                capsule=collect_failure_capsule(job,reports)
                if capsule.get('server_failure_found'):
                    entry['root_error_type']=capsule['server_exception']
            except Exception as diagnostic_error:
                write(reports/'failure_capsule_error.json',{
                    'status':'DIAGNOSTIC_INCOMPLETE','exception':type(diagnostic_error).__name__,
                    'original_failure_preserved':True,'local_values_exported':False})
        (reports/'failure_trace.txt').write_text(traceback.format_exc())
        print('STOP [main_block]: '+str(e),flush=True)
    finally:
        try:
            if before is not None and environment(prior,reports/'environment_after.json')!=before:raise ValueError('Accepted environment changed')
            if original is not None and original_snapshot(reports/'original_environment_after.json')!=original:raise ValueError('Original environment changed')
            verify_campaign(campaign)
        except BaseException as e:rc=1;entry.update(status='FAILED',postcheck_error=str(e),root_error_type=type(e).__name__)
        if rc==0:
            entry['status']='COMPLETE';progress['completed_blocks'].append(block)
        entry['completed_cells']=done;progress['attempts'][block]=entry;write(campaign/'progress.json',progress)
        write(reports/'block_status.json',{**entry,'test_data_opened':False,'privacy_claim_validated':False})
        publish(reports,campaign,'Main_Block_Report_'+block+f'_attempt{attempt}')
    if rc==0 and len(progress['completed_blocks'])==10:seal_selected_models(campaign)
    return rc


def seal_selected_models(campaign:Path):
    progress=read(campaign/'progress.json')
    if progress['completed_blocks']!=ordered_blocks():raise ValueError('All ten planned blocks must finish first')
    models=[]
    for row in plan():
        attempt=progress['attempts'][row['block_id']]
        if attempt['status']!='COMPLETE':raise ValueError('Incomplete primary block')
        job=campaign/attempt['path']/row['configuration'];s=validate_finished_job(job)
        for key,value in [('configuration',row['configuration']),('condition',row['condition']),('seed',int(row['training_seed']))]:
            if s[key]!=value:raise ValueError('Main cell identity differs from run matrix')
        cp=job/'checkpoints'/row['condition']/'selected_state.pt'
        models.append({**s,'job':job.relative_to(campaign).as_posix(),
                       'checkpoint':cp.relative_to(campaign).as_posix(),
                       'calibration':(job/'calibration.json').relative_to(campaign).as_posix()})
    write(campaign/'selected_models_manifest.json',{'models':models,'test_data_authorised':False,
        'campaign_manifest_sha256':sha(campaign/'campaign_manifest.json'),
        'package_manifest_sha256':package_integrity()},exclusive=True)
    from main_study.costs import campaign_summary
    report=campaign/'main_completion_reports';report.mkdir(exist_ok=False)
    shutil.copyfile(campaign/'selected_models_manifest.json',report/'selected_models_manifest.json')
    write(report/'system_cost_analysis.json',campaign_summary(campaign,models))
    write(report/'completion_status.json',{'status':'ALL_30_MODELS_CALIBRATED_TESTS_NOT_AUTHORISED',
        'selected_models_manifest_sha256':sha(campaign/'selected_models_manifest.json'),
        'test_data_opened':False,'main_training_complete':True,'privacy_certification':False})
    publish(report,campaign,'Main_Completion_Report')
    print('ALL_MAIN_MODELS_AND_VALIDATION_CALIBRATIONS_SEALED_TESTS_STILL_CLOSED',flush=True)
