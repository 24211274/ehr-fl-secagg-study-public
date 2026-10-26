"""SV-07 bounded public-canary export checks; no benchmark inputs or model loads."""
from pathlib import Path
import argparse, contextlib, datetime, hashlib, io, json, os, subprocess, sys, traceback

ROOT=Path('/home/mai/ehr_me06_reproduction_20261011_v1')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

CASES=[
 ('SV07-01','Scanner positive control detects public synthetic canaries in a deliberately unsafe export.'),
 ('SV07-02','Actual native RecordDict byte counter exports numeric counts only.'),
 ('SV07-03','Actual client observer success export omits payload canaries and retains permitted counts/metadata.'),
 ('SV07-04','Actual client observer failure export records exception type but omits exception message canary.'),
 ('SV07-05','Actual server grid observer/report omit request/reply payload canaries.'),
 ('SV07-06','Actual approved local timing spans export names, durations and completion only.'),
 ('SV07-07','Actual failure capsule redacts unknown exception messages and ignores extra record fields.'),
 ('SV07-08','Actual failure capsule retains allowlisted safe message and frame/round metadata, omits source/locals/losses.'),
 ('SV07-09','Actual failure capsule replaces syntactically invalid exception class with UNKNOWN.'),
 ('SV07-10','Actual failure capsule ignores symlinked result/trace/history fixtures.'),
 ('SV07-11','Actual failure capsule rejects unsupported job identity without emitting a capsule.'),
 ('SV07-12','All selected public payload-safe output artifacts are scanned; every canary is absent.'),
 ('SV07-13','Boundary observation: permitted request ID/frame/function/class metadata are not arbitrary-value redaction.'),
]

def seal():
    inv=json.loads((ROOT/'input_inventory.json').read_text())
    entries=[]
    for section in ['source','preprocessing_source']:
        for row in inv[section]:
            p=ROOT/section/row['path']
            h=digest(p)
            entries.append({'section':section,'relative_path':row['path'],'bytes':p.stat().st_size,'sha256':h,'matches_sealed_input':h==row['sha256']})
    protected=[ROOT/'full_requirements.lock',ROOT/'.venv/lib/python3.12/site-packages/flwr/common/secure_aggregation/secaggplus_utils.py',ROOT/'build/flwr-1.22.0+mai.prg7570.1-py3-none-any.whl',ROOT/'build/flwr-1.22.0-py3-none-any.whl']
    records=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in protected]
    return {'at_utc':now(),'source_files':entries,'protected_runtime_files':records,'source_failures':sum(not r['matches_sealed_input'] for r in entries)}

def launch():
    HERE.mkdir(parents=True,exist_ok=True)
    n=1
    while (HERE/f'attempt_{n:02d}').exists(): n+=1
    attempt=HERE/f'attempt_{n:02d}';attempt.mkdir()
    contract={'group':'SV-07','created_before_execution_utc':now(),'scope':'Public synthetic payload/error canaries through actual observer and failure capsule; no benchmark rows, model loads, training, actual secrets or network traffic.','cases':[{'id':a,'acceptance':b} for a,b in CASES], 'required_security_acceptance':'Cases01-12 pass; payload-safe outputs contain none of the public canaries; no unexpected exception; source/runtime seals unchanged. Case13 is a diagnostic boundary observation, not proof of arbitrary metadata redaction.','allowed_public_fields':['request_id','node_id','client','condition','round','stage','phase','message_type','numeric record byte counts','clock durations','completed','error_type','measurement_scope','traceback file/line/function metadata','exception class','allowlisted safe exception message','file SHA256','last recorded round'],'exclusions':['raw fixture inputs','positive-control unsafe export','script/contract containing public test markers','metadata-boundary observation output','full private logs','full trusted-host/process memory','network transport'],'code_sha256':digest(__file__),'original_source_mutation_authorised':False,'new_training_authorised':False}
    write(attempt/'contract.json',contract)
    (attempt/'contract.sha256').write_text(digest(attempt/'contract.json')+'\n')
    before=seal();write(attempt/'original_seal_before.json',before)
    cmd=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--child',str(attempt)]
    start=now();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    proc=subprocess.run(cmd,cwd=attempt,env=env,capture_output=True,text=True)
    end=now();(attempt/'stdout.log').write_text(proc.stdout);(attempt/'stderr.log').write_text(proc.stderr)
    after=seal();write(attempt/'original_seal_after.json',after)
    same=before['source_files']==after['source_files'] and before['protected_runtime_files']==after['protected_runtime_files']
    receipt={'group':'SV-07','attempt':n,'command':cmd,'cwd':str(attempt),'start_utc':start,'end_utc':end,'exit_code':proc.returncode,'script_sha256':digest(__file__),'contract_sha256':digest(attempt/'contract.json'),'source_and_protected_runtime_unchanged':same,'source_seal_checks':len(before['source_files']),'source_input_hash_failures':before['source_failures'],'stdout_sha256':digest(attempt/'stdout.log'),'stderr_sha256':digest(attempt/'stderr.log'),'status':'PASS' if proc.returncode==0 and same and before['source_failures']==0 else 'FAIL'}
    write(attempt/'execution_receipt.json',receipt)
    write(HERE/'Latest_Attempt.json',{'attempt':str(attempt),'receipt':receipt})
    print(json.dumps(receipt,indent=2));return 0 if receipt['status']=='PASS' else 1

def child(attempt):
    # Any original benchmark/campaign/array/model read is refused in the test process.
    denied=[]
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(ROOT):
                forbidden=any(p.is_relative_to(ROOT/x) for x in ['assets','campaign','preprocessing','reference']) or p.suffix in ['.pt','.pth','.ckpt','.npy','.npz']
                if forbidden:
                    denied.append(str(p));raise PermissionError('SV07 benchmark/model access forbidden')
    sys.addaudithook(guard)
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(SOURCE),str(SOURCE/'app'),str(SOURCE/'tests')]
    import types
    from unittest.mock import patch
    import numpy as np
    from flwr.common import Message,RecordDict,ConfigRecord,MetricRecord,Context,ndarrays_to_parameters
    from flwr.common.recorddict_compat import parameters_to_arrayrecord
    from flower_message_fixture import received_message
    import mai_measurement as mm
    from main_study.failure_evidence import collect_failure_capsule
    outputs=attempt/'payload_safe_outputs';outputs.mkdir()
    fixtures=attempt/'public_fixture_inputs';fixtures.mkdir()
    controls=attempt/'controls';controls.mkdir()
    observations=attempt/'metadata_boundary_observations';observations.mkdir()
    canaries={x:f'PUBLIC_SV07_{x.upper()}_CANARY_20261011_ONLY' for x in ['payload','seed','mask','share','key','array','error','local','loss']}
    markers=list(canaries.values())
    scope={'rounds':30,'conditions':['reference','noniid'],'stage':'MAIN_CONTROLLED_COMPARISON','fixed_config_sha256':'0'*64}
    results=[]
    def scan(paths,tokens=markers):
        hits=[]
        for p in paths:
            raw=p.read_bytes()
            for token in tokens:
                if token.encode() in raw: hits.append({'file':str(p.relative_to(attempt)),'public_canary_category':next((k for k,v in canaries.items() if v==token),'metadata')})
        return hits
    def message(ident='public-sv07-request',with_payload=True):
        content=RecordDict({'fitins.config':ConfigRecord({'server_round':1})})
        if with_payload: content['fixture_payload']=ConfigRecord({k:v for k,v in canaries.items()})
        return received_message(Message(content=content,dst_node_id=11,message_type='train',group_id='1',ttl=60),message_id=ident)
    def reply(m): return Message(content=RecordDict({'fixture_masked':ConfigRecord({'masked_parameters':[canaries['mask'].encode()],'private_seed':canaries['seed'],'secret_share':canaries['share']})}),reply_to=m)
    def ctx(directory): return Context(run_id=1,node_id=11,node_config={'client-id':'C1'},state=RecordDict(),run_config={'runtime-root':str(directory),'condition':'reference','num-server-rounds':30})
    def makejob(name,condition='reference',configuration='C'):
        d=fixtures/name;d.mkdir();write(d/'job.json',{'condition':condition,'configuration':configuration});return d
    def do(caseid,fn,observation=False):
        start=now();record={'id':caseid,'start_utc':start,'acceptance':dict(CASES)[caseid]}
        try:
            evidence=fn();record.update(status='OBSERVED_BOUNDARY' if observation else 'PASS',evidence=evidence or {})
        except BaseException as exc:
            record.update(status='HARNESS_OR_TEST_FAILURE',error_type=type(exc).__name__,error_message=str(exc),traceback=traceback.format_exc())
        record['end_utc']=now();results.append(record)
        write(attempt/'case_results'/f'{caseid}.json',record)
        print(f'{caseid}: {record["status"]}')

    def c01():
        p=controls/'deliberately_unsafe_public_canaries.json';write(p,canaries)
        hits=scan([p]);assert len(hits)==len(canaries)
        return {'detected_categories':len(hits),'expected_categories':len(canaries),'control_export_is_not_publishable':True}
    do('SV07-01',c01)
    def c02():
        data=np.frombuffer(canaries['array'].encode(),dtype=np.uint8).copy()
        content=RecordDict({'fixture_config':ConfigRecord(canaries),'fixture_metric':MetricRecord({'public_count':7}),'fixture_array':parameters_to_arrayrecord(ndarrays_to_parameters([data]),True)})
        row=mm.record_bytes(content);p=outputs/'record_counts.json';write(p,row)
        assert row['total_bytes']==row['config_bytes']+row['metric_bytes']+row['array_bytes']
        assert all(type(v)is int and v>=0 for v in row.values()) and not scan([p])
        return {'counts':row,'native_records_used':True}
    do('SV07-02',c02)
    def c03():
        d=outputs/'client_success';m=message()
        with patch.object(mm,'measurement_scope',return_value=scope): r=mm.timing_boundary_mod(m,ctx(d),lambda a,b:reply(a))
        paths=list(d.rglob('*.json'));assert len(paths)==1 and not scan(paths)
        row=json.loads(paths[0].read_text());assert row['status']=='PASS' and row['record_values_written']is False and row['wire_bytes_measured']is False
        assert row['request_id']=='public-sv07-request' and row['incoming_record_bytes']['total_bytes']>0 and row['outgoing_record_bytes']['total_bytes']>0
        return {'export_fields':sorted(row),'counts':{'incoming':row['incoming_record_bytes'],'outgoing':row['outgoing_record_bytes']},'message_payload_remains_in_memory':r.has_content()}
    do('SV07-03',c03)
    def c04():
        d=outputs/'client_failure'
        def fail(a,b): raise RuntimeError(canaries['error'])
        try:
            with patch.object(mm,'measurement_scope',return_value=scope): mm.timing_boundary_mod(message('public-sv07-failure'),ctx(d),fail)
        except RuntimeError: pass
        else: raise AssertionError('Expected exception not propagated')
        paths=list(d.rglob('*.json'));assert len(paths)==1 and not scan(paths)
        row=json.loads(paths[0].read_text());assert row['status']=='FAIL' and row['error_type']=='RuntimeError' and row['outgoing_record_bytes']is None
        return {'error_type_exported':row['error_type'],'exception_message_exported':False,'callback_exception_propagated':True}
    do('SV07-04',c04)
    def c05():
        inner=types.SimpleNamespace(send_and_receive=lambda sent,timeout=None:[reply(m) for m in sent])
        mm.reset_server()
        with patch.object(mm,'measurement_scope',return_value=scope): grid=mm.ObservedRawGrid(inner)
        grid.send_and_receive([message('public-sv07-server')],timeout=1)
        p=outputs/'server_observer.json';write(p,mm.server_report(grid));assert not scan([p])
        row=json.loads(p.read_text());assert row['record_values_written']is False and row['wire_bytes_measured']is False and len(row['exchanges'])==1
        return {'report_fields':sorted(row),'exchange_count':1,'message_fields':sorted(row['exchanges'][0]['messages'][0])}
    do('SV07-05',c05)
    def c06():
        d=outputs/'client_span'
        def callback(m,c):
            with mm.client_span('local_fit_with_checks'): pass
            return reply(m)
        with patch.object(mm,'measurement_scope',return_value=scope): mm.timing_boundary_mod(message('public-sv07-span'),ctx(d),callback)
        paths=list(d.rglob('*.json'));assert len(paths)==1 and not scan(paths)
        spans=json.loads(paths[0].read_text())['local_spans'];assert len(spans)==1 and set(spans[0])=={'name','wall_ns','cpu_ns','completed'} and spans[0]['completed']is True
        return {'span_fields':sorted(spans[0]),'approved_name':spans[0]['name'],'numeric_timers':all(type(spans[0][k])is int and spans[0][k]>=0 for k in ['wall_ns','cpu_ns'])}
    do('SV07-06',c06)
    def c07():
        d=makejob('unknown_error');write(d/'server_outputs/reference/server_result.json',{'status':'FAIL','exception':'RuntimeError','message':canaries['error'],'keys':canaries['key'],'private_seed':canaries['seed'],'masked_parameters':[canaries['mask']]})
        cap=collect_failure_capsule(d,outputs/'capsule_unknown');p=outputs/'capsule_unknown/server_failure_capsule.json';assert not scan([p]) and cap['server_exception']=='RuntimeError' and cap['message']=='[redacted; inspect retained local log]'
        return {'capsule_fields':sorted(cap),'unknown_message_redacted':True,'extra_fields_exported':False}
    do('SV07-07',c07)
    def c08():
        d=makejob('safe_error');safe='Message outside frozen measurement scope: round=31; phase=fit; max_round=30'
        write(d/'server_outputs/reference/server_result.json',{'status':'FAIL','exception':'ValueError','message':safe})
        (d/'error_reference_LOCAL_ONLY.txt').write_text('Traceback (most recent call last):\n  File "public_fixture.py", line 12, in public_callback\n    private_seed = '+canaries['local']+'\nValueError: '+canaries['error']+'\n')
        write(d/'server_outputs/reference/history.json',{'history':[{'round':r,'loss':canaries['loss']} for r in [0,1,2]]})
        cap=collect_failure_capsule(d,outputs/'capsule_safe');p=outputs/'capsule_safe/server_failure_capsule.json';assert not scan([p]) and cap['message']==safe and cap['last_recorded_evaluation_round']==2 and cap['frames']==[{'file':'public_fixture.py','line':12,'function':'public_callback'}]
        assert cap['local_variables_exported']is False and cap['full_logs_exported']is False
        return {'safe_message_preserved':True,'public_frames_preserved':True,'last_round':2,'losses_and_locals_exported':False}
    do('SV07-08',c08)
    def c09():
        d=makejob('invalid_class');write(d/'server_outputs/reference/server_result.json',{'status':'FAIL','exception':'invalid-'+canaries['key'],'message':canaries['error']})
        cap=collect_failure_capsule(d,outputs/'capsule_class');assert cap['server_exception']=='UNKNOWN' and not scan([outputs/'capsule_class/server_failure_capsule.json'])
        return {'invalid_class_replaced_with':'UNKNOWN'}
    do('SV07-09',c09)
    def c10():
        d=makejob('symlinked');targets=fixtures/'symlink_targets';targets.mkdir()
        write(targets/'server_result.json',{'status':'FAIL','exception':'RuntimeError','message':canaries['error']});(targets/'trace.txt').write_text(canaries['local']);write(targets/'history.json',{'history':[{'round':0,'loss':canaries['loss']}]})
        server=d/'server_outputs/reference';server.mkdir(parents=True)
        (server/'server_result.json').symlink_to(targets/'server_result.json');(server/'history.json').symlink_to(targets/'history.json');(d/'error_reference_LOCAL_ONLY.txt').symlink_to(targets/'trace.txt')
        cap=collect_failure_capsule(d,outputs/'capsule_symlink');assert cap['server_failure_found']is False and 'frames'not in cap and 'last_recorded_evaluation_round'not in cap and not scan([outputs/'capsule_symlink/server_failure_capsule.json'])
        return {'symlinked_fixture_fields_ignored':True,'links_stay_within_public_fixture_tree':True}
    do('SV07-10',c10)
    def c11():
        d=makejob('bad_identity',configuration='INVALID');dest=outputs/'capsule_invalid'
        try:collect_failure_capsule(d,dest)
        except ValueError:pass
        else:raise AssertionError('Unsupported job identity accepted')
        assert not (dest/'server_failure_capsule.json').exists()
        return {'identity_rejected':True,'capsule_emitted':False}
    do('SV07-11',c11)
    def c12():
        paths=sorted(outputs.rglob('*.json'));hits=scan(paths);assert len(paths)>=9 and not hits
        return {'selected_publishable_json_files':len(paths),'public_canary_categories':len(markers),'canary_matches':0,'scanned_only_runtime_outputs':True,'source_scripts_and_fixture_inputs_excluded':True,'files':[{'path':str(p.relative_to(attempt)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]}
    do('SV07-12',c12)
    def c13():
        # Deliberately adversarial public metadata, separated from payload-safe outputs.
        marker='PUBLIC_SV07_METADATA_BOUNDARY_CANARY_ONLY'
        d=makejob('metadata_boundary');write(d/'server_outputs/reference/server_result.json',{'status':'FAIL','exception':marker,'message':canaries['error']})
        (d/'error_reference_LOCAL_ONLY.txt').write_text(f'  File "{marker}.py", line 12, in {marker}\n')
        cap=collect_failure_capsule(d,observations/'capsule');assert cap['server_exception']==marker and marker in cap['frames'][0]['file'] and cap['frames'][0]['function']==marker
        client=observations/'observer'
        with patch.object(mm,'measurement_scope',return_value=scope):mm.timing_boundary_mod(message(marker,False),ctx(client),lambda a,b:reply(a))
        paths=sorted(observations.rglob('*.json'));hits=scan(paths,[marker]);assert len(hits)==2
        assert not scan(paths,markers)
        return {'permitted_metadata_copied':['request_id','server_exception','traceback_file','traceback_function'],'public_metadata_canary_output_files':len(hits),'payload_canaries_found':0,'interpretation':'Field allowlisting/redaction of payloads and exception messages does not sanitise arbitrary metadata values. Trusted public metadata assumption required; review metadata before external sharing. This is not evidence of leakage from actual EHR data or protocol secrets.'}
    do('SV07-13',c13,True)
    passed=sum(r['status']=='PASS' for r in results);failed=[r['id'] for r in results if r['status']=='HARNESS_OR_TEST_FAILURE']
    summary={'group':'SV-07','status':'PASS_WITH_METADATA_BOUNDARY' if not failed and passed==12 else 'FAIL','cases_passed':passed,'boundary_observations':sum(r['status']=='OBSERVED_BOUNDARY' for r in results),'failed_cases':failed,'cases':results,'original_benchmark_or_model_open_attempts_denied':len(denied),'benchmark_data_read':False,'model_loaded':False,'training_performed':False,'network_experiment':False,'actual_secret_values_used':False,'public_synthetic_canaries_only':True,'privacy_certification':False,'limits':['Actual native records with controlled callbacks and in-memory transport; not a full secure training run.','File/frame/function/exception-class/request-ID metadata remain allowed and are not universal arbitrary-value sanitisation.','Only selected output JSON artifacts scanned; full private logs/source/test scripts/trusted host memory excluded.','Zero literal canary matches does not prove absence of all possible leakage or side channels.']}
    write(attempt/'SV07_Results.json',summary);return 0 if not failed and passed==12 else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',type=Path);args=p.parse_args()
    sys.exit(child(args.child) if args.child else launch())
