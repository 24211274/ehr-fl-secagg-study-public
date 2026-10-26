"""Inspect a native Flower 1.22 FAB and its unpacked runtime before services start.

The local tests may use an explicitly labelled extension-filtered test ZIP.
Only preflight_native_fab uses Flower's installed builder and can emit the
FLOWER_APP_BUNDLE_CHECK_PASS marker. No servers, rounds, training or downloads
are performed by this gate.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

from central_smoke.identity import write_json
from mai_fab_policy import VERSION as APPROVED_FLOWER_VERSION, require_fab_runtime
from ehr_fedavg.resources import verify_source_resources, resource_sha256

ALLOWED_SOURCE_SUFFIXES = {'.py', '.toml', '.md'}
MAX_BYTES = 10 * 1024 * 1024


def source_files(app_directory: Path) -> dict[str, bytes]:
    """This project has no app-level .gitignore overrides or additional assets."""
    root=Path(app_directory).resolve()
    files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Symlinks not permitted in app staging')
        if p.is_file() and '__pycache__' not in p.relative_to(root).parts and p.suffix in ALLOWED_SOURCE_SUFFIXES:
            if p.name!='pyproject.toml':files[p.relative_to(root).as_posix()]=p.read_bytes()
    if 'ehr_fedavg/_embedded_resources.py' not in files:
        raise ValueError('Missing required bundled-resource carrier')
    for name in ['ehr_fedavg/resources.py','ehr_fedavg/task.py','ehr_fedavg/server_app.py','ehr_fedavg/client_app.py','pilot_core/model.py']:
        if name not in files:raise ValueError('Missing application source: '+name)
    return files


def verify_fab_contents(fab: bytes, app_directory: Path) -> dict:
    """Validate exact code payload and Flower's .info/CONTENT hash/size ledger."""
    if not isinstance(fab,bytes) or len(fab)>MAX_BYTES:raise ValueError('Invalid or oversized FAB')
    expected=source_files(app_directory)
    with zipfile.ZipFile(io.BytesIO(fab)) as z:
        members=z.namelist()
        if len(members)!=len(set(members)):raise ValueError('Duplicate FAB paths')
        for name in members:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix()!=name:
                raise ValueError('Unsafe FAB member: '+name)
        required=set(expected)|{'pyproject.toml','.info/CONTENT'}
        if set(members)!=required:
            raise ValueError('FAB member set mismatch; missing='+str(sorted(required-set(members)))+' extra='+str(sorted(set(members)-required)))
        manifest={}
        for line in z.read('.info/CONTENT').decode('utf-8').splitlines():
            name,digest,bits=line.split(',')
            if name in manifest:raise ValueError('Duplicate FAB CONTENT path')
            manifest[name]=(digest,int(bits))
        if set(manifest)!=set(expected):raise ValueError('FAB CONTENT manifest set mismatch')
        for name,raw in expected.items():
            included=z.read(name)
            if raw!=included:raise ValueError('FAB file differs from staged source: '+name)
            if manifest[name]!=(hashlib.sha256(raw).hexdigest(),len(raw)*8):raise ValueError('FAB CONTENT digest/bit-size mismatch: '+name)
        # Native build_fab strips federation connection settings, not app config.
        import tomllib
        source_conf=tomllib.loads((Path(app_directory)/'pyproject.toml').read_text())
        source_conf.get('tool',{}).get('flwr',{}).pop('federations',None)
        archive_conf=tomllib.loads(z.read('pyproject.toml').decode('utf-8'))
        if archive_conf!=source_conf:raise ValueError('FAB pyproject changed app settings')
        if sum(x.file_size for x in z.infolist())>MAX_BYTES:raise ValueError('Oversized extracted FAB')
    return {'status':'PASS','members_checked':len(members),'python_sources_checked':sum(x.endswith('.py') for x in expected),
        'fab_sha256':hashlib.sha256(fab).hexdigest(),'fab_bytes':len(fab),
        'original_json_files_absent_as_expected':not any(x.endswith('.json') for x in members),
        'embedded_config_carrier_present':True,'dataset_or_weight_files_bundled':False,
        'members':sorted(members),'content_ledger_sizes_use_bits':True}


PROBE_CODE = r'''
import sys,json,hashlib
from pathlib import Path
unpacked=Path(sys.argv[1]).resolve()
runtime_root=sys.argv[2]
require_flower=sys.argv[3]=='1'
output=Path(sys.argv[4])
# -I ignores PYTHONPATH/current directory. Only the unpacked FAB supplies our code.
sys.path.insert(0,str(unpacked))
result={'status':'FAIL','native_flower_callbacks_requested':require_flower,
        'services_started':False,'flower_rounds_executed':False,
        'optimiser_steps':0,'test_data_opened':False}
try:
    import ehr_fedavg.resources as res
    import ehr_fedavg.task as task
    import pilot_core.model as model_module
    from central_smoke.identity import sha256_file
    for module in [res,task,model_module]:
        if not Path(module.__file__).resolve().is_relative_to(unpacked):
            raise ValueError('Fallback import outside unpacked FAB: '+module.__name__)
    for name in res.NAMES:
        if (unpacked/'ehr_fedavg'/name).exists():raise ValueError('Unexpected raw JSON in FAB probe')
    conf=task.cfg();schema=res.read_document('encoding_schema.json')
    if conf['schema_sha256']!=res.resource_sha256('encoding_schema.json'):raise ValueError('Schema identity link')
    model=task.fresh_global()
    digest=model_module.state_digest(model)
    result.update(input_dim=model.input_dim,model_parameters=sum(p.numel() for p in model.parameters()),
        initial_state_digest=digest,config_sha256=res.resource_sha256('fixed_config.json'),
        schema_sha256=res.resource_sha256('encoding_schema.json'),
        resources_loaded_from_supported_bundle_carrier=True,original_json_fallback_used=False,
        imported_code_from_unpacked_fab=True,flower_callbacks_created=False,clients_checked=[])
    if require_flower:
        from ehr_secagg.client_app import app as ca
        from ehr_secagg.server_app import app as sa
        from flwr.client import ClientApp
        from flwr.server import ServerApp
        if not isinstance(ca,ClientApp) or not isinstance(sa,ServerApp):raise ValueError('Flower callbacks have wrong types')
        from ehr_secagg.workflow import StrictSecAggPlusWorkflow
        from ehr_secagg.settings import scfg
        w=StrictSecAggPlusWorkflow()
        result['flower_callbacks_created']=True
        from mai_runtime_identity import binding_identity
        result['candidate_implementation_identity']=binding_identity()
        result['secagg_workflow_created']=True
        result['security_claim_gate']=scfg()['security_claim_gate']
        result['callbacks']='ehr_secagg'
    if runtime_root!='-':
        runtime=json.loads((Path(runtime_root)/'runtime.json').read_text())
        if runtime['fixed_config_sha256']!=result['config_sha256'] or runtime['expected_initial_digest']!=digest:
            raise ValueError('Runtime identity does not match installed bundle resources/model')
        for client in conf['clients']:
            # Loads only that client's staged routine training/validation arrays.
            data,samples,draws,_=task.load_client(runtime_root,client)
            if len(samples['training'])!=conf['training_sessions_per_client'][client] or len(samples['validation'])!=conf['validation_sessions_per_client'][client]:raise ValueError('Wrong local sample counts')
            if set(data['training'])&set(data['validation']):raise ValueError('Local session overlap')
            result['clients_checked'].append({'client':client,'training':len(samples['training']),'validation':len(samples['validation']),
                                             'local_loader_passed':True})
    result['status']='PASS'
except BaseException as exc:
    result.update(error_type=type(exc).__name__,message=str(exc))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    raise
else:
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
'''


def probe_unpacked_fab(fab: bytes, app_directory: Path, work: Path, runtime_root: Path|None,
                       env: dict[str,str], *, require_flower: bool=True) -> dict:
    """Validates first, extracts into a new folder, imports in an isolated process."""
    contents=verify_fab_contents(fab,app_directory)
    work=Path(work);work.mkdir(parents=True,exist_ok=False)
    extracted=work/'unpacked';extracted.mkdir()
    # Names, bytes and sizes have been checked against the known local app source.
    with zipfile.ZipFile(io.BytesIO(fab)) as z:z.extractall(extracted)
    report=work/'probe_result.json'
    child_env=dict(env);child_env.pop('PYTHONPATH',None)
    command=[sys.executable,'-I','-B','-c',PROBE_CODE,str(extracted),str(runtime_root) if runtime_root else '-',
             '1' if require_flower else '0',str(report)]
    run=subprocess.run(command,cwd=work,env=child_env,capture_output=True,text=True,timeout=180)
    (work/'probe.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    result=json.loads(report.read_text()) if report.is_file() else {'status':'FAIL','message':'Isolated import produced no report'}
    result['exit_code']=run.returncode
    if run.returncode or result.get('status')!='PASS':
        raise RuntimeError('Unpacked FAB probe failed: '+str(result.get('message','see bundle probe log')))
    if result['config_sha256']!=resource_sha256('fixed_config.json') or result['schema_sha256']!=resource_sha256('encoding_schema.json'):
        raise ValueError('Unpacked FAB resource digests differ')
    return {'contents':contents,'isolated_import':result}


def preflight_native_fab(app_directory: Path, runtime_root: Path, condition: str,
                         report_path: Path, env: dict[str,str]) -> dict:
    """Build with installed Flower 1.22 and prove resources survive relocation."""
    audit={'status':'FAIL','condition':condition,'services_started_by_this_gate':False,
           'builder':'flwr.cli.build.build_fab','native_flower_build_executed':False,
           'expected_flower_version':APPROVED_FLOWER_VERSION,
           'version_policy':'exact_approved_local_build_with_source_and_binding_checks',
           'training_or_validation_metrics_computed':False,'test_data_opened':False}
    probe_work=Path(runtime_root)/'bundle_preflight'/condition
    try:
        version=importlib.metadata.version('flwr')
        audit['flower_version']=version
        audit['candidate_implementation_identity']=require_fab_runtime(version)
        from flwr.cli.build import build_fab
        from flwr.common.constant import FAB_ALLOWED_EXTENSIONS
        if set(FAB_ALLOWED_EXTENSIONS)!=ALLOWED_SOURCE_SUFFIXES:raise RuntimeError('Pinned FAB extension policy differs')
        audit.update(flower_version=version,allowed_extensions=sorted(FAB_ALLOWED_EXTENSIONS),
                     source_resource_identity=verify_source_resources(app_directory))
        data,fab_hash,metadata=build_fab(Path(app_directory))
        audit['native_flower_build_executed']=True
        if hashlib.sha256(data).hexdigest()!=fab_hash:raise ValueError('Native FAB digest mismatch')
        built_again,hash_again,_=build_fab(Path(app_directory))
        if data!=built_again or fab_hash!=hash_again:raise ValueError('Native FAB rebuild is not deterministic')
        check=probe_unpacked_fab(data,app_directory,probe_work,runtime_root,env)
        archive=probe_work/'verified_app.fab';archive.write_bytes(data)
        audit.update(check,native_rebuild_byte_identical=True,status='PASS',
                     fab_stored_locally=True,flwr_run_will_rebuild_same_staged_sources=True)
        print('FLOWER_APP_BUNDLE_CHECK_PASS '+condition,flush=True)
        return audit
    except BaseException as exc:
        audit.update(error_type=type(exc).__name__,message=str(exc))
        result_path=probe_work/'probe_result.json'
        if result_path.is_file():audit['isolated_import']=json.loads(result_path.read_text())
        raise
    finally:
        write_json(Path(report_path),audit)
