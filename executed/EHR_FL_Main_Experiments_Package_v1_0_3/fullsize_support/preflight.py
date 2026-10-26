"""Pinned CLI/API checks; no training or installation."""
import json,sys,os,subprocess,importlib.metadata
from pathlib import Path
from central_smoke.identity import write_json
from ehr_fedavg.host import local_environment,inspect_flower_command_path,FLOWER_COMMANDS
def preflight_flower(cfg,out):
    env = local_environment(out/'probe_home')
    pathcheck = inspect_flower_command_path(env)
    write_json(out/'reports/flower_command_path.json', pathcheck)
    if pathcheck['status'] != 'PASS':
        raise RuntimeError('; '.join(pathcheck['errors']) + '; no automatic installation or system-command fallback')
    paths={name:Path(os.path.abspath(sys.executable)).parent/name for name in FLOWER_COMMANDS};helptexts={}
    for name,p in paths.items():
        # A bare-name check reproduces the lookup performed by SuperLink itself.
        # --help does not start a server or a training application.
        command = name if name == 'flower-superexec' else str(p)
        r=subprocess.run([command,'--help'],capture_output=True,text=True,timeout=45,env=env)
        (out/'reports'/(name+'_help.txt')).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('CLI help failed: '+name)
        helptexts[name]=r.stdout+r.stderr
    pathcheck['command_help_executed'] = True
    pathcheck['all_four_help_checks_passed'] = True
    pathcheck['superexec_bare_name_help_passed'] = True
    write_json(out/'reports/flower_command_path.json', pathcheck)
    print('FLOWER_SUBPROCESS_PATH_CHECK_PASS', flush=True)
    sl=['--insecure','--serverappio-api-address','--fleet-api-address','--control-api-address','--storage-dir','--isolation']
    sn=['--insecure','--superlink','--clientappio-api-address','--node-config','--isolation']
    for key,flags in [('flower-superlink',sl),('flower-supernode',sn)]:
        if any(x not in helptexts[key] for x in flags):raise RuntimeError('Pinned CLI flags missing: '+key)
    from ehr_fedavg.server_app import app as serverapp,BoundedGrid
    from ehr_fedavg.client_app import app as clientapp
    from flwr.common import ndarrays_to_parameters,parameters_to_ndarrays,FitRes,Status,Code
    from flwr.server.strategy import FedAvg
    from types import SimpleNamespace
    import numpy as np
    from ehr_fedavg.math_checks import check_aggregate
    spec=[{'name':'toy','shape':[3],'dtype':'float32'}]
    updates=[[np.array([1,2,-4],np.float32)],[np.array([8,-5,3],np.float32)],[np.array([-2,10,1],np.float32)]];weights=[1,3,7]
    pairs=[(SimpleNamespace(cid=str(i)),FitRes(Status(Code.OK,''),ndarrays_to_parameters(a),n,{})) for i,(a,n) in enumerate(zip(updates,weights))]
    params,_=FedAvg(inplace=False,accept_failures=False,fit_metrics_aggregation_fn=lambda _: {}).aggregate_fit(1,pairs,[])
    check=check_aggregate(parameters_to_ndarrays(params),updates,weights,spec,cfg['aggregation_atol'],cfg['aggregation_rtol'])
    write_json(out/'reports/flower_api_preflight.json',{'status':'PASS','version':importlib.metadata.version('flwr'),'clientapp_created':True,'serverapp_created':True,'unequal_weight_numeric_test':check,'deployment_executed_in_this_check':False})
    return {k:str(v) for k,v in paths.items()}

