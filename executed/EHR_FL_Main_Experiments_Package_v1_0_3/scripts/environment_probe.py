#!/usr/bin/env python3
"""Read-only probe of the EXISTING accepted complete candidate environment."""
import sys,os,json,platform,importlib.metadata as md
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'app'),str(ROOT/'validation_support')]
from central_smoke.identity import sha256_file as sha,write_json
from fullsize_support.integrity import read
from mai_runtime_identity import binding_identity,VERSION
from prg_validation.runtime_checks import installed_wheel_check
from prg_validation.snapshot import canon

def probe(prior):
    prior=Path(prior).resolve();expected=read(ROOT/'reference/accepted_bc/full_environment_after.json')
    env=prior/'.venv_full_candidate'
    if platform.system()!='Linux' or sys.version_info[:2]!=(3,12) or os.geteuid()==0 or Path(sys.prefix).resolve()!=env:raise ValueError('Use accepted non-root Linux Python 3.12 full candidate venv in place; no copies/installations')
    packages={canon(d.metadata['Name']):d.version for d in md.distributions()}
    if packages!=expected['packages']:raise ValueError('Installed package versions changed since accepted B/C run')
    items=read(prior/'results/full_wheel_inventory.json');record=items['flwr'];wheel=prior/'wheels'/record['filename']
    policy=read(ROOT/'config/integration_policy.json')
    if sha(wheel)!=policy['candidate_wheel_sha256']:raise ValueError('Accepted PRG wheel changed')
    ident=binding_identity()
    if ident!=expected['identity']:raise ValueError('PRG runtime differs from accepted B/C report')
    installed=installed_wheel_check(wheel,policy['candidate_wheel_sha256'])
    import flwr,torch,numpy,pandas,psutil,sklearn
    for m in [flwr,torch,numpy,pandas,psutil,sklearn]:
        if not Path(m.__file__).resolve().is_relative_to(env):raise ValueError('Import outside accepted venv: '+m.__name__)
    if torch.__version__!='2.10.0+cpu' or torch.version.cuda is not None:raise ValueError('CPU PyTorch identity changed')
    flwr_root=Path(flwr.__file__).resolve().parent
    hashes={str(p.relative_to(flwr_root)):sha(p) for p in sorted(flwr_root.rglob('*')) if p.is_file() and p.suffix not in ['.pyc','.pyo']}
    return {'status':'PASS','python':platform.python_version(),'packages':packages,'identity':ident,'installed_wheel':installed,'flower_runtime_hashes':hashes,'environment_reused_in_place':True,'environment_modified':False,'pip_executed':False,'benchmark_data_read':False}
if __name__=='__main__':
    write_json(Path(sys.argv[2]),probe(sys.argv[1]))
