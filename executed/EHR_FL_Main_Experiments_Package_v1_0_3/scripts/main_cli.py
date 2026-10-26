#!/usr/bin/env python3
"""Explicit main-study stages. Prepare is not permission to start main training or tests."""
from __future__ import annotations
import argparse,fcntl,os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'app'),str(ROOT/'validation_support')]
from main_study.campaign import prepare,train_next,retry_block
from main_study.common import read

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['Prepare','PrepareRepair','TrainNextBlock','RetryBlock','Evaluate'])
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--inference-run',type=Path)
    p.add_argument('--accepted-run',type=Path);p.add_argument('--preprocessing-run',type=Path)
    p.add_argument('--parent-campaign',type=Path)
    p.add_argument('--authorisation',type=Path);p.add_argument('--benchmark-source',type=Path)
    a=p.parse_args();c=a.campaign.expanduser().resolve()
    if os.geteuid()==0 or sys.version_info[:2]!=(3,12):raise ValueError('Use original accepted non-root Linux Python 3.12; no installation')
    locks=[]
    try:
        for path in [Path.home()/'EHR_FL_Main_v1/main.lock',Path.home()/'EHR_FL_FullSize_BC_v1/fullsize.lock',
                     Path.home()/'EHR_FL_Timing_BC_v1/timing.lock',Path.home()/'EHR_FL_Inference_Seed_v1/inference_seed.lock']:
            path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise RuntimeError('Another protected workload is active; no process stopped')
            locks.append(f)
        if a.action in ('Prepare','PrepareRepair'):
            if not all([a.inference_run,a.accepted_run,a.preprocessing_run]):raise ValueError('Three accepted run paths required')
            if a.action=='PrepareRepair' and a.parent_campaign is None:
                raise ValueError('Reviewed parent campaign is required for code repair')
            return prepare(c,a.inference_run.resolve(),a.accepted_run.resolve(),a.preprocessing_run.resolve(),
                           repair_parent=a.parent_campaign.resolve() if a.action=='PrepareRepair' else None)
        if a.authorisation is None:raise PermissionError('A run-specific reviewed authorisation file is required')
        if a.action=='TrainNextBlock':return train_next(c,a.authorisation)
        if a.action=='RetryBlock':return retry_block(c,a.authorisation)
        if a.benchmark_source is None:raise ValueError('Final evaluation needs explicitly supplied original benchmark')
        from main_study.final_evaluator import evaluate
        return evaluate(c,a.authorisation,a.benchmark_source)
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
if __name__=='__main__':raise SystemExit(main())
