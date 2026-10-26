"""Export an allowlisted failure capsule, never whole Flower stores or local logs."""
from __future__ import annotations
import json
import re
from pathlib import Path
from .common import read, write, sha

_FRAME = re.compile(r'^\s*File "([^"\n]+)", line ([0-9]+), in ([^\n]+)$', re.MULTILINE)
_CLASS = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_SAFE_ROUND = re.compile(r'^Message outside frozen measurement scope: round=-?\d+; phase=(fit|initial|round|selected); max_round=\d+$')
_SAFE_EXACT = {'Message outside fixed timing rounds/phases',
               'Condition outside frozen measurement scope',
               'Run-config and bundled measurement round budgets differ',
               'Workflow outside frozen measurement round budget',
               'Missing measured instruction record'}


def collect_failure_capsule(job: Path, reports: Path) -> dict:
    """Diagnostic only. Unknown exception messages and source/locals are redacted.

    No raw traceback export: only file/line/function frames from the local error
    file. No history losses, client outputs, model fingerprints, parameter arrays,
    serialized messages, protocol secrets or Flower databases are copied.
    """
    job, reports = Path(job), Path(reports)
    meta=read(job/'job.json');co=meta['condition'];mode=meta['configuration']
    if co not in ('reference','noniid') or mode not in ('A','B','C'):
        raise ValueError('Unknown failure job identity')
    result={'status':'DIAGNOSTIC_ONLY','configuration':mode,'condition':co,
            'server_failure_found':False,'record_values_exported':False,
            'local_variables_exported':False,'full_logs_exported':False}
    f=job/'server_outputs'/co/'server_result.json'
    if f.is_file() and not f.is_symlink():
        v=read(f)
        if v.get('status')=='FAIL':
            kind=v.get('exception','UNKNOWN');message=v.get('message','')
            result.update(server_failure_found=True,server_exception=kind if isinstance(kind,str) and _CLASS.fullmatch(kind) else 'UNKNOWN',
                message=message if isinstance(message,str) and (message in _SAFE_EXACT or _SAFE_ROUND.fullmatch(message)) else '[redacted; inspect retained local log]',
                server_result_sha256=sha(f))
    trace=job/f'error_{co}_LOCAL_ONLY.txt'
    if trace.is_file() and not trace.is_symlink():
        text=trace.read_text(encoding='utf-8',errors='replace')
        result['frames']=[{'file':a,'line':int(b),'function':c.strip()} for a,b,c in _FRAME.findall(text)]
        result['traceback_sha256']=sha(trace)
    h=job/'server_outputs'/co/'history.json'
    if h.is_file() and not h.is_symlink():
        rows=read(h).get('history',[])
        rounds=[r.get('round') for r in rows]
        if rounds and all(type(r) is int for r in rounds) and rounds==list(range(rounds[-1]+1)):
            result['last_recorded_evaluation_round']=rounds[-1]
            result['next_round_completion_not_inferred']=True
    write(reports/'server_failure_capsule.json',result)
    return result
