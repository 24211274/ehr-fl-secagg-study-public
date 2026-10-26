"""Public implementation identity checks, NOT remote attestation or authentication."""
from __future__ import annotations
import hashlib,importlib,importlib.metadata,inspect
from pathlib import Path
from .build import manifest,sha

ID_RECORD='mai_prg_identity'
class IdentityMismatch(RuntimeError):pass

def require_id(record,expected):
    if not isinstance(record,dict) or set(record)!={'build_id'} or record['build_id']!=expected:
        raise IdentityMismatch('PRG implementation identity mismatch; no protocol fallback')

def require_cohort(records,expected):
    if len(records)!=3 or {r.get('client') for r in records}!={'C1','C2','C3'}:
        raise IdentityMismatch('Three distinct client identities required')
    for r in records:
        if set(r)!={'client','build_id'}:raise IdentityMismatch('Unexpected identity fields')
        require_id({'build_id':r['build_id']},expected)

def binding_identity(mode='candidate'):
    m=manifest()
    if mode not in ('candidate','reference'):raise ValueError('Bad identity mode')
    ver=m['candidate_version'] if mode=='candidate' else m['stock_version']
    want=m['new_function_sha256'] if mode=='candidate' else m['old_function_sha256']
    helper=importlib.import_module('flwr.common.secure_aggregation.secaggplus_utils')
    client=importlib.import_module('flwr.client.mod.secure_aggregation.secaggplus_mod')
    server=importlib.import_module('flwr.server.workflow.secure_aggregation.secaggplus_workflow')
    observed=sha(inspect.getsource(helper.pseudo_rand_gen).encode())
    if importlib.metadata.version('flwr')!=ver or observed!=want:
        raise IdentityMismatch('Installed Flower version/PRG fingerprint is not the selected build')
    if not (client.pseudo_rand_gen is helper.pseudo_rand_gen and server.pseudo_rand_gen is helper.pseudo_rand_gen):
        raise IdentityMismatch('Client/server mask-function bindings are inconsistent')
    file_key='backport_secaggplus_utils.py' if mode=='candidate' else 'flower_v1_22_0_secaggplus_utils.py'
    if sha(Path(helper.__file__).read_bytes())!=m['sources'][file_key]:raise IdentityMismatch('Helper module file differs')
    build_id=f'flwr:{ver}:prg:{observed}'
    return {'build_id':build_id,'version':ver,'function_sha256':observed,'client_and_server_bindings_match':True,
            'mode':mode,'remote_attestation':False,'privacy_claim_validated':False}

def guarded_client_call(app,message,context,mode):
    from flwr.common import ConfigRecord
    local=binding_identity(mode)['build_id']
    record=message.content.config_records.pop(ID_RECORD,None)
    require_id(dict(record) if record is not None else None,local)
    reply=app(message,context)
    if not reply.has_error():reply.content.config_records[ID_RECORD]=ConfigRecord({'build_id':local})
    return reply
