"""Public build-identity gate for the approved local PRG backport.

This detects accidental version mismatches, not hostile clients. It is NOT
remote attestation, authentication, TLS, or a privacy certification. No model
updates, update hashes, masks, keys, or secret shares enter these records.
"""
from __future__ import annotations
import hashlib
import copy
import importlib
import importlib.metadata
import inspect
import sys
from pathlib import Path

VERSION = '1.22.0+mai.prg7570.1'
FUNCTION_SHA256 = '2513f6c64f83c4ca6cd69800a517426b701e3e41e8e349f001542b1a7144c230'
HELPER_SHA256 = '62935b91cf10f41f2036f8ccdbe65f257de86d4ada65cff7ff9fe254927c361c'
EXPECTED_BUILD_ID = f'flwr:{VERSION}:prg:{FUNCTION_SHA256}'
OLD_BUILD_ID = 'flwr:1.22.0:prg:19ee2545ed58c0174c0b1a56f4211d635347156163ae76a711613badad4b25e9'
ID_RECORD = 'mai_prg_identity'
PROBE_RECORD = 'mai_prg_probe'
QUERY_TYPE = 'query.mai_prg_identity'
CLIENTS = {'C1','C2','C3'}

class IdentityMismatch(RuntimeError):
    """Fail closed on a missing, inconsistent or unrecognised public build ID."""

def _sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()

def require_request(record,expected=EXPECTED_BUILD_ID):
    if not isinstance(record,dict) or set(record)!={'build_id'} or record['build_id']!=expected:
        raise IdentityMismatch('PRG implementation identity mismatch; no fallback')

def require_reply(record,expected=EXPECTED_BUILD_ID):
    if not isinstance(record,dict) or set(record)!={'client','build_id'} or record.get('client') not in CLIENTS or record.get('build_id')!=expected:
        raise IdentityMismatch('Wrong reply implementation identity')

def binding_identity()->dict:
    """Read and verify installed source and both imported function bindings."""
    helper=importlib.import_module('flwr.common.secure_aggregation.secaggplus_utils')
    client=importlib.import_module('flwr.client.mod.secure_aggregation.secaggplus_mod')
    server=importlib.import_module('flwr.server.workflow.secure_aggregation.secaggplus_workflow')
    path=Path(helper.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise IdentityMismatch('PRG helper is outside the selected Python environment')
    observed=_sha(inspect.getsource(helper.pseudo_rand_gen).encode())
    if importlib.metadata.version('flwr')!=VERSION or observed!=FUNCTION_SHA256 or _sha(path.read_bytes())!=HELPER_SHA256:
        raise IdentityMismatch('Installed PRG source/version differs from the approved local build')
    if client.pseudo_rand_gen is not helper.pseudo_rand_gen or server.pseudo_rand_gen is not helper.pseudo_rand_gen:
        raise IdentityMismatch('Mask generation bindings differ between client and server')
    return {'status':'PASS','build_id':EXPECTED_BUILD_ID,'version':VERSION,
            'function_sha256':FUNCTION_SHA256,'helper_sha256':HELPER_SHA256,
            'client_server_bindings_match':True,'official_release':False,
            'remote_attestation':False,'privacy_claim_validated':False}

def implementation_identity_mod(msg,context,call_next):
    """Outermost client mod: check ID BEFORE training, evaluation or SecAgg mod.

    Dedicated queries cannot reach the data-loading client_fn, even when the ID
    is intentionally wrong. Other message types raise on every ID mismatch.
    """
    from flwr.common import Message,RecordDict,ConfigRecord
    ident=binding_identity();cid=context.node_config.get('client-id')
    if cid not in CLIENTS:raise IdentityMismatch('Unknown logical client')
    record=msg.content.config_records.pop(ID_RECORD,None)
    record=dict(record) if record is not None else None
    if msg.metadata.message_type==QUERY_TYPE:
        if msg.content.array_records or msg.content.metric_records or msg.content.config_records:
            raise IdentityMismatch('Identity query must not include datasets or other payloads')
        try:
            require_request(record);accepted=True
        except IdentityMismatch:
            accepted=False
        reply=Message(content=RecordDict({PROBE_RECORD:ConfigRecord({
            'client':cid,'build_id':ident['build_id'],'accepted':accepted,
            'data_callback_called':False})}),reply_to=msg)
        return reply
    require_request(record)
    reply=call_next(msg,context)
    if not reply.has_error():
        if ID_RECORD in reply.content.config_records:raise IdentityMismatch('Duplicate reply ID carrier')
        reply.content.config_records[ID_RECORD]=ConfigRecord({'build_id':ident['build_id'],'client':cid})
    return reply

class IdentityCheckedTransport:
    """Delegate to Flower; remove identity acknowledgements before C schema checks."""
    def __init__(self,inner):
        self.inner=inner;self.identity=binding_identity();self.node_clients={};self.checked_exchanges=0
    def __getattr__(self,name):return getattr(self.inner,name)
    def send_and_receive(self,messages,*,timeout=None):
        from flwr.common import ConfigRecord
        # Some SecAgg stages reuse one content object across destinations.
        # Clone each request before adding identity so no shared object is mutated.
        sent=[copy.deepcopy(m) for m in messages]
        nodes=[m.metadata.dst_node_id for m in sent]
        if not sent or len(nodes)!=len(set(nodes)):
            raise IdentityMismatch('Unique nonempty destinations required')
        for m in sent:
            if ID_RECORD in m.content.config_records:raise IdentityMismatch('Duplicate request identity record')
            m.content.config_records[ID_RECORD]=ConfigRecord({'build_id':self.identity['build_id']})
        replies=list(self.inner.send_and_receive(sent,timeout=timeout))
        if len(replies)!=len(sent) or any(m.has_error() for m in replies):
            raise IdentityMismatch('Missing/error reply before implementation ID validation')
        if {m.metadata.src_node_id for m in replies}!=set(nodes):
            raise IdentityMismatch('Reply cohort does not match request cohort')
        clients=[]
        for m in replies:
            record=m.content.config_records.pop(ID_RECORD,None)
            record=dict(record) if record is not None else None
            require_reply(record,self.identity['build_id'])
            node=m.metadata.src_node_id;cid=record['client'];clients.append(cid)
            if node in self.node_clients and self.node_clients[node]!=cid:
                raise IdentityMismatch('Node changed logical client identity')
            self.node_clients[node]=cid
        if len(clients)!=len(set(clients)):raise IdentityMismatch('Duplicate client identities')
        self.checked_exchanges+=1
        return replies

def verify_handshake_report(report:dict)->None:
    if report.get('status')!='PASS' or report.get('build_id')!=EXPECTED_BUILD_ID or report.get('all_clients')!=['C1','C2','C3']:
        raise IdentityMismatch('Missing successful three-client implementation handshake')
    if report.get('old_identity_rejected_before_data_callbacks') is not True or report.get('valid_identity_accepted') is not True:
        raise IdentityMismatch('Missing positive/negative identity evidence')
    if report.get('requests')!=6 or report.get('replies')!=6 or report.get('remote_attestation') is not False:
        raise IdentityMismatch('Invalid handshake counts or claim')

def network_handshake(grid,timeout)->dict:
    """Two public queries per node, on actual Flower transport, before datasets.

    First advertise the old public build ID to check refusal, then advertise
    the approved ID. This does not run a genuinely old remote process: it tests
    rejection of that ID by the deployed candidate client middleware.
    """
    from flwr.common import Message,ConfigRecord,RecordDict
    identity=binding_identity();nodes=sorted(grid.get_node_ids())
    if len(nodes)!=3 or len(set(nodes))!=3:raise IdentityMismatch('Need exactly three distinct nodes')
    mapping={}
    for advertised,want in [(OLD_BUILD_ID,False),(EXPECTED_BUILD_ID,True)]:
        sent=[Message(content=RecordDict({ID_RECORD:ConfigRecord({'build_id':advertised})}),
            dst_node_id=n,message_type=QUERY_TYPE,group_id='prg-pretraining-check',ttl=timeout) for n in nodes]
        replies=list(grid.send_and_receive(sent,timeout=timeout))
        if len(replies)!=3 or any(m.has_error() for m in replies) or {m.metadata.src_node_id for m in replies}!=set(nodes):
            raise IdentityMismatch('Incomplete deployed identity probe')
        seen=set()
        for msg in replies:
            content=msg.content
            if content.array_records or content.metric_records or set(content.config_records)!={PROBE_RECORD}:
                raise IdentityMismatch('Unexpected identity-probe payload')
            record=dict(content.config_records[PROBE_RECORD])
            if set(record)!={'client','build_id','accepted','data_callback_called'}:
                raise IdentityMismatch('Unexpected identity-probe fields')
            if record['accepted'] is not want or record['data_callback_called'] is not False:
                raise IdentityMismatch('ID probe reached callback or accepted the wrong build')
            require_reply({'client':record['client'],'build_id':record['build_id']})
            cid=record['client'];node=msg.metadata.src_node_id
            if cid in seen:raise IdentityMismatch('Duplicate logical client in identity probe')
            seen.add(cid)
            if node in mapping and mapping[node]!=cid:raise IdentityMismatch('Client mapping changed across probes')
            mapping[node]=cid
        if seen!=CLIENTS:raise IdentityMismatch('Missing logical client in identity probe')
    result={'status':'PASS','build_id':identity['build_id'],'all_clients':sorted(CLIENTS),
            'node_mapping':{str(n):c for n,c in mapping.items()},'requests':6,'replies':6,
            'old_identity_rejected_before_data_callbacks':True,'valid_identity_accepted':True,
            'actual_transport_used':True,'remote_old_process_used':False,
            'remote_attestation':False,'privacy_claim_validated':False,
            'secret_material_or_client_update_fingerprints_logged':False}
    verify_handshake_report(result)
    return result
