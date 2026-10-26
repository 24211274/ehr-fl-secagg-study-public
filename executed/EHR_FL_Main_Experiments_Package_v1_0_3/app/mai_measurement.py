"""Read-only instrumentation bound to the immutable configuration of each job.

Only numeric counts, public phase labels and durations are exported. Record values,
model arrays, update fingerprints, keys, masks and shares are never written here.
Byte counts follow Flower's message_size_mod record-content convention with a
read-only zero-length ConfigRecord value repair, not TCP or protobuf wire bytes. Both-end counters are cross-checked; only client totals count.
"""
from __future__ import annotations
import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ACTIVE_CLIENT = ContextVar('mai_timing_client', default=None)
_SERVER_EVENTS: list[dict] = []
_SERVER_ROUND_START: dict[int,int] = {}
_SERVER_PHASE = ContextVar('mai_timing_server_phase', default=None)

BYTE_COUNTING_RULE_ID = 'flower122_record_content_zero_length_v1'


def _config_record_bytes(record) -> int:
    """Count config content without changing the Message or the installed library.

    Flower 1.22 indexes v[0] for a list, even when it is empty. It also rejects
    a zero-length scalar str/bytes in get_var_bytes. These are valid ConfigRecord
    values. For these cases only, count a temporary projection of the non-empty
    entries with the native method and add ALL omitted key lengths. A zero-length
    value has no element bytes under this logical-content convention.

    The live record keeps all its keys/values, including protocol lists. Native
    counting remains responsible for every non-empty value and every non-config
    record. Unrelated exceptions are deliberately NOT caught. This is not a
    protobuf/TCP counter; the native len(str)/len(key) convention is retained.
    """
    zero_keys = [key for key, value in record.items()
                 if isinstance(value, (list, str, bytes)) and len(value) == 0]
    if not zero_keys:
        return record.count_bytes()

    from flwr.common import ConfigRecord
    if not isinstance(record, ConfigRecord):
        raise TypeError('Zero-length config repair requires a Flower ConfigRecord')
    omitted = set(zero_keys)
    projection = ConfigRecord(
        {key: value for key, value in record.items() if key not in omitted},
        keep_input=True,
    )
    native_size = projection.count_bytes()
    if type(native_size) is not int or native_size < 0:
        raise ValueError('Config counter must yield a nonnegative Python integer')
    return native_size + sum(len(key) for key in zero_keys)


def record_bytes(content):
    """Flower 1.22 logical record-content count, with zero-length config repair."""
    groups = [('array', content.array_records), ('metric', content.metric_records),
              ('config', content.config_records)]
    result = {}
    for kind, records in groups:
        sizes = [(_config_record_bytes(record) if kind == 'config'
                  else record.count_bytes()) for record in records.values()]
        if any(type(n) is not int or n < 0 for n in sizes):
            raise ValueError('Record counter must yield nonnegative Python integers')
        result[kind+'_bytes'] = sum(sizes)
    result['total_bytes'] = sum(result.values())
    # Guard against silently uncounted record types.
    if sum(len(rs) for _,rs in groups) != len(content):
        raise ValueError('Unknown or overlapping RecordDict record types')
    return result


def measurement_scope() -> dict:
    """Read the verified, job-specific resource, never a hard-coded pilot limit.

    This is a fresh read (no process-global cached job identity). The existing
    resource loader checks the embedded byte digest. In deployed applications the
    resource is bound by Prepare and is checked against runtime.json elsewhere.
    This function reads no benchmark rows, model state, masks, keys or shares.
    """
    from ehr_fedavg.resources import read_document, resource_sha256
    conf = read_document('fixed_config.json')
    rounds, conditions = conf.get('rounds'), conf.get('conditions')
    stage = conf.get('stage')
    if type(rounds) is not int or not 1 <= rounds <= 30:
        raise ValueError('Invalid frozen measurement round budget')
    if (not isinstance(conditions, list) or not conditions
            or any(c not in ('reference', 'noniid') for c in conditions)
            or len(conditions) != len(set(conditions))):
        raise ValueError('Invalid frozen measurement condition set')
    if not isinstance(stage, str) or not stage:
        raise ValueError('Missing frozen measurement stage')
    if stage == 'MAIN_CONTROLLED_COMPARISON' and rounds != 30:
        raise ValueError('Main measurement requires the frozen 30-round budget')
    if stage == 'TIMING_REPEATED_REFERENCE_PILOT' and (rounds != 6 or conditions != ['reference']):
        raise ValueError('Timing pilot contract changed')
    return {'rounds': rounds, 'conditions': conditions, 'stage': stage,
            'fixed_config_sha256': resource_sha256('fixed_config.json')}


def _round_integer(value) -> int:
    # Reject lossy conversions, bools, missing values and arbitrary strings.
    if type(value) is int:
        return value
    if isinstance(value, str) and value and value.isascii() and value.isdecimal():
        return int(value)
    raise ValueError('Measured round must be an integer or a decimal group ID')


def descriptor(message, *, scope=None):
    scope = measurement_scope() if scope is None else scope
    typ = message.metadata.message_type
    content = message.content
    if typ.startswith('query.'):
        return {'message_type':'query','stage':'identity','round':0,'phase':'handshake'}
    if typ not in ('train','evaluate'):
        raise ValueError('Unexpected measured message type')
    if typ == 'train' and 'secaggplus_configs' in content.config_records:
        stage = content.config_records['secaggplus_configs']['stage']
        if stage not in ('setup','share_keys','collect_masked_vectors','unmask'):
            raise ValueError('Unknown secure stage')
        r = _round_integer(message.metadata.group_id)
        phase='fit'
    else:
        key='fitins.config' if typ=='train' else 'evaluateins.config'
        if key not in content.config_records:
            return {'message_type':typ,'stage':'invalid','round':-1,'phase':'invalid'}
        conf=content.config_records[key]; r=_round_integer(conf['server_round'])
        stage='fit' if typ=='train' else 'evaluate'
        phase='fit' if typ=='train' else conf['phase']
    valid = (phase == 'initial' and typ == 'evaluate' and r == 0) or (
        1 <= r <= scope['rounds'] and (
            (typ == 'train' and phase == 'fit') or
            (typ == 'evaluate' and phase in ('round', 'selected'))))
    if not valid:
        raise ValueError('Message outside frozen measurement scope: '
                         f'round={r}; phase={phase}; max_round={scope["rounds"]}')
    return {'message_type':typ,'stage':stage,'round':r,'phase':phase}


@contextmanager
def client_span(name):
    """Local wall/process CPU spans; never add overlapping client wall times."""
    if name not in ('data_load_with_checks','local_fit_with_checks','train_scoring','validation_scoring'):
        raise ValueError('Unknown local timing boundary')
    start=time.perf_counter_ns();cpu=time.process_time_ns();ok=False
    try:
        yield
        ok=True
    finally:
        end=time.perf_counter_ns();used=time.process_time_ns()-cpu
        active=_ACTIVE_CLIENT.get()
        if active is not None:
            active.append({'name':name,'wall_ns':end-start,'cpu_ns':used,'completed':ok})


def timing_boundary_mod(msg,context,call_next):
    """Outermost: count requests before mutation and replies AFTER secure masking.

    No fields are added to the exchanged Message. Records go to a local researcher
    directory which the ServerApp never reads. Code identity checks remain inside
    this read-only observer and still precede all data callbacks.
    """
    scope=measurement_scope()
    info=descriptor(msg,scope=scope);cid=context.node_config['client-id']
    if cid not in ('C1','C2','C3'):raise ValueError('Unknown client label')
    root=Path(context.run_config['runtime-root']).resolve()
    co=context.run_config['condition']
    if co not in scope['conditions']:
        raise ValueError('Condition outside frozen measurement scope')
    if ('num-server-rounds' in context.run_config
            and context.run_config['num-server-rounds'] != scope['rounds']):
        raise ValueError('Run-config and bundled measurement round budgets differ')
    # Preserve ordering for deliberate identity-negative train probes: an absent
    # instruction is still classified invalid, then the inner identity guard
    # rejects its old build ID before any data callback. The outer server
    # observer separately rejects a missing instruction on an actual send.
    ident=str(msg.metadata.message_id)
    if not ident:raise ValueError('Measured deployment request must have a message ID')
    path=root/'measurement_client'/co/cid/(hashlib.sha256(ident.encode()).hexdigest()+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Repeated request observed; no silent duplicate/retry')
    count_start=time.perf_counter_ns();incoming=record_bytes(msg.content)
    counter_ns=time.perf_counter_ns()-count_start
    local=[];token=_ACTIVE_CLIENT.set(local);start=time.perf_counter_ns();cpu=time.process_time_ns()
    ok=False;outgoing=None;error_type=None
    try:
        result=call_next(msg,context)
        end=time.perf_counter_ns();cpu_ns=time.process_time_ns()-cpu
        if result.has_error():raise RuntimeError('Error reply in measured exchange')
        count_start=time.perf_counter_ns();outgoing=record_bytes(result.content)
        counter_ns+=time.perf_counter_ns()-count_start
        ok=True
        return result
    except BaseException as exc:
        error_type=type(exc).__name__
        if 'end' not in locals():end=time.perf_counter_ns();cpu_ns=time.process_time_ns()-cpu
        raise
    finally:
        _ACTIVE_CLIENT.reset(token)
        row={'status':'PASS' if ok else 'FAIL','client':cid,'request_id':ident,
             'node_id':int(msg.metadata.dst_node_id),**info,
             'incoming_record_bytes':incoming,'outgoing_record_bytes':outgoing,
             'callback_wall_ns':end-start,'callback_cpu_ns':cpu_ns,
             'byte_counter_wall_ns':counter_ns,'local_spans':local,'error_type':error_type,
             'record_values_written':False,'wire_bytes_measured':False,
             'local_export_not_message_payload':True,
             'condition':co,'measurement_scope':scope,
             'byte_counting_rule_id':BYTE_COUNTING_RULE_ID}
        # exclusive creation; numeric reports only, strict JSON
        with path.open('x',encoding='utf-8') as f:
            json.dump(row,f,indent=2,allow_nan=False);f.write('\n')


@contextmanager
def server_span(name,round_no):
    start=time.perf_counter_ns();cpu=time.process_time_ns();done=False
    try:
        yield
        done=True
    finally:
        end=time.perf_counter_ns()
        _SERVER_EVENTS.append({'name':str(name),'round':int(round_no),
                               'start_ns':start,'end_ns':end,'wall_ns':end-start,
                               'cpu_ns':time.process_time_ns()-cpu,'completed':done})


def measured_workflow(fn,phase):
    """Wrap the stock fit/evaluation workflow without replacing its behaviour."""
    if phase not in ('fit_workflow','evaluation_workflow'):raise ValueError('Unknown workflow')
    def call(grid,context):
        from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD,Key
        r=_round_integer(context.state.config_records[MAIN_CONFIGS_RECORD][Key.CURRENT_ROUND])
        if not 1 <= r <= measurement_scope()['rounds']:
            raise ValueError('Workflow outside frozen measurement round budget')
        token=_SERVER_PHASE.set((phase,r))
        if phase=='fit_workflow':
            if r in _SERVER_ROUND_START:raise ValueError('Repeated measured round')
            _SERVER_ROUND_START[r]=time.perf_counter_ns()
        try:
            with server_span(phase,r):value=fn(grid,context)
            if phase=='evaluation_workflow':
                end=time.perf_counter_ns();start=_SERVER_ROUND_START.pop(r)
                _SERVER_EVENTS.append({'name':'outer_round_including_evaluation','round':r,
                    'start_ns':start,'end_ns':end,'wall_ns':end-start,'cpu_ns':None,'completed':True})
            return value
        finally:_SERVER_PHASE.reset(token)
    return call


class ObservedRawGrid:
    """Observer below identity addition/removal; equal-boundary byte cross-check."""
    def __init__(self,inner):
        self.inner=inner;self.rows=[];self.scope=measurement_scope()
    def __getattr__(self,name):return getattr(self.inner,name)
    def send_and_receive(self,messages,*,timeout=None):
        sent=list(messages)
        if not sent:raise ValueError('Empty measured send')
        counting=time.perf_counter_ns()
        request_rows=[{'node_id':int(m.metadata.dst_node_id),**descriptor(m,scope=self.scope),
                       'outgoing_record_bytes':record_bytes(m.content)} for m in sent]
        if any(row['phase'] == 'invalid' for row in request_rows):
            raise ValueError('Missing measured instruction record')
        count_ns=time.perf_counter_ns()-counting
        start=time.perf_counter_ns()
        replies=list(self.inner.send_and_receive(sent,timeout=timeout))
        end=time.perf_counter_ns()
        if len(replies)!=len(sent) or any(m.has_error() for m in replies):
            raise RuntimeError('Incomplete observed exchange')
        mapping={int(m.metadata.src_node_id):m for m in replies}
        if len(mapping)!=len(sent) or set(mapping)!={r['node_id'] for r in request_rows}:
            raise RuntimeError('Observed cohort differs')
        counting=time.perf_counter_ns()
        for row in request_rows:
            reply=mapping[row['node_id']]
            row.update(request_id=str(reply.metadata.reply_to_message_id),
                       incoming_record_bytes=record_bytes(reply.content))
        count_ns+=time.perf_counter_ns()-counting
        self.rows.append({'exchange':len(self.rows)+1,'wall_ns':end-start,
                          'byte_counter_wall_ns':count_ns,'messages':request_rows,
                          'measurement_phase':_SERVER_PHASE.get(),
                          'record_values_written':False,'wire_bytes_measured':False})
        return replies


def reset_server():
    _SERVER_EVENTS.clear();_SERVER_ROUND_START.clear()


def server_report(grid,*,partial=False):
    if _SERVER_ROUND_START and not partial:raise ValueError('Unfinished measured round')
    return {'status':'INCOMPLETE' if partial else 'PASS','events':list(_SERVER_EVENTS),'exchanges':list(grid.rows),
            'clock':'perf_counter_ns','cpu_clock':'process_time_ns',
            'clock_info':{'resolution_seconds':time.get_clock_info('perf_counter').resolution,
                          'monotonic':time.get_clock_info('perf_counter').monotonic},
            'byte_definition':'Flower 1.22 logical Record content count; zero-length config values contribute zero value bytes and retain key lengths; not protobuf/TCP bytes',
            'byte_counting_rule_id':BYTE_COUNTING_RULE_ID,
            'wire_bytes_measured':False,'server_bytes_for_crosscheck_only':True,
            'instrumented_pipeline_not_uninstrumented_crypto_benchmark':True,
            'record_values_written':False,'measurement_scope':getattr(grid,'scope',None)}
