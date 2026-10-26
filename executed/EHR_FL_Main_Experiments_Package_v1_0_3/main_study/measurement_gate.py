"""Bounded envelope dry-run, not training, secure aggregation or deployment.

Exercise both production measurement boundaries with the INSTALLED Flower Message
and Record classes over every configured round. Delivery and callbacks are local
fixtures; no sockets, keys, masks, benchmark arrays, optimisation or learned states
are involved. This gate runs before any main-block training.
"""
from __future__ import annotations
import copy
import json
import tempfile
import types
from pathlib import Path


def check_measurement_horizon(job: Path, configuration: str, condition: str) -> dict:
    from flwr.common import Message, Metadata, ConfigRecord, RecordDict
    import mai_measurement as mm
    from timing_support.analysis import reconcile_boundary

    if configuration not in ('B','C'):
        raise ValueError('Measurement envelope probe requires B or C')
    scope = mm.measurement_scope()
    if scope['conditions'] != [condition]:
        raise ValueError('Envelope probe must use the bound single-condition job')
    rounds = scope['rounds']
    clients = {11:'C1',12:'C2',13:'C3'}
    sequence = 0
    callback_count = 0
    rejects = []

    def request(node, r, phase, stage=None):
        nonlocal sequence
        sequence += 1
        if phase == 'handshake':
            kind = 'query.measurement_probe'
            record = {'measurement_probe': ConfigRecord({'public_fixture': True})}
        elif phase == 'fit':
            kind = 'train'
            record = ({'secaggplus_configs': ConfigRecord({'stage':stage})}
                      if configuration == 'C' else
                      {'fitins.config': ConfigRecord({'server_round':r,'condition':condition})})
        else:
            kind = 'evaluate'
            record = {'evaluateins.config':ConfigRecord({'server_round':r,'phase':phase,'condition':condition})}
        m = Message(content=RecordDict(record), dst_node_id=node,
                    message_type=kind, group_id=str(r))
        src = m.metadata
        metadata = Metadata(run_id=src.run_id,message_id=f'gate-{sequence}',
            src_node_id=src.src_node_id,dst_node_id=src.dst_node_id,
            reply_to_message_id=src.reply_to_message_id,group_id=src.group_id,
            created_at=src.created_at,ttl=src.ttl,message_type=src.message_type)
        return Message(metadata=metadata,content=copy.deepcopy(m.content))

    with tempfile.TemporaryDirectory(prefix='measurement_envelope_',dir=job) as temp:
        def callback(message, context):
            nonlocal callback_count
            callback_count += 1
            desc = mm.descriptor(message)
            if desc['stage'] in ('fit','collect_masked_vectors'):
                # Empty spans exercise bookkeeping, not actual data/compute work.
                with mm.client_span('data_load_with_checks'):
                    pass
                with mm.client_span('local_fit_with_checks'):
                    pass
            return Message(content=RecordDict({'gate':ConfigRecord({'public_fixture':True})}),
                           reply_to=message)

        class LocalEnvelopeTransport:
            def send_and_receive(self, messages, *, timeout=None):
                replies=[]
                for message in messages:
                    ctx=types.SimpleNamespace(node_config={'client-id':clients[message.metadata.dst_node_id]},
                        run_config={'runtime-root':temp,'condition':condition,'num-server-rounds':rounds})
                    replies.append(mm.timing_boundary_mod(message,ctx,callback))
                return replies

        grid=mm.ObservedRawGrid(LocalEnvelopeTransport())
        def exchange(r,phase,stage=None):
            grid.send_and_receive([request(node,r,phase,stage) for node in clients])
        exchange(0,'handshake');exchange(0,'handshake');exchange(0,'initial')
        for r in range(1,rounds+1):
            for stage in (('fit',) if configuration=='B' else
                          ('setup','share_keys','collect_masked_vectors','unmask')):
                exchange(r,'fit',stage)
            exchange(r,'round')
        exchange(rounds,'selected')
        positive_callbacks=callback_count
        rows=[json.loads(p.read_text()) for p in sorted((Path(temp)/'measurement_client'/condition).rglob('*.json'))]
        evidence=reconcile_boundary({'exchanges':grid.rows},rows,configuration,rounds=rounds)
        # Negative examples must stop before the transport/callback and before a record is written.
        for label,r,phase,stage in (
                ('fit_zero',0,'fit','setup'),('fit_over_budget',rounds+1,'fit','setup'),
                ('eval_over_budget',rounds+1,'round',None),('selected_zero',0,'selected',None),
                ('initial_nonzero',1,'initial',None)):
            m=request(11,r,phase,stage)
            try:
                grid.send_and_receive([m])
            except ValueError:
                rejects.append(label)
            else:
                raise ValueError('Measurement bound was not enforced: '+label)
        wrong = 'noniid' if condition=='reference' else 'reference'
        ctx=types.SimpleNamespace(node_config={'client-id':'C1'},
            run_config={'runtime-root':temp,'condition':wrong,'num-server-rounds':rounds})
        try:
            mm.timing_boundary_mod(request(11,1,'fit','setup'),ctx,callback)
        except ValueError:
            rejects.append('other_condition')
        else:
            raise ValueError('Unbound condition accepted by client observer')
        if callback_count != positive_callbacks:
            raise ValueError('A rejected envelope reached its callback')
        expected=3*(4+rounds*(2 if configuration=='B' else 5))
        if len(rows)!=expected or evidence['matched_messages']!=expected:
            raise ValueError('Incomplete all-round envelope probe')
        for r in range(1,rounds+1):
            if mm.descriptor(request(11,r,'selected'))['round']!=r:
                raise ValueError('Selected-checkpoint phase not accepted')
    return {'status':'PASS','configuration':configuration,'condition':condition,
            'measurement_scope':scope,'rounds_exercised':list(range(1,rounds+1)),
            'initial_round_checked':0,'selected_rounds_checked':list(range(1,rounds+1)),
            'matched_request_response_pairs':expected,'negative_cases_rejected':rejects,
            'callback_count':positive_callbacks,'native_Flower_message_records_used':True,
            'local_fixture_transport_only':True,'actual_training_performed':False,
            'secagg_protocol_executed':False,'deployment_engine_used':False,
            'benchmark_data_used':False,'test_data_opened':False,
            'cost_result':False,'record_values_exported':False,
            'boundary_byte_counting_rule_id':mm.BYTE_COUNTING_RULE_ID}
