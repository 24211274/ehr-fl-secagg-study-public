"""Upstream SecAgg+ modifier with pinned PRG backport and constrained acknowledgements."""
from __future__ import annotations
from mai_runtime_identity import implementation_identity_mod
from mai_measurement import timing_boundary_mod
from pathlib import Path
from flwr.client import ClientApp,NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS,Key,Stage
from pilot_core.core import derived_seed
from central_smoke.identity import write_json
from ehr_fedavg.task import LocalTask,cfg,fit_one_client
from ehr_fedavg.math_checks import arrays_from_model
from .policy import local_range_check
from .wire import check_masked_reply
from .settings import scfg
from .local_task import SecAggLocalTask


class EHRClient(NumPyClient):
    def __init__(self,task):self.task=task
    def fit(self,parameters,config):return self.task.fit(parameters,config)
    def evaluate(self,parameters,config):return self.task.evaluate(parameters,config)
    def get_parameters(self,config):raise RuntimeError('Server-defined initial model required')


def client_fn(context:Context):
    return EHRClient(SecAggLocalTask(context.run_config['runtime-root'],context.node_config['client-id'],context.run_config['condition'])).to_client()


def guarded_secaggplus_mod(msg,context,call_next):
    if msg.metadata.message_type!='train':return call_next(msg,context)
    if RECORD_KEY_CONFIGS not in msg.content.config_records:raise RuntimeError('Unprotected fit instruction refused')
    record=msg.content.config_records[RECORD_KEY_CONFIGS];stage=record[Key.STAGE]
    if stage==Stage.SETUP:
        p=scfg()['secagg']
        expected={Key.SAMPLE_NUMBER:3,Key.SHARE_NUMBER:p['num_shares'],Key.THRESHOLD:p['reconstruction_threshold'],
                  Key.CLIPPING_RANGE:p['clipping_range'],Key.TARGET_RANGE:p['quantization_range'],
                  Key.MOD_RANGE:p['modulus_range'],Key.MAX_WEIGHT:p['max_weight']}
        if any(record.get(k)!=v or type(record.get(k))!=type(v) for k,v in expected.items()):raise ValueError('Unexpected SecAgg settings')
    if stage==Stage.UNMASK:
        if len(record[Key.ACTIVE_NODE_ID_LIST])!=3 or record[Key.DEAD_NODE_ID_LIST]:raise RuntimeError('Incomplete cohort refused before unmasking')
    # The selected wheel contains the approved backport. Never reseed secrets or monkey-patch functions here.
    reply=secaggplus_mod(msg,context,call_next)
    if reply.has_error():raise RuntimeError('SecAgg client returned error')
    check_masked_reply(reply.content,stage)
    return reply

app=ClientApp(client_fn=client_fn,mods=[timing_boundary_mod,implementation_identity_mod,guarded_secaggplus_mod])
