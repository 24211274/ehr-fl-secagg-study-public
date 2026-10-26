"""Fail-closed wrappers around the installed Flower 1.22 SecAggPlusWorkflow."""
from __future__ import annotations
import time
from mai_measurement import server_span
from flwr.server.workflow import SecAggPlusWorkflow
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS,Key,Stage
from flwr.server.workflow.constant import Key as WorkflowKey,MAIN_CONFIGS_RECORD
from ehr_fedavg.server_app import BoundedGrid
from .policy import validate_cohort
from .wire import check_masked_reply
from .settings import scfg


class SecAggGrid(BoundedGrid):
    def send_and_receive(self,messages,*,timeout=None):
        sent=list(messages)
        train=bool(sent) and all(m.metadata.message_type=='train' for m in sent)
        phase=None
        if train:
            stages=[m.content.config_records[RECORD_KEY_CONFIGS][Key.STAGE] for m in sent]
            if len(set(stages))!=1:raise ValueError('Mixed secure protocol stages')
            phase=stages[0]
        replies=list(super().send_and_receive(sent,timeout=timeout))
        validate_cohort([m.metadata.dst_node_id for m in sent],[m.metadata.src_node_id for m in replies])
        for m in replies:
            if train:check_masked_reply(m.content,phase)
        masked_bytes=0
        if train and phase==Stage.COLLECT_MASKED_VECTORS:
            masked_bytes=sum(len(blob) for msg in replies for blob in msg.content.config_records[RECORD_KEY_CONFIGS][Key.MASKED_PARAMETERS])
        self.exchanges[-1].update(masked_parameter_payload_bytes=masked_bytes,secagg_stage=phase,cohort_check_pass=True,
            plaintext_training_array_records_observed=0 if train else None,
            content_values_logged=False,wire_bytes_measured=False)
        return replies


class StrictSecAggPlusWorkflow(SecAggPlusWorkflow):
    def __init__(self):super().__init__(**scfg()['secagg']);self.audit=[]
    def __call__(self,grid,context):
        r=context.state.config_records[MAIN_CONFIGS_RECORD][WorkflowKey.CURRENT_ROUND]
        previous=context.strategy.last_round;start=time.perf_counter();self._round=r
        context.strategy.secagg_stage=None
        if r!=previous+1:raise RuntimeError('Unexpected secure round')
        super().__call__(grid,context)
        if context.strategy.last_round!=r:raise RuntimeError('Secure workflow halted; stale model forbidden')
        self.audit.append({'round':r,'status':'PASS','four_stages_completed':True,'wall_seconds':time.perf_counter()-start,
                           'stock_workflow_executed':True,'privacy_claim_validated':False})
        context.strategy.secagg_stage=None
    def _checked(self,stage,fn,grid,context,state):
        context.strategy.secagg_stage=stage;start=time.perf_counter()
        with server_span('secagg_'+stage,self._round):
            ok=fn(grid,context,state)
        if not ok or state.failures or len(state.sampled_node_ids)!=3 or state.active_node_ids!=state.sampled_node_ids:
            raise RuntimeError('Incomplete secure stage; no fallback or dropout acceptance')
        context.strategy.protocol_stages.append({'round':self._round,'stage':stage,'participants':3,'status':'PASS','wall_seconds':time.perf_counter()-start})
        return True
    def setup_stage(self,g,c,s):return self._checked(Stage.SETUP,super().setup_stage,g,c,s)
    def share_keys_stage(self,g,c,s):return self._checked(Stage.SHARE_KEYS,super().share_keys_stage,g,c,s)
    def collect_masked_vectors_stage(self,g,c,s):return self._checked(Stage.COLLECT_MASKED_VECTORS,super().collect_masked_vectors_stage,g,c,s)
    def unmask_stage(self,g,c,s):return self._checked(Stage.UNMASK,super().unmask_stage,g,c,s)
