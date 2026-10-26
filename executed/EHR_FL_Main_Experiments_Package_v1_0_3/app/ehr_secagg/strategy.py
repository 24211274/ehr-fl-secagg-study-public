"""Consume the one already-weighted aggregate produced by SecAgg+, not client vectors."""
from __future__ import annotations
import time
import torch
from flwr.common import Code,parameters_to_ndarrays,ndarrays_to_parameters
from flwr.common.secure_aggregation.secaggplus_constants import Stage
from central_smoke.identity import write_json
from ehr_fedavg.strategy import CheckedFedAvg
from ehr_fedavg.math_checks import arrays_digest,load_arrays,validate_participants
from .policy import validate_public_fit_metrics,restored_float32,conservative_quantization_bound

class ProtectedFedAvg(CheckedFedAvg):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.protocol_stages=[];self.secagg_stage=None
    def aggregate_fit(self,server_round,results,failures):
        if self.secagg_stage!=Stage.UNMASK:raise RuntimeError('Aggregate callback outside SecAgg unmask; no plaintext fallback')
        if server_round!=self.last_round+1 or server_round not in self.round_start:raise ValueError('Unexpected aggregate round')
        if failures or len(results)!=3:raise RuntimeError('Missing secure participant')
        rows=[];copies=[]
        for proxy,res in results:
            if res.status.code!=Code.OK:raise RuntimeError('Non-OK protected fit status')
            validate_public_fit_metrics(res.metrics,self.digest)
            copies.append(parameters_to_ndarrays(res.parameters))
            rows.append({'node_id':proxy.node_id,'count':res.num_examples,'metrics':res.metrics})
        ordered=validate_participants(rows,'fit',self.digest);self.check_nodes(ordered)
        # Flower has already replaced each legacy result with the SAME secure average.
        arrays,cast_error=restored_float32(copies,self.spec)
        digest=arrays_digest(arrays,self.spec)
        if digest==self.digest:raise ValueError('No change in global state')
        # Convert only the diagnostic bound to Python float for strict JSON.
        # Recovered model dtype, quantisation arithmetic and guards are unchanged.
        audit={'status':'PASS','round':server_round,'configuration':'C','condition':self.condition,
               'input_digest':self.digest,'output_digest':digest,'client_labels':['C1','C2','C3'],
               'weights':[r['count'] for r in ordered],'sum_weights':sum(r['count'] for r in ordered),
               'local_training_updates':sum(self.cfg['local_updates_per_round'].values()),'received_legacy_carriers_are_same_aggregate':True,
               'per_client_updates_received_by_strategy':False,'float64_to_float32_max_error':cast_error,
               'a_priori_coordinate_quantization_bound':float(conservative_quantization_bound([r['count'] for r in ordered])),
               'live_plaintext_oracle_executed':False,'separate_toy_protocol_oracle_required':True,
               'fit_rpc_seconds':time.perf_counter()-self.round_start[server_round],
               'wire_bytes_measured':False,'privacy_claim_validated':False}
        self.fit_checks.append(audit);self.params=ndarrays_to_parameters(arrays);self.digest=digest;self.last_round=server_round
        load_arrays(self.model,arrays);torch.save(self.model.state_dict(),self.checkpoints/f'round_{server_round:02d}.pt')
        write_json(self.folder/f'aggregate_round_{server_round:02d}.json',audit)
        return self.params,{'participants':3}
    def consume_evaluation(self,r,records,phase):
        # Reuse the exact B arithmetic and validation. Only its printed prefix changes.
        import contextlib,io
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):value=super().consume_evaluation(r,records,phase)
        text=buf.getvalue().replace('[B/','[C/')
        if text:print(text,end='',flush=True)
        return value
