"""Separate public-toy protocol check using installed Flower, no benchmark inputs.

This validation module is not a benchmark ServerApp/ClientApp.
The pinned PRG build is selected explicitly; workflow mathematics remain upstream. The in-memory
transport holds all toy parties and is not a deployment or privacy experiment.
"""
from __future__ import annotations
import copy,hashlib,importlib,importlib.metadata,inspect
import numpy as np
from .settings import scfg
from .policy import conservative_quantization_bound


def verify_toy_completion(aggregate_calls, observed_stages, expected_stages):
    """Check public workflow metadata, independent of list/tuple representation.

    Flower 1.22.0 Stage.all() returns a tuple whereas ToyGrid records a list.
    Compare ordered elements after normalisation, NOT sets or sorted elements.
    Missing, repeated, additional and reordered stages must still be rejected.
    Only public stage names/counts enter diagnostics; no masks, keys or shares.
    """
    if type(aggregate_calls) is not int or aggregate_calls != 1:
        raise ValueError(
            f'Toy workflow requires exactly one aggregate callback; observed={aggregate_calls!r}'
        )
    for name, stages in (("observed", observed_stages), ("expected", expected_stages)):
        if not isinstance(stages, (list, tuple)):
            raise ValueError(f'{name} toy stages must be an ordered list or tuple')
        if len(stages) != 4 or any(type(stage) is not str for stage in stages):
            raise ValueError(f'{name} toy stages must contain four string stage names; '
                             f'observed={observed_stages!r}; expected={expected_stages!r}')
    if len(set(expected_stages)) != 4:
        raise ValueError('Expected toy stage names must be distinct')
    if tuple(observed_stages) != tuple(expected_stages):
        raise ValueError(f'Toy stage sequence mismatch; observed={observed_stages!r}; '
                         f'expected={expected_stages!r}')
    return {
        'status': 'PASS',
        'aggregate_calls': aggregate_calls,
        'observed_stages': list(observed_stages),
        'expected_stages': list(expected_stages),
        'observed_sequence_type': type(observed_stages).__name__,
        'expected_sequence_type': type(expected_stages).__name__,
        'ordered_four_stages_match': True,
        'sequence_container_normalised_only': True,
    }


def run_case(weights,lattice=True,drop=False,mode="candidate",mismatch=False):
    from types import SimpleNamespace
    from flwr.client import ClientApp,NumPyClient
    from flwr.client.mod import secaggplus_mod
    from flwr.common import Context,RecordDict,ConfigRecord,FitIns,ndarrays_to_parameters,parameters_to_ndarrays,bytes_to_ndarray
    from flwr.server import Grid,LegacyContext,ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.server.workflow import SecAggPlusWorkflow
    from flwr.server.workflow.constant import MAIN_PARAMS_RECORD,MAIN_CONFIGS_RECORD,Key as WK
    from flwr.common import recorddict_compat as compat
    from flwr.common.secure_aggregation.secaggplus_constants import Stage,Key,RECORD_KEY_CONFIGS
    from flwr.common.secure_aggregation.crypto.shamir import combine_shares
    from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen
    from .wire import check_masked_reply
    from .policy import validate_cohort
    from .identity import binding_identity,guarded_client_call,require_id,ID_RECORD,IdentityMismatch
    implementation=binding_identity(mode)['build_id']
    class MissingPublicReply(RuntimeError):pass
    conf=scfg()['secagg'];nids=[11,22,33]
    values=[np.array([-1.,0.,1.,2.],dtype=np.float32),np.array([2.,1.,-2.,0.],np.float32),np.array([0.,3.,1.,-1.],np.float32)]
    if not lattice:values=[a+np.array([.001,.023,-.041,.079],np.float32)*(i+1) for i,a in enumerate(values)]
    expected=sum((a.astype(np.float64)*w/sum(weights) for a,w in zip(values,weights)),np.zeros(4,np.float64))
    public_input='0'*64
    class ToyClient(NumPyClient):
        def __init__(self,i):self.i=i
        def fit(self,parameters,config):
            return [values[self.i].copy()],weights[self.i],{'client':f'C{self.i+1}','input_digest':public_input,'steps':0,'draws':weights[self.i]}
    ca=ClientApp(client_fn=lambda context:ToyClient(int(context.node_config['toy-index'])).to_client(),mods=[secaggplus_mod])
    contexts={n:Context(run_id=101,node_id=n,node_config={'toy-index':i},state=RecordDict(),run_config={}) for i,n in enumerate(nids)}
    class ToyStrategy(FedAvg):
        def __init__(self):
            super().__init__(initial_parameters=ndarrays_to_parameters([np.zeros(4,np.float32)]),fit_metrics_aggregation_fn=lambda _: {})
            self.result=None;self.calls=0
        def configure_fit(self,server_round,parameters,client_manager):
            return [(SimpleNamespace(node_id=n,cid=str(n)),FitIns(parameters,{})) for n in nids]
        def aggregate_fit(self,server_round,results,failures):
            if failures or len(results)!=3:raise RuntimeError('Missing toy participant')
            arrs=[parameters_to_ndarrays(res.parameters)[0] for _,res in results]
            if any(not np.array_equal(arrs[0],a) for a in arrs):raise ValueError('Toy aggregate carriers differ')
            self.result=arrs[0].copy();self.calls+=1
            return ndarrays_to_parameters([self.result]),{}
    strategy=ToyStrategy()
    context=LegacyContext(context=Context(run_id=101,node_id=0,node_config={},state=RecordDict(),run_config={}),
                          config=ServerConfig(num_rounds=1),strategy=strategy)
    context.state.array_records[MAIN_PARAMS_RECORD]=compat.parameters_to_arrayrecord(ndarrays_to_parameters([np.zeros(4,np.float32)]),True)
    context.state.config_records[MAIN_CONFIGS_RECORD]=ConfigRecord({WK.CURRENT_ROUND:1})
    class ToyGrid(Grid):
        def __init__(self):self.stage_names=[];self.masked=[];self.shares={n:[] for n in nids}
        @property
        def run(self):return SimpleNamespace(run_id=101)
        def set_run(self,run_id):raise RuntimeError('Unused toy transport operation')
        def create_message(self,*a,**k):raise RuntimeError('Unused toy transport operation')
        def get_node_ids(self):return nids
        def push_messages(self,messages):raise RuntimeError('Unused toy transport operation')
        def pull_messages(self,message_ids):raise RuntimeError('Unused toy transport operation')
        def send_and_receive(self,messages,*,timeout=None):
            sent=list(messages);stage=sent[0].content.config_records[RECORD_KEY_CONFIGS][Key.STAGE]
            self.stage_names.append(stage);replies=[]
            for msg in sent:
                if drop and stage==Stage.SHARE_KEYS and msg.metadata.dst_node_id==33:continue
                # Toy parties must not mutate a shared request ConfigRecord instance.
                incoming=copy.deepcopy(msg)
                incoming.content.config_records[ID_RECORD]=ConfigRecord({'build_id':implementation if not (mismatch and msg.metadata.dst_node_id==33) else 'deliberately-wrong-public-build-id'})
                reply=guarded_client_call(ca,incoming,contexts[msg.metadata.dst_node_id],mode)
                identity=reply.content.config_records.pop(ID_RECORD,None) if not reply.has_error() else None
                require_id(dict(identity) if identity is not None else None,implementation)
                if reply.has_error():raise RuntimeError('Toy client returned an error')
                check_masked_reply(reply.content,stage,toy=True)
                if stage==Stage.COLLECT_MASKED_VECTORS:
                    self.masked.append([bytes_to_ndarray(b).copy() for b in reply.content.config_records[RECORD_KEY_CONFIGS][Key.MASKED_PARAMETERS]])
                elif stage==Stage.UNMASK:
                    c=reply.content.config_records[RECORD_KEY_CONFIGS]
                    for nid,share in zip(c[Key.NODE_ID_LIST],c[Key.SHARE_LIST]):self.shares[nid].append(share)
                replies.append(reply)
            try:validate_cohort([m.metadata.dst_node_id for m in sent],[m.metadata.src_node_id for m in replies])
            except RuntimeError as e:raise MissingPublicReply('Public toy reply missing; abort') from e
            return replies
    grid=ToyGrid();saved_np=np.random.get_state()
    try:
        # Only public toy quantisation is seeded; secrets still come from stock OS RNG.
        np.random.seed(177)
        if mismatch:
            try:SecAggPlusWorkflow(**conf)(grid,context)
            except IdentityMismatch:
                if strategy.calls!=0:raise ValueError('Mismatch produced an aggregate')
                return {'status':'PASS','negative_case':'mixed public identity refused before training','aggregate_calls':0,'benchmark_data_read':False,'deployment_engine_used':False}
            raise ValueError('Mismatched identity was not rejected')
        if drop:
            try:SecAggPlusWorkflow(**conf)(grid,context)
            except MissingPublicReply:
                if strategy.calls!=0:raise ValueError('Dropout unexpectedly produced an aggregate')
                return {'status':'PASS','negative_case':'missing toy reply refused','aggregate_calls':0,'benchmark_data_read':False}
            raise ValueError('Missing toy reply was not rejected')
        SecAggPlusWorkflow(**conf)(grid,context)
        completion=verify_toy_completion(strategy.calls,grid.stage_names,Stage.all())
        error=float(np.max(np.abs(strategy.result-expected)))
        bound=conservative_quantization_bound(weights)
        if error>bound:raise ValueError('Toy dequantized average exceeded predeclared numeric bound')
        integer_check=None
        if lattice:
            # Independent INTEGER recovery for PUBLIC toy vectors only, never live C data.
            total=[sum((x[j].astype(np.int64) for x in grid.masked),np.zeros_like(grid.masked[0][j],dtype=np.int64)) for j in range(2)]
            for n in nids:
                private=pseudo_rand_gen(combine_shares(grid.shares[n]),conf['modulus_range'],[a.shape for a in total])
                total=[(a-b)%conf['modulus_range'] for a,b in zip(total,private)]
            qratios=[round(w/conf['max_weight']*conf['quantization_range']) for w in weights]
            grids=[(a*(qr/conf['quantization_range'])+conf['clipping_range'])*(conf['quantization_range']/(2*conf['clipping_range'])) for a,qr in zip(values,qratios)]
            if any(not np.array_equal(a,np.rint(a)) for a in grids):raise ValueError('Toy fixture is not exactly on quantizer lattice')
            expected_integer=[np.array([sum(qratios)],np.int64),sum((a.astype(np.int64) for a in grids),np.zeros(4,np.int64))]
            if any(not np.array_equal(a,b) for a,b in zip(total,expected_integer)):raise ValueError('Public toy integer sum mismatch')
            integer_check=True
        # NumPy may keep the computed bound as float32. Convert ONLY its
        # diagnostic representation; the comparison above remains unchanged.
        return {'status':'PASS','public_weights':weights,'lattice_fixture':lattice,'exact_integer_sum_pass':integer_check,
                'max_abs_average_error':error,'predeclared_error_bound':float(bound),'stages':grid.stage_names,
                'aggregate_calls':strategy.calls,'completion_check':completion,'benchmark_data_read':False,'deployment_engine_used':False,
                'upstream_workflow_and_client_mod_executed':True,'prg_build_mode':mode,'public_identity_checked_each_stage':True,'transport':'in_memory_public_toy_only',
                'no_keys_masks_or_shares_exported':True,'privacy_claim_validated':False}
    finally:
        np.random.set_state(saved_np)
        contexts.clear();grid.masked.clear();grid.shares.clear()


def run_oracle(mode="candidate"):
    cases=[run_case([256,256,256],mode=mode),run_case([64,128,256],mode=mode),
           run_case([256,256,256],False,mode=mode),run_case([256,256,256],drop=True,mode=mode),
           run_case([256,256,256],mode=mode,mismatch=True)]
    return {'status':'PASS','cases':cases,'public_protocol_executed':True,'mode':mode,
            'benchmark_data_read':False,'deployment_engine_used':False,'privacy_claim_validated':False,
            'no_keys_masks_or_shares_exported':True}
