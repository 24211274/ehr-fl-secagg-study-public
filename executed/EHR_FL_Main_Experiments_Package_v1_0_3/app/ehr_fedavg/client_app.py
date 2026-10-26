"""Flower 1.22 NumPyClient adapter; only this client's local routine data are loaded."""
from mai_runtime_identity import implementation_identity_mod
from mai_measurement import timing_boundary_mod
from flwr.client import ClientApp,NumPyClient
from flwr.common import Context
from .task import LocalTask

class EHRClient(NumPyClient):
    def __init__(self,task):self.task=task
    def fit(self,parameters,config):return self.task.fit(parameters,config)
    def evaluate(self,parameters,config):return self.task.evaluate(parameters,config)
    def get_parameters(self,config):raise RuntimeError('Global initial parameters must be server-defined')

def client_fn(context:Context):
    task=LocalTask(context.run_config['runtime-root'],context.node_config['client-id'],context.run_config['condition'])
    return EHRClient(task).to_client()

# No secaggplus_mod in B. Do not advertise B diagnostic metrics as private.
app=ClientApp(client_fn=client_fn,mods=[timing_boundary_mod,implementation_identity_mod])
