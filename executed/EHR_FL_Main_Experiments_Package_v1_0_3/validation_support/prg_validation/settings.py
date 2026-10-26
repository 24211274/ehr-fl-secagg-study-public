"""Only public numeric settings; old C stage flags are NOT inherited as approval."""
from .build import manifest

def scfg():
    return {'secagg':manifest()['secagg_parameters'],'unweighted_parameter_absolute_limit':8.0}
