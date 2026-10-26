"""Payload-shape and scalar-policy guards. Never write masks, shares or secrets."""
from __future__ import annotations
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS,Key,Stage
from .policy import validate_public_fit_metrics


def check_masked_reply(content,stage,toy=False):
    if any(len(record)>0 for record in content.array_records.values()):
        raise RuntimeError('A plaintext ArrayRecord escaped the SecAgg modifier')
    config=content.config_records.get(RECORD_KEY_CONFIGS)
    if config is None:raise ValueError('Missing SecAgg response record')
    keys={
        Stage.SETUP:{Key.PUBLIC_KEY_1,Key.PUBLIC_KEY_2},
        Stage.SHARE_KEYS:{Key.DESTINATION_LIST,Key.CIPHERTEXT_LIST},
        Stage.COLLECT_MASKED_VECTORS:{Key.MASKED_PARAMETERS},
        Stage.UNMASK:{Key.NODE_ID_LIST,Key.SHARE_LIST},
    }
    if stage not in keys or set(config)!=keys[stage]:raise ValueError('Unexpected SecAgg response fields')
    allowed_configs={RECORD_KEY_CONFIGS}
    allowed_metrics=set();allowed_arrays=set()
    if stage==Stage.COLLECT_MASKED_VECTORS:
        allowed_configs|={'fitres.metrics','fitres.status'}
        allowed_metrics={'fitres.num_examples'};allowed_arrays={'fitres.parameters'}
        validate_public_fit_metrics(dict(content.config_records['fitres.metrics']),toy=toy)
        if set(content.metric_records['fitres.num_examples'])!={'num_examples'}:raise ValueError('Extra metric fields')
        count=content.metric_records['fitres.num_examples']['num_examples']
        if type(count)!=int or count<1 or (not toy and count!=content.config_records['fitres.metrics']['draws']):raise ValueError('Wrong public count')
        status=content.config_records['fitres.status']
        if set(status)!={'code','message'} or status['code']!=0 or status['message'] not in ('','Success'):
            raise ValueError('Unexpected status channel; no free-text diagnostic payload')
        masked=config[Key.MASKED_PARAMETERS]
        if not isinstance(masked,list) or not masked or not all(isinstance(v,bytes) for v in masked):raise ValueError('Missing masked parameter byte arrays')
    if set(content.config_records)!=allowed_configs or set(content.metric_records)!=allowed_metrics or set(content.array_records)!=allowed_arrays:
        raise ValueError('Unexpected response side channel')
    return {'stage':stage,'populated_array_records':0,'reply_schema_pass':True}
