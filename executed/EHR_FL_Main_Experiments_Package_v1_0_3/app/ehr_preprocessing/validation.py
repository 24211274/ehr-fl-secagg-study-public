"""Full encoded-array checks and real-data forward/backward smoke checks.

The model probe never invokes optimizer.step, saves no model checkpoint and reports
no threat-detection metrics. It is a data/model interface check, not the overfit pilot.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from pilot_core.model import CompactLSTMAutoencoder, session_losses, set_training_seed, state_digest
from .dataset import SessionDataset, collate_sessions
from .pipeline import approved_schema, CLIENTS, ALLOWED_PARTITIONS, EXPECTED, SCHEMA_SHA
from .io_utils import verify_artefacts, sha256_file


def validate_arrays(root: Path) -> dict:
    files=verify_artefacts(root)
    schema=approved_schema(root/'encoding_schema.json')
    summary=json.loads((root/'preprocessing_summary.json').read_text())
    checks=[]
    all_ids=set(); training_relative=[]
    with (root/'session_manifest.csv').open(encoding='utf-8',newline='') as f:
        sessions=list(csv.DictReader(f))
    for client in CLIENTS:
        for part in ALLOWED_PARTITIONS:
            ds=SessionDataset(root,client,part,verify=False)
            n=len(ds.features)
            check={
                'client':client,'partition':part,'sessions':len(ds),'events':n,
                'expected_counts':(len(ds),n)==EXPECTED[client,part],
                'finite_features':bool(np.isfinite(ds.features).all()),
                'binary_01_only':bool(np.isin(ds.features[:,110:113],[0.0,1.0]).all()),
                'cyclic_unit_pairs':bool(np.allclose(np.square(ds.features[:,113:115]).sum(1),1,rtol=0,atol=2e-7)
                                         and np.allclose(np.square(ds.features[:,115:117]).sum(1),1,rtol=0,atol=2e-7)),
                'categorical_heads_exact':True,'categorical_masks_exact':True,
                'no_duplicate_or_cross_partition_session':not bool(all_ids & set(ds.ids)),
                'positive_lengths_exact_offsets':bool(np.array_equal(np.diff(ds.offsets),ds.lengths) and (ds.lengths>0).all()),
            }
            all_ids.update(ds.ids)
            for j,f in enumerate(schema['categorical_fields']):
                lo,hi=f['one_hot_start_inclusive'],f['one_hot_stop_exclusive']
                block=ds.features[:,lo:hi]
                if not np.isin(block,[0.0,1.0]).all() or not np.all(block.sum(1)==1) or not np.array_equal(block.argmax(1),ds.categories[:,j]):
                    check['categorical_heads_exact']=False
                if not np.array_equal(ds.categorical_valid[:,j],~np.isin(ds.categories[:,j],f['masked_target_indices'])):
                    check['categorical_masks_exact']=False
            rows=[r for r in sessions if r['client']==client and r['partition']==part]
            for r in rows:
                j=int(r['session_index'])
                if int(r['event_offset'])!=ds.offsets[j] or int(r['length'])!=ds.lengths[j]:
                    raise ValueError('Linkage manifest differs from arrays')
            # Walk the entire dataset in unshuffled batches, including partial last batch.
            batches,padded,event_total,last_size=0,0,0,0
            for start in range(0,len(ds),64):
                items=[ds[j] for j in range(start,min(start+64,len(ds)))]
                b=collate_sessions(items); batches+=1;last_size=len(items)
                mask=b['padding_mask'];padded+=int((~mask).sum());event_total+=int(mask.sum())
                if b['categorical_valid'][~mask].any() or torch.count_nonzero(b['x'][~mask]) or not torch.equal(mask.sum(1),b['lengths']):
                    raise ValueError('Invalid padding/masking')
            check.update({'all_sessions_batched_once':event_total==n,'batches':batches,
                          'last_batch_sessions':last_size,'padding_positions_checked':padded})
            boolean_keys=[k for k,v in check.items() if type(v) is bool]
            if not all(check[k] for k in boolean_keys):
                raise ValueError(f'Encoded array validation failed: {client}/{part}: {check}')
            if part=='training': training_relative.append(np.array(ds.features[:,117:119],dtype=np.float64))
            check['status']='PASS';checks.append(check)
    tr=np.concatenate(training_relative,axis=0)
    mean,var=tr.mean(0),tr.var(0,ddof=0)
    scaler=json.loads((root/'relative_time_scaler.json').read_text())
    expected_variance=np.where(np.asarray(scaler['constant_columns']),0.0,1.0)
    if not np.allclose(mean,0,atol=1e-6,rtol=0) or not np.allclose(var,expected_variance,atol=1e-6,rtol=0):
        raise ValueError('Training scaled moments outside float32 tolerance')
    if len(all_ids)!=14455 or summary['selection']['test_feature_rows_encoded']!=0:
        raise ValueError('Unexpected development membership')
    return {'status':'PASS','artefact_files_verified':len(files),'groups':checks,
            'sessions_checked':len(all_ids),'events_checked':sum(c['events'] for c in checks),
            'training_scaled_mean_float64_check':mean.tolist(),'training_scaled_variance_float64_check':var.tolist(),
            'scaled_moment_tolerance':1e-6,'test_data_encoded':False,
            'source_and_schema_identity_verified':True}


def model_probe(root: Path) -> dict:
    schema=approved_schema(root/'encoding_schema.json')
    set_training_seed(20260907)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()!=1:
            raise
    model=CompactLSTMAutoencoder([len(f['categories']) for f in schema['categorical_fields']])
    model.eval()  # padding invariance must not compare different dropout draws
    before=state_digest(model)
    with (root/'pilot_membership/pilot_membership.csv').open(encoding='utf-8',newline='') as f:
        membership=list(csv.DictReader(f))
    reports=[]
    for client in CLIENTS:
        for part in ALLOWED_PARTITIONS:
            ids=[r['session_id'] for r in membership if r['client']==client and r['partition']==part][:64]
            ds=SessionDataset(root,client,part,session_ids=ids,verify=False)
            b=collate_sessions([ds[i] for i in range(len(ds))])
            model.zero_grad(set_to_none=True)
            outputs=model(b['x'],b['lengths'])
            losses,_,mask=session_losses(outputs,b['categorical'],b['binary'],b['temporal'],b['lengths'],b['categorical_valid'])
            if not torch.isfinite(losses).all() or not torch.equal(mask,b['padding_mask']):
                raise ValueError('Nonfinite or invalid data/model interface')
            # Numerical stress probe: finite values only in padded feature slots.
            altered=b['x'].clone();altered[~mask]=17.25
            with torch.no_grad():
                alternative=model(altered,b['lengths'])
                alternative_losses=session_losses(alternative,b['categorical'],b['binary'],b['temporal'],b['lengths'],b['categorical_valid'])[0]
            difference=float(torch.max(torch.abs(losses.detach()-alternative_losses)))
            if not torch.allclose(losses.detach(),alternative_losses,atol=1e-6,rtol=1e-6):
                raise ValueError('Padded inputs alter valid session loss')
            # Gradient availability tested only on routine TRAINING rows.
            gradients=None
            if part=='training':
                losses.mean().backward()
                gradients=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
                if not gradients:
                    raise ValueError('Missing/nonfinite gradients')
            if state_digest(model)!=before:
                raise ValueError('Probe unexpectedly changed model weights')
            reports.append({'client':client,'partition':part,'batch_sessions':len(ds),'batch_shape':list(b['x'].shape),
                            'finite_losses':True,'gradient_check_pass':gradients,
                            'padding_invariance_max_abs_loss_difference':difference,'status':'PASS'})
    return {'status':'PASS','purpose':'real routine-data interface check; no optimiser steps, no checkpoint',
            'input_dim':model.input_dim,'parameter_count':sum(p.numel() for p in model.parameters()),
            'model_source':'unchanged pilot_core/model.py from pilot v1.1',
            'model_mode':'eval for deterministic padding checks; gradients enabled on training probes only',
            'seed':20260907,'state_hash_before':before,'state_hash_after':state_digest(model),
            'weights_unchanged':state_digest(model)==before,'probes':reports,
            'optimiser_steps':0,'full_training_performed':False,'threat_metrics_computed':False,
            'flower_executed':False,'secagg_protocol_executed':False}
