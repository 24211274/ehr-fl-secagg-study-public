"""Independent count/metric consistency checks; no individual C update oracle."""
from __future__ import annotations
import json,math
from pathlib import Path
from .data import batch_sizes,read_csv
from ehr_fedavg.task import cfg
from pilot_core.core import derived_seed

def read(p):return json.loads(Path(p).read_text())

def verify_runtime_report(out,condition,conf,configuration):
    out=Path(out);folder=out/'server_outputs'/condition;s=read(folder/'server_result.json')
    R=conf['rounds'];updates=sum(conf['local_updates_per_round'].values())*R
    if s.get('status')!='PASS' or s.get('configuration')!=configuration or s.get('rounds_completed')!=R or s.get('total_local_optimiser_steps')!=updates or s.get('flower_executed') is not True:raise ValueError('Incomplete full-size server result')
    if s.get('secagg_protocol_executed') is not (configuration=='C') or s.get('test_data_opened') is not False or s.get('main_experiments_frozen') is not False:raise ValueError('Misstated scope/secure execution')
    from mai_runtime_identity import verify_handshake_report,EXPECTED_BUILD_ID
    verify_handshake_report(read(folder/'implementation_handshake.json'))
    if s.get('implementation_build_id')!=EXPECTED_BUILD_ID:raise ValueError('Missing approved implementation identity')
    hist=read(folder/'history.json')['history']
    if [h['round'] for h in hist]!=list(range(R+1)):raise ValueError('Full evaluation sequence incomplete')
    before=s['initial_state_digest']
    for r in range(1,R+1):
        a=read(folder/f'aggregate_round_{r:02d}.json')
        if a.get('status')!='PASS' or a['weights']!=list(conf['training_sessions_per_client'].values()) or a['sum_weights']!=sum(conf['training_sessions_per_client'].values()) or a['local_training_updates']!=sum(conf['local_updates_per_round'].values()):raise ValueError('Wrong full-size aggregate weights/update count')
        if a['input_digest']!=before or hist[r]['model_digest']!=a['output_digest']:raise ValueError('Broken model chain')
        before=a['output_digest']
        if configuration=='C':
            from ehr_secagg.policy import conservative_quantization_bound
            if a.get('per_client_updates_received_by_strategy') is not False or a.get('received_legacy_carriers_are_same_aggregate') is not True or a.get('live_plaintext_oracle_executed') is not False:raise ValueError('C boundary/oracle condition not met')
            if a['a_priori_coordinate_quantization_bound']!=float(conservative_quantization_bound(a['weights'])):raise ValueError('Wrong count-dependent quantization bound')
        elif a.get('max_tolerance_fraction',float('inf'))>1:raise ValueError('B independent aggregate check did not pass')
    if before!=s['final_state_digest']:raise ValueError('Wrong final model digest')
    minimum=min(h['validation_loss'] for h in hist[1:])
    best=next(h for h in hist[1:] if h['validation_loss']<=minimum+conf['checkpoint_tie_tolerance'])
    if best['round']!=s['selected_round']:raise ValueError('Wrong selected checkpoint')
    for c in conf['clients']:
        d=out/'client_outputs'/condition/c;n=conf['training_sessions_per_client'][c];nv=conf['validation_sessions_per_client'][c];sizes=batch_sizes(n)
        membership=read_csv(out/'client_data'/c/'membership.csv')
        expect_ids={p:{x['session_id'] for x in membership if x['partition']==p} for p in ['training','validation']}
        for r in range(1,R+1):
            f=read(d/f'fit_round_{r:02d}.json')
            if f.get('status')!='PASS' or f['draws']!=n or f['steps']!=len(sizes) or f['fresh_optimizer'] is not True:raise ValueError('Wrong client round budget')
            actual=f['batch_sizes'] if configuration=='C' else [b['sessions'] for b in f['batch_history']]
            if actual!=sizes:raise ValueError('Partial minibatch missing, padded or dropped')
            seed=f['training_seed'] if configuration=='C' else f['seed']
            if seed!=derived_seed(conf['training_seed'],'client-fit',c,r):raise ValueError('Client fit seed changed')
            if configuration=='C' and (f.get('range_check_passed') is not True or any(k in f for k in ['output_digest','batch_history','loss','gradients'])):raise ValueError('C fit export unsafe or range not checked')
            if f['input_digest']!=hist[r-1]['model_digest']:raise ValueError('Stale client input')
        for phase,r in [('initial',0)]+[('round',r) for r in range(1,R+1)]+[('selected',s['selected_round'])]:
            v=read(d/f'evaluate_{phase}_{r:02d}.json')
            if v.get('status')!='PASS' or v['test_data_opened'] or not v['weights_unchanged'] or not v['gradients_unchanged']:raise ValueError('Invalid evaluation scope/state')
            if len(v['session_losses'])!=n+nv or v['train_count']!=n or v['validation_count']!=nv:raise ValueError('Wrong full-size evaluation membership counts')
            for part,count,key in [('training',n,'train_loss'),('validation',nv,'validation_loss')]:
                rows=[x for x in v['session_losses'] if x['partition']==part]
                if len(rows)!=count or {x['session_id'] for x in rows}!=expect_ids[part] or any(x['client']!=c for x in rows):raise ValueError('Evaluation records do not match fixed full membership')
                if any(not math.isfinite(x['loss']) or x['loss']<0 for x in rows):raise ValueError('Invalid loss')
                mean=math.fsum(x['loss'] for x in rows)/count
                if abs(mean-v[key])>1e-10 or abs(hist[r]['per_client'][c][key]-v[key])>1e-7:raise ValueError('Local/global metric mismatch')
    for h in hist:
        for metric,countkey in [('train_loss','train_count'),('validation_loss','validation_count')]:
            n=sum(h['per_client'][c][countkey] for c in conf['clients'])
            v=math.fsum(h['per_client'][c][metric]*h['per_client'][c][countkey] for c in conf['clients'])/n
            if abs(v-h[metric])>1e-12:raise ValueError('Unequal client-weighted loss calculation wrong')
    if configuration=='C':
        from flwr.common.secure_aggregation.secaggplus_constants import Stage
        st=read(folder/'protocol_stages.json')['stages']
        if len(st)!=4*R:raise ValueError('Missing secure protocol stage')
        for r in range(1,R+1):
            rows=[x for x in st if x['round']==r]
            if [x['stage'] for x in rows]!=list(Stage.all()) or any(x['participants']!=3 or x['status']!='PASS' for x in rows):raise ValueError('Invalid secure stage/cohort')
        exchanges=read(folder/'grid_exchanges.json')['exchanges']
        tr=[x for x in exchanges if x.get('secagg_stage') is not None]
        if len(tr)!=4*R or any(x.get('plaintext_training_array_records_observed')!=0 or x.get('content_values_logged') is not False for x in tr):raise ValueError('C wire boundary record missing')
    return s


def paired_review(root):
    root=Path(root);conf=cfg();results={}
    from central_smoke.identity import sha256_file
    for name in ['full_membership.csv','full_client_draws_five_rounds.csv']:
        if sha256_file(root/'B/reports'/name)!=sha256_file(root/'C/reports'/name):raise ValueError('B/C manifest mismatch')
    for co in conf['conditions']:
        b=read(root/'B/server_outputs'/co/'server_result.json');c=read(root/'C/server_outputs'/co/'server_result.json')
        if b['initial_state_digest']!=c['initial_state_digest'] or b['rounds_completed']!=c['rounds_completed'] or b['total_local_optimiser_steps']!=c['total_local_optimiser_steps']:raise ValueError('B/C unmatched initial state or budget')
        for client in conf['clients']:
            for r in range(1,conf['rounds']+1):
                fb=read(root/'B/client_outputs'/co/client/f'fit_round_{r:02d}.json');fc=read(root/'C/client_outputs'/co/client/f'fit_round_{r:02d}.json')
                if fb['seed']!=fc['training_seed'] or fb['draws']!=fc['draws'] or fb['steps']!=fc['steps']:raise ValueError('B/C local pairing mismatch')
        results[co]={'B_final_routine_validation_loss':b['final_validation_loss'],'C_final_routine_validation_loss':c['final_validation_loss'],'C_minus_B':c['final_validation_loss']-b['final_validation_loss'],'loss_difference_acceptance_threshold':None,'detection_comparison_performed':False}
    return {'status':'PASS','conditions':results,'matched_fullsize_counts_schedules_initial_state_and_seeds':True,'tuning_performed':False,'main_experiments_frozen':False,'privacy_claim_validated':False,'timing_overhead_estimate':False}
