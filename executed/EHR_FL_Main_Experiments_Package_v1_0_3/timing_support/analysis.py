"""Validate actual counters and form descriptive paired timing summaries.

Rounds within one run are dependent. Three median-based paired estimates are
reported; no p-value, independence claim, equivalence test or precision target.
"""
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path
from collections import Counter
from central_smoke.identity import write_json

MEASURED=(2,3,4,5,6)
STAGES=('setup','share_keys','collect_masked_vectors','unmask')

def read(p):return json.loads(Path(p).read_text())

def overhead(b,c):
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (b,c)):
        raise ValueError('Finite numerical observations required')
    return None if b<=0 else 100.*(c-b)/b

def descriptive(values):
    if not values or any(not math.isfinite(x) for x in values):raise ValueError('Invalid observations')
    mean=statistics.mean(values)
    sd=statistics.stdev(values) if len(values)>1 else None
    return {'n':len(values),'mean':mean,'median':statistics.median(values),
            'sample_sd':sd,'minimum':min(values),'maximum':max(values),
            'coefficient_of_variation_percent':None if sd is None or mean<=0 else 100.*sd/mean}

def check_byte_row(row):
    if not isinstance(row,dict) or set(row)!= {'array_bytes','metric_bytes','config_bytes','total_bytes'}:
        raise ValueError('Incomplete content byte breakdown')
    if any(type(x) is not int or x<0 for x in row.values()):raise ValueError('Byte counts must be nonnegative integers')
    if sum(row[k] for k in ('array_bytes','metric_bytes','config_bytes'))!=row['total_bytes']:
        raise ValueError('Byte total does not match breakdown')

def reconcile_boundary(server,clients,configuration,rounds=6):
    """Compare duplicated boundary counters, then sum ONLY client-side records."""
    if configuration not in ('B','C'):raise ValueError('Unknown configuration')
    sr=[m for e in server['exchanges'] for m in e['messages']]
    sm={r['request_id']:r for r in sr};cm={r['request_id']:r for r in clients}
    if len(sm)!=len(sr) or len(cm)!=len(clients) or not sm or set(sm)!=set(cm):
        raise ValueError('Missing, duplicate or unmatched counted messages')
    stages=('fit',) if configuration=='B' else STAGES
    expected=Counter()
    for c in ('C1','C2','C3'):
        expected[(c,0,'identity','handshake')]=2
        expected[(c,0,'evaluate','initial')]=1
        for r in range(1,rounds+1):
            for stage in stages:expected[(c,r,stage,'fit')]=1
            expected[(c,r,'evaluate','round')]=1
    # selected checkpoint may be any trained round; require exactly one per client
    counts=Counter();selected=[]
    for key,row in cm.items():
        s=sm[key]
        if row.get('status')!='PASS' or row.get('record_values_written') is not False or row.get('wire_bytes_measured') is not False:
            raise ValueError('Unsafe/incomplete client counter row')
        for field in ['round','stage','phase','message_type','node_id']:
            if row[field]!=s[field]:raise ValueError('Server/client descriptor mismatch')
        check_byte_row(row['incoming_record_bytes']);check_byte_row(row['outgoing_record_bytes'])
        if row['incoming_record_bytes']!=s['outgoing_record_bytes'] or row['outgoing_record_bytes']!=s['incoming_record_bytes']:
            raise ValueError('Client/server byte counts differ at the same boundary')
        if row['phase']=='selected':
            if row['stage']!='evaluate' or row['round'] not in range(1,rounds+1):raise ValueError('Invalid selection measurement')
            selected.append(row['client'])
        else:counts[(row['client'],row['round'],row['stage'],row['phase'])]+=1
        for k in ('callback_wall_ns','callback_cpu_ns','byte_counter_wall_ns'):
            if type(row[k]) is not int or row[k]<0:raise ValueError('Invalid client timer')
        if row['stage'] in ('fit','collect_masked_vectors'):
            fit=[x for x in row['local_spans'] if x['name']=='local_fit_with_checks']
            load=[x for x in row['local_spans'] if x['name']=='data_load_with_checks']
            if len(fit)!=1 or len(load)!=1:raise ValueError('Client fit/data-load timing incomplete')
        for sp in row['local_spans']:
            if sp['completed'] is not True or type(sp['wall_ns']) is not int or sp['wall_ns']<0 or type(sp['cpu_ns']) is not int or sp['cpu_ns']<0:
                raise ValueError('Incomplete/invalid local span')
    if counts!=expected or Counter(selected)!=dict.fromkeys(('C1','C2','C3'),1):
        raise ValueError('Timing message cohort or phase counts differ from the plan')
    return {'status':'PASS','matched_messages':len(sr),'client_total_incoming_record_bytes':sum(r['incoming_record_bytes']['total_bytes'] for r in clients),
            'client_total_outgoing_record_bytes':sum(r['outgoing_record_bytes']['total_bytes'] for r in clients),
            'server_totals_added':False,'wire_bytes_measured':False}


def extract_cell(out,configuration,pair):
    out=Path(out);server=read(out/'server_outputs/reference/measurement_server.json')
    clients=[read(p) for p in sorted((out/'measurement_client/reference').rglob('*.json'))]
    check=reconcile_boundary(server,clients,configuration)
    events=server['events'];fit={e['round']:e for e in events if e['name']=='fit_workflow'}
    val={e['round']:e for e in events if e['name']=='evaluation_workflow'}
    outer={e['round']:e for e in events if e['name']=='outer_round_including_evaluation'}
    for name,rows in [('fit_workflow',fit),('evaluation_workflow',val),('outer_round_including_evaluation',outer)]:
        if set(rows)!=set(range(1,7)) or len([e for e in events if e['name']==name])!=6:raise ValueError('Incomplete/duplicate server timing spans')
    for e in events:
        if e['completed'] is not True or type(e['wall_ns']) is not int or e['wall_ns']<0 or e['end_ns']-e['start_ns']!=e['wall_ns']:
            raise ValueError('Invalid/incomplete server interval')
    rows=[]
    for r in range(1,7):
        fi=[m for m in clients if m['round']==r and m['phase']=='fit']
        ev=[m for m in clients if m['round']==r and m['phase']=='round']
        local={m['client']:next(x for x in m['local_spans'] if x['name']=='local_fit_with_checks')
               for m in fi if m['stage'] in ('fit','collect_masked_vectors')}
        if set(local)!={'C1','C2','C3'}:raise ValueError('Missing fit spans')
        if outer[r]['wall_ns']<fit[r]['wall_ns']+val[r]['wall_ns']:raise ValueError('Outer interval smaller than included phases')
        row={'pair':pair,'configuration':configuration,'round':r,'warmup':r==1,
             'fit_workflow_wall_ns':fit[r]['wall_ns'],'evaluation_workflow_wall_ns':val[r]['wall_ns'],
             'outer_round_wall_ns':outer[r]['wall_ns'],
             'fit_record_bytes_down':sum(m['incoming_record_bytes']['total_bytes'] for m in fi),
             'fit_record_bytes_up':sum(m['outgoing_record_bytes']['total_bytes'] for m in fi),
             'evaluation_record_bytes_down':sum(m['incoming_record_bytes']['total_bytes'] for m in ev),
             'evaluation_record_bytes_up':sum(m['outgoing_record_bytes']['total_bytes'] for m in ev),
             'maximum_client_fit_wall_ns_not_sum':max(x['wall_ns'] for x in local.values()),
             'sum_client_fit_cpu_ns':sum(x['cpu_ns'] for x in local.values()),
             'client_byte_counter_wall_ns_sum':sum(m['byte_counter_wall_ns'] for m in fi+ev)}
        row['fit_record_bytes_total']=row['fit_record_bytes_down']+row['fit_record_bytes_up']
        for c,x in local.items():row[c+'_local_fit_wall_ns']=x['wall_ns'];row[c+'_local_fit_cpu_ns']=x['cpu_ns']
        if configuration=='C':
            se=[e for e in events if e['round']==r and e['name'].startswith('secagg_')]
            if [e['name'] for e in se]!=['secagg_'+x for x in STAGES]:raise ValueError('Secure stage timing incomplete')
            for e in se:row[e['name']+'_wall_ns']=e['wall_ns']
        else:
            for label in ('fedavg_native_aggregate','independent_float_average_audit'):
                e=[e for e in events if e['round']==r and e['name']==label]
                if len(e)!=1:raise ValueError('B core/audit timer missing')
                row[label+'_wall_ns']=e[0]['wall_ns']
        rows.append(row)
    result=read(out/'server_outputs/reference/server_result.json')
    metrics=('fit_workflow_wall_ns','evaluation_workflow_wall_ns','outer_round_wall_ns','fit_record_bytes_total')
    med={k:statistics.median(x[k] for x in rows if not x['warmup']) for k in metrics}
    return {'status':'PASS','pair':pair,'configuration':configuration,'rounds':rows,'boundary_reconciliation':check,
            'measured_rounds':list(MEASURED),'per_run_medians':med,'initial_state_digest':result['initial_state_digest'],
            'final_validation_loss':result['final_validation_loss'],
            'clock':'perf_counter_ns','wire_bytes_measured':False,'instrumented_pipeline':True,
            'no_inference_microbenchmark':True,'main_experiments_frozen':False}


def summarise(cells,plan):
    if [(x['pair'],x['configuration']) for x in cells]!=[(p['pair'],p['configuration']) for p in plan]:
        raise ValueError('Incomplete or reordered repeated measurements')
    if len(cells)!=6 or any(x['status']!='PASS' for x in cells):raise ValueError('Six successful runs required, no selective summaries')
    if len({x['initial_state_digest'] for x in cells})!=1:raise ValueError('Different initial states')
    contrasts=[]
    for pair in range(1,4):
        b=next(x for x in cells if x['pair']==pair and x['configuration']=='B')
        c=next(x for x in cells if x['pair']==pair and x['configuration']=='C')
        if b['measured_rounds']!=list(MEASURED) or c['measured_rounds']!=list(MEASURED):raise ValueError('Warm-up policy changed')
        for metric,bv in b['per_run_medians'].items():
            cv=c['per_run_medians'][metric]
            contrasts.append({'pair':pair,'metric':metric,'B_run_median':bv,'C_run_median':cv,
                              'C_minus_B':cv-bv,'overhead_percent':overhead(bv,cv)})
    across={}
    for metric in cells[0]['per_run_medians']:
        rows=[r for r in contrasts if r['metric']==metric]
        across[metric]={'B':descriptive([r['B_run_median'] for r in rows]),
                        'C':descriptive([r['C_run_median'] for r in rows]),
                        'paired_difference':descriptive([r['C_minus_B'] for r in rows]),
                        'paired_percent':None if any(r['overhead_percent'] is None for r in rows) else descriptive([r['overhead_percent'] for r in rows])}
    return {'status':'PASS','paired_repetitions':3,'measured_rounds_per_run':5,'run_order':[(x['pair'],x['configuration']) for x in cells],
            'contrasts':contrasts,'across_run_descriptives':across,
            'unit_of_repetition':'complete run within a B/C pair; rounds are dependent observations',
            'confidence_intervals_or_significance_tests':False,'performance_acceptance_threshold':None,
            'same_model_training_seed_each_run':True,'wire_bytes_measured':False,
            'inference_microbenchmark_executed':False,'privacy_claim_validated':False,'main_experiments_frozen':False,
            'interpretation':'Conditional instrumented timing/payload pilot at Reference 256/128 per client; NOT full-size deployment cost or final threat-detection evidence.'}


def write_csv(path,rows):
    if not rows:raise ValueError('Empty CSV')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
