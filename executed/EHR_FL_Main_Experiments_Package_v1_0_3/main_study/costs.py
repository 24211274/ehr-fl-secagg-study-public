"""Main per-run system costs: include all planned rounds; no timing-pilot warm-up deletion."""
from __future__ import annotations
import statistics
from pathlib import Path
from .common import read,write,csv_write
from timing_support.analysis import reconcile_boundary


def extract(job:Path,condition:str,configuration:str,rounds:int)->dict:
    job=Path(job);server=read(job/f'server_outputs/{condition}/measurement_server.json')
    clients=[read(p) for p in sorted((job/'measurement_client'/condition).rglob('*.json'))]
    proof=reconcile_boundary(server,clients,configuration,rounds=rounds)
    ev=server['events'];rows=[]
    for e in ev:
        if not e['completed'] or type(e['wall_ns']) is not int or e['wall_ns']<0 or e['end_ns']-e['start_ns']!=e['wall_ns']:
            raise ValueError('Invalid measured interval')
    for r in range(1,rounds+1):
        durations={}
        for label in ('fit_workflow','evaluation_workflow','outer_round_including_evaluation'):
            found=[e for e in ev if e['round']==r and e['name']==label]
            if len(found)!=1:raise ValueError('Incomplete/repeated main interval')
            durations[label+'_wall_ns']=found[0]['wall_ns']
        fit=[m for m in clients if m['round']==r and m['phase']=='fit']
        evaluation=[m for m in clients if m['round']==r and m['phase']=='round']
        row={'round':r,**durations}
        for label,parts in [('fit',fit),('evaluation',evaluation)]:
            row[label+'_record_bytes_down']=sum(m['incoming_record_bytes']['total_bytes'] for m in parts)
            row[label+'_record_bytes_up']=sum(m['outgoing_record_bytes']['total_bytes'] for m in parts)
        rows.append(row)
    keys=[k for k in rows[0] if k!='round']
    totals={k:sum(r[k] for r in rows) for k in keys}
    totals['fit_record_bytes_total']=totals['fit_record_bytes_down']+totals['fit_record_bytes_up']
    host=read(job/f'reports/{condition}/host_measurements.json')
    phases=read(job/f'reports/{condition}/host_phase_timings.json')
    result={'status':'PASS','configuration':configuration,'condition':condition,'rounds':rounds,
        'totals_all_rounds':totals,'first_round_retained':True,'warmup_rounds_excluded':[],
        'secondary_round_medians':{k:statistics.median(r[k] for r in rows) for k in keys},
        'host':host,'host_phase_timings':phases,'boundary_reconciliation':proof,
        'wire_bytes_measured':False,'pure_crypto_time_claim':False,'unit':'one complete training run per seed/condition',
        'B_C_diagnostic_paths_not_identical':True}
    write(job/'reports/main_cost_summary.json',result);csv_write(job/'reports/main_round_costs.csv',rows)
    return result


def summarise_cost_cells(cells:list[dict])->dict:
    """All-round totals, paired by seed; no confidence interval for these costs."""
    from .common import SEEDS
    from .evaluation import summaries,paired_summary
    expected={(m,co,s) for m in 'ABC' for co in ('reference','noniid') for s in SEEDS}
    keys=[(r['configuration'],r['condition'],r['seed']) for r in cells]
    if len(keys)!=30 or set(keys)!=expected:raise ValueError('All 30 unique planned cost cells required')
    by=dict(zip(keys,cells));out={}
    for co in ('reference','noniid'):
        out[co]={}
        for metric in ('instrumented_fit_seconds','evaluation_seconds','application_fit_record_bytes',
                       'inference_single_median_ms','inference_single_p95_ms','inference_batch_sessions_per_second'):
            values={m:{s:by[(m,co,s)][metric] for s in SEEDS} for m in 'ABC'}
            ratios=[]
            for seed in SEEDS:
                b,c=values['B'][seed],values['C'][seed]
                ratios.append(100*(c-b)/b if b is not None and c is not None and b>0 else None)
            out[co][metric]={'by_method':{m:summaries([values[m][s] for s in SEEDS]) for m in 'ABC'},
                            'C_minus_B':paired_summary(values['B'],values['C'],interval=False),
                            'C_vs_B_percent_per_seed':summaries(ratios),'seed_order':list(SEEDS)}
    return {'status':'PASS','cells':cells,'summary':out,'warmup_rounds_removed':False,
            'wire_bytes_measured':False,'pure_crypto_time_claim':False,
            'centralised_network_volume':None,'intervals_computed':False,
            'inference_order_not_counterbalanced':True,
            'interpretation':'Run-level engineering costs; inference uses fixed validation inputs, not arrival-to-alert latency.'}


def campaign_summary(campaign:Path,models:list[dict])->dict:
    cells=[]
    for row in models:
        job=Path(campaign)/row['job'];inf=read(job/'reports/selected_model_inference.json')
        m=row['configuration']
        if m=='A':
            raw=read(job/'reports/central_summary.json')
            fit=raw['fit_wall_ns_total']/1e9;ev=raw['evaluation_wall_seconds_total'];volume=None
            start_to_end=raw['elapsed_seconds']
        else:
            raw=read(job/'reports/main_cost_summary.json');total=raw['totals_all_rounds']
            fit=total['fit_workflow_wall_ns']/1e9;ev=total['evaluation_workflow_wall_ns']/1e9
            volume=total['fit_record_bytes_total'];start_to_end=raw['host']['elapsed_seconds']
        cells.append({'configuration':m,'condition':row['condition'],'seed':row['seed'],
            'instrumented_fit_seconds':fit,'evaluation_seconds':ev,
            'application_fit_record_bytes':volume,'elapsed_seconds_scope_differs_A_vs_BC':start_to_end,
            'inference_single_median_ms':inf['median_ms'],
            'inference_single_p95_ms':inf['p95_ms'],
            'inference_batch_sessions_per_second':inf['throughput_sessions_per_second']})
    return summarise_cost_cells(cells)
