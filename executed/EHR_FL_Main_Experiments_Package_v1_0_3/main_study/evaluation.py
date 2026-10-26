"""Predeclared validation-only calibration and fixed-benchmark paired-seed metrics.

This module performs no file discovery, model selection, tuning, or dataset read.
Final-test access is exclusively in the separately authorised evaluator entry point.
"""
from __future__ import annotations
import math, statistics
from collections import Counter
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import t
from .common import CLIENTS, SEEDS


def finite_scores(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Nonempty finite one-dimensional scores required')
    return x


def calibrate_validation(rows:list[dict],expected_ids:set[str]) -> dict:
    if len(rows)!=len(expected_ids) or {r['session_id'] for r in rows}!=expected_ids:
        raise ValueError('Calibration must cover every distinct expected validation session exactly once')
    if any(r['partition']!='validation' or r['client'] not in CLIENTS for r in rows):
        raise ValueError('Calibration refuses non-validation data')
    out={}
    for c in CLIENTS:
        selected=[r for r in rows if r['client']==c]
        x=finite_scores([r['loss'] for r in selected])
        if np.any(x<0):raise ValueError('Negative reconstruction loss')
        centre=float(np.median(x));mad=float(np.median(np.abs(x-centre)))
        scale=1.482602218505602*mad;kind='MAD'
        if scale<=1e-12:
            q25,q75=np.quantile(x,[.25,.75],method='linear')
            scale=float(q75-q25)/1.3489795003921634;kind='IQR'
        if scale<=1e-12:scale=1.;kind='unit_fallback'
        thresholds={str(q):float(np.quantile(x,q,method='linear')) for q in (.99,.98,.995)}
        out[c]={'sessions':len(x),'median':centre,'scale':scale,'scale_method':kind,
                'degenerate_scale_flag':kind=='unit_fallback','thresholds':thresholds,
                'calibrated_thresholds':{k:(v-centre)/scale for k,v in thresholds.items()},
                'quantile_method':'linear','alert_operator':'>','fit_partition':'validation'}
    return {'version':'1.0.0','fit_partition':'validation','clients':out,
            'test_data_used':False,'primary_quantile':.99,'nominal_FPR_guaranteed':False}


def binary_metrics(labels,scores,predictions) -> dict:
    y=np.asarray(labels);s=finite_scores(scores);p=np.asarray(predictions)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(s) or len(p)!=len(y) or not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Aligned binary labels/predictions required')
    y=y.astype(bool);p=p.astype(bool)
    tp=int((y&p).sum());tn=int((~y&~p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
    pos=tp+fn;neg=tn+fp;both=pos>0 and neg>0
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/pos if pos else None
    # F1 is meaningful when its count denominator exists; undefined if all are true negatives.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return {'sessions':len(y),'positives':pos,'negatives':neg,'predicted_positives':tp+fp,
            'AP':float(average_precision_score(y,s)) if both else None,
            'AUROC':float(roc_auc_score(y,s)) if both else None,
            'ranking_undefined_reason':None if both else 'positive_or_negative_class_absent',
            'precision':precision,'recall':recall,'F1':f1,'FPR':fp/neg if neg else None,
            'FNR':fn/pos if pos else None,'false_alerts_per_1000_legitimate':1000*fp/neg if neg else None,
            'TP':tp,'TN':tn,'FP':fp,'FN':fn}


def join_scores_labels(scores:list[dict],labels:list[dict],partition:str,expected_ids:set[str]) -> list[dict]:
    """Exact one-to-one join; preserve prevalence; never deduplicate/inner-join away failures."""
    if partition not in ('test_held_out','test_known'):raise ValueError('Unknown test')
    def index(rows):
        if len(rows)!=len(expected_ids) or {r['session_id'] for r in rows}!=expected_ids:
            raise ValueError('Missing, duplicate, extra or cross-test session')
        return {r['session_id']:r for r in rows}
    ss,ll=index(scores),index(labels);out=[]
    for sid in sorted(expected_ids):
        a,b=ss[sid],ll[sid]
        if a['partition']!=partition or b['partition']!=partition or a['client']!=b['client'] or a['client'] not in CLIENTS:
            raise ValueError('Client/partition mismatch in score-label join')
        if type(b['label']) is not int or b['label'] not in (0,1):raise ValueError('Invalid threat label')
        loss=float(a['loss'])
        if not math.isfinite(loss) or loss<0:raise ValueError('Invalid anomaly score')
        out.append({**b,'loss':loss})
    return out


def score_endpoints(rows:list[dict],calibration:dict)->dict:
    if calibration.get('fit_partition')!='validation' or calibration.get('test_data_used') is not False:
        raise ValueError('Invalid calibration provenance')
    if {r['client'] for r in rows}!=set(CLIENTS):raise ValueError('Expected three test clients')
    raw=finite_scores([r['loss'] for r in rows]);y=np.array([r['label'] for r in rows])
    zs=np.array([(r['loss']-calibration['clients'][r['client']]['median'])/calibration['clients'][r['client']]['scale'] for r in rows])
    result={}
    for q in (.99,.98,.995):
        pred=np.array([r['loss']>calibration['clients'][r['client']]['thresholds'][str(q)] for r in rows])
        global_result=binary_metrics(y,zs,pred)
        global_result['score_representation']='validation_client_calibrated'
        global_result['raw_score_AP_secondary']=binary_metrics(y,raw,pred)['AP']
        global_result['raw_score_AUROC_secondary']=binary_metrics(y,raw,pred)['AUROC']
        clients={}
        for c in CLIENTS:
            ix=np.array([r['client']==c for r in rows]);clients[c]=binary_metrics(y[ix],raw[ix],pred[ix])
        aps={c:v['AP'] for c,v in clients.items()}
        valid=all(v is not None for v in aps.values())
        worst=min(aps.values()) if valid else None
        # Scenario summaries are detection counts only, no threat-only AUROC/AP.
        scenarios=[]
        for name in sorted({str(r.get('scenario','')) for r in rows}):
            ix=np.array([str(r.get('scenario',''))==name for r in rows]);n=int(ix.sum());yp=y[ix];pp=pred[ix]
            scenarios.append({'scenario':name,'sessions':n,'positives':int((yp==1).sum()),
                              'TP':int(((yp==1)&pp).sum()),'FN':int(((yp==1)&~pp).sum()),
                              'FP':int(((yp==0)&pp).sum()),'TN':int(((yp==0)&~pp).sum()),
                              'ranking_metrics_computed':False})
        result[str(q)]={'global':global_result,'per_client':clients,'worst_client_AP':worst,
                        'worst_client_ids':[c for c,v in aps.items() if valid and v==worst],
                        'worst_client_undefined_reason':None if valid else 'one_or_more_clients_lack_both_classes',
                        'scenario_counts':scenarios}
    return {'primary_quantile':.99,'quantiles':result,'primary_global_score':'validation_client_calibrated',
            'per_client_score':'raw','test_prevalence_unchanged':True,'threshold_search_performed':False}


def summaries(values)->dict:
    finite=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {'values':values,'n':len(finite),'missing':len(values)-len(finite),
            'mean':statistics.mean(finite) if finite else None,
            'sample_sd':statistics.stdev(finite) if len(finite)>1 else None,
            'median':statistics.median(finite) if finite else None,
            'minimum':min(finite) if finite else None,'maximum':max(finite) if finite else None}


def paired_summary(left:dict,right:dict,*,interval:bool)->dict:
    """Right-minus-left, on the adopted five seed identities, no session resampling."""
    if any(k not in SEEDS for k in set(left)|set(right)):raise ValueError('Unexpected/substitute seed')
    dif=[]
    for seed in SEEDS:
        a,b=left.get(seed),right.get(seed)
        ok=a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
        dif.append(float(b)-float(a) if ok else None)
    out=summaries(dif);out.update(seeds=list(SEEDS),interval_95=None,interval_reason='not_planned_for_endpoint')
    if interval:
        if out['n']!=5:out['interval_reason']='five_finite_matched_pairs_required'
        elif out['sample_sd']==0:out['interval_reason']='zero_observed_variability_not_certainty'
        else:
            width=float(t.ppf(.975,4))*out['sample_sd']/math.sqrt(5)
            out.update(interval_95=[out['mean']-width,out['mean']+width],interval_reason=None,
                       interval_label='Exploratory model-based conditional interval; nominal coverage not established')
    return out


def complete_analysis(cells:list[dict])->dict:
    key=lambda r:(r['condition'],r['test'],r['configuration'],int(r['seed']))
    by={key(r):r for r in cells}
    if len(by)!=len(cells):raise ValueError('Duplicate result cell')
    expected={(co,part,m,seed) for co in ('reference','noniid') for part in ('test_held_out','test_known') for m in 'ABC' for seed in SEEDS}
    if set(by)-expected:raise ValueError('Unexpected result identity or substitute seed')
    result={}
    endpoints=['global_AP','worst_AP']+[c+'_AP' for c in CLIENTS]+['global_'+m for m in ['AUROC','precision','recall','F1','FPR','FNR','false_alerts_per_1000_legitimate']]
    def value(cell,name):
        if cell is None:return None
        v=cell['metrics']['quantiles']['0.99']
        if name=='worst_AP':return v['worst_client_AP']
        if name[:2] in CLIENTS:return v['per_client'][name[:2]]['AP']
        return v['global'][name[len('global_'):]]
    for co in ('reference','noniid'):
        for test in ('test_held_out','test_known'):
            values={e:{m:{s:value(by.get((co,test,m,s)),e) for s in SEEDS} for m in 'ABC'} for e in endpoints}
            result[co+'/'+test]={e:{'by_method':{m:summaries([d[m][s] for s in SEEDS]) for m in 'ABC'},
                'B_minus_A':paired_summary(d['A'],d['B'],interval=e in ('global_AP','worst_AP')),
                'C_minus_B':paired_summary(d['B'],d['C'],interval=e in ('global_AP','worst_AP'))} for e,d in values.items()}
    return {'status':'COMPLETE' if set(by)==expected else 'INCOMPLETE','cells':len(cells),'analysis':result,
            'uncertainty_target':'algorithmic seed variability conditional on the fixed benchmark',
            'p_values_computed':False,'equivalence_claim':False,'practical_AP_reference_margin':.02}
