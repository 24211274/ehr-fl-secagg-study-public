#!/usr/bin/env python3
"""Read-only arithmetic audit of exported EHR scores; never loads or runs a model.

Inputs: score evidence ZIP, original final report ZIP, completion ZIP, frozen
source package ZIP, benchmark ZIP. Outputs go to a NEW audit directory.
AP is independently evaluated at grouped descending-score cut-offs. AUROC is
independently evaluated as P(score_positive > score_negative) + half ties.
scikit-learn is a second numerical implementation, not the sole calculation.
"""
from __future__ import annotations
import argparse, ast, collections, csv, hashlib, io, json, math, platform, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

CAMPAIGN='20260909T142009Z_2ad79bf6'
SEEDS=(11,29,47,71,97)
CONDS=('reference','noniid')
TESTS=('test_held_out','test_known')
CLIENTS=('C1','C2','C3')
QS=('0.99','0.98','0.995')
TOL=1e-12
PREFIX='EHR_FL_Main_Experiments_Package_v1_0_3/'

def sha(b): return hashlib.sha256(b).hexdigest()
def strict_json(b):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d: raise ValueError('Duplicate JSON key: '+k)
            d[k]=v
        return d
    return json.loads(b.decode('utf-8-sig'),object_pairs_hook=unique)

def zip_data(p):
    with zipfile.ZipFile(p) as z:
        ns=[i.filename for i in z.infolist() if not i.is_dir()]
        if len(ns)!=len(set(ns)):raise ValueError('Duplicate ZIP member')
        for n in ns:
            pp=PurePosixPath(n)
            if pp.is_absolute() or '..' in pp.parts or '\\' in n:raise ValueError('Unsafe ZIP member')
        return {n:z.read(n) for n in ns}

def inventory(data,name):
    m=strict_json(data[name]);rows=m['files']; ns=[r['path'] for r in rows]
    assert len(ns)==len(set(ns)) and set(data)==set(ns)|{name}
    for r in rows:
        assert len(data[r['path']])==r['bytes'],r['path']
        assert sha(data[r['path']])==r['sha256'],r['path']
    return {'verified_files':len(ns),'manifest_sha256':sha(data[name]),'missing':0,'extra':0,'mismatches':0}

def csv_rows(raw):
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
    assert len(reader.fieldnames)==len(set(reader.fieldnames))
    rows=list(reader)
    assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
    return rows

def independent_ranking(y,s):
    y=np.asarray(y,dtype=np.int64);s=np.asarray(s,dtype=np.float64)
    assert y.ndim==s.ndim==1 and len(y)==len(s) and np.isfinite(s).all()
    assert np.isin(y,[0,1]).all()
    P=int(y.sum());N=len(y)-P
    if not P or not N:return {'AP':None,'AUROC':None}
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(y)-1]
    cp=np.cumsum(yy)[ends]; group_pos=np.diff(np.r_[0,cp])
    ap=float(np.sum((group_pos/P)*(cp/(ends+1))))
    neg=np.sort(s[y==0]);pos=s[y==1]
    left=np.searchsorted(neg,pos,side='left');right=np.searchsorted(neg,pos,side='right')
    auc=float(np.sum(left+(right-left)*0.5)/(P*N))
    return {'AP':ap,'AUROC':auc}

def counts(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
    tp=int(np.sum(y&p));tn=int(np.sum(~y&~p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p))
    P=tp+fn;N=fp+tn
    return {'sessions':len(y),'positives':P,'negatives':N,'predicted_positives':tp+fp,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/P if P else None,
            'F1':2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None,
            'FPR':fp/N if N else None,'FNR':fn/P if P else None,
            'false_alerts_per_1000_legitimate':1000*fp/N if N else None,
            'TP':tp,'TN':tn,'FP':fp,'FN':fn}

def describe(vals):
    x=np.array(vals,dtype=float)
    return {'values':list(map(float,x)),'n':len(x),'missing':0,'mean':float(np.mean(x)),
            'sample_sd':float(np.std(x,ddof=1)) if len(x)>1 else None,
            'median':float(np.median(x)),'minimum':float(np.min(x)),'maximum':float(np.max(x))}

class Checks:
    def __init__(self):self.n=0;self.maxima=collections.defaultdict(float);self.failures=[]
    def eq(self,actual,expected,loc,family='structural'):
        self.n+=1
        if isinstance(actual,(float,np.floating)) and isinstance(expected,(float,int)) and not isinstance(expected,bool):
            delta=abs(float(actual)-float(expected));self.maxima[family]=max(self.maxima[family],delta)
            ok=math.isfinite(delta) and delta<=TOL
        else:ok=actual==expected
        if not ok:
            self.failures.append({'location':loc,'actual':str(actual)[:400],'expected':str(expected)[:400]})
    def fields(self,a,b,loc,family):
        for k,v in a.items(): self.eq(v,b[k],loc+'/'+k,family)

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    paths={k:getattr(args,k) for k in ('scores','report','completion','source','benchmark')}
    hashes_before={k:sha(p.read_bytes()) for k,p in paths.items()}
    S,R,C,P,D={k:zip_data(p) for k,p in paths.items()}.values()
    # Entire zip reads are for byte-integrity checks. Parse only labels, not events/features.
    checks=Checks();summ={
        'audit_id':'FINAL_STORED_SCORE_ARITHMETIC_AUDIT_'+CAMPAIGN,
        'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'mode':'READ_ONLY_RECOMPUTATION_FROM_EXPORTED_EXISTING_SCORES',
        'input_archives':{k:{'name':p.name,'sha256':hashes_before[k],'bytes':p.stat().st_size} for k,p in paths.items()},
        'scope':{'model_loaded':False,'model_inference_performed':False,'training_performed':False,
                 'thresholds_refitted':False,'scores_negated':False,'labels_reassigned':False,
                 'test_session_resampling':False,'campaign_accessed':False,'test_scores_read':True},
        'audit_environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                             'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
        'tolerance_absolute':TOL,
        'integrity':{'score_export':inventory(S,'audit_export_manifest.json'),
                     'final_report':inventory(R,'report_files.json'),
                     'completion':inventory(C,'report_files.json')}}
    expected_report='498869717d0415a4bdca5dfa5824805d390df8ddf984635df9a5d20e525f67fc'
    expected_selection='26519dd2e95f55667eac71ae5a5fc84529a25a14e39a5cd9a1b26648bd13510c'
    assert sha(R['report_files.json'])==expected_report
    assert sha(C['selected_models_manifest.json'])==expected_selection
    assert S['reviewed_final_report_manifest.json']==R['report_files.json']
    assert S['selected_models_manifest.json']==C['selected_models_manifest.json']
    assert S['stage_status.json']==R['stage_status.json']
    assert S['source_integrity.json']==R['source_integrity.json']
    manifest=strict_json(S['audit_export_manifest.json']);selection=strict_json(S['selected_models_manifest.json'])
    assert manifest['source_final_report_manifest_sha256']==expected_report
    assert manifest['source_selected_models_manifest_sha256']==expected_selection
    assert manifest['status']=='EXISTING_SCORE_EVIDENCE_EXPORT_NOT_RESCORE'
    assert manifest['model_loaded'] is manifest['model_inference_performed'] is manifest['training_performed'] is False
    # Frozen source membership and hash; source is not executed or imported.
    pmanifest=P[PREFIX+'PACKAGE_SHA256SUMS.txt'];p_entries=[]
    for line in pmanifest.decode().splitlines():
        h,n=line.split('  ',1);p_entries.append(n);assert sha(P[PREFIX+n])==h,n
    assert set(P)=={PREFIX+n for n in p_entries}|{PREFIX+'PACKAGE_SHA256SUMS.txt'}
    assert selection['package_manifest_sha256']==sha(pmanifest)
    summ['integrity']['frozen_source']={'verified_files':len(p_entries),'manifest_sha256':sha(pmanifest),'mismatches':0}
    # Bind benchmark to the original final evaluation source identity.
    dcheck=[]
    for rec in strict_json(S['source_integrity.json'])['files']:
        n=next(n for n in D if n.endswith('/data/'+rec['file']))
        assert len(D[n])==rec['bytes'] and sha(D[n])==rec['sha256']
        dcheck.append({'file':rec['file'],'sha256':sha(D[n])})
    summ['integrity']['benchmark_byte_checks']=dcheck
    labs=csv_rows(D[next(n for n in D if n.endswith('/data/labels.csv'))])
    assert len(labs)==len({r['session_id'] for r in labs})
    hmap=None
    for node in ast.parse(P[PREFIX+'app/ehr_preprocessing/pipeline.py'].decode()).body:
        if isinstance(node,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='CLIENTS' for a in node.targets):
            hmap=ast.literal_eval(node.value)
    assert hmap is not None
    revmap={h:c for c,h in hmap.items()}
    source_rows={r['session_id']:r for r in labs}
    testids={part:{r['session_id'] for r in labs if r['partition']==part} for part in TESTS}
    dev_ids={r['session_id'] for r in labs if r['partition'] in ('training','validation')}
    assert not testids[TESTS[0]]&testids[TESTS[1]]
    assert not (testids[TESTS[0]]|testids[TESTS[1]])&dev_ids
    composition=[]
    for part in TESTS:
        rows=[r for r in labs if r['partition']==part]
        for (cl,y,kind,sub,fam),n in sorted(collections.Counter((revmap[r['hospital_id']],int(r['binary_label']=='TRUE'),r['primary_class'],r['legitimate_subtype'],r['threat_family']) for r in rows).items()):
            composition.append({'test':part,'client':cl,'label':y,'primary_class':kind,'legitimate_subtype':sub,'threat_family':fam,'sessions':n})
    pd.DataFrame(composition).to_csv(out/'test_composition.csv',index=False)
    models={(m['configuration'],m['condition'],m['seed']):m for m in selection['models']}
    assert len(selection['models'])==len(models)==30
    assert set(models)=={(m,c,s) for m in 'ABC' for c in CONDS for s in SEEDS}
    assert set(n for n in S if n.startswith('scores/'))=={
        f'scores/{m}_{co}_{se}_{te}_scores.csv' for m,co,se in models for te in TESTS}
    assert set(n for n in S if n.startswith('calibrations/'))=={f'calibrations/{m}_{co}_{se}.json' for m,co,se in models}
    cells={};ranking_rows=[];decision_rows=[];audit_cells=[];group_rows=[];pair_diagnostics=[];within_client_checks=0
    scenario_groups_checked=0;calibration_checks=0;total_rows=0
    # Inspect (not infer) source score orientation: session_losses is nonnegative loss.
    summ['frozen_evaluation_source_sha256']=sha(P[PREFIX+'main_study/evaluation.py'])
    for (m,co,se),model in sorted(models.items()):
        stem=f'{m}_{co}_{se}';cal_raw=S[f'calibrations/{stem}.json'];cal=strict_json(cal_raw)
        assert sha(cal_raw)==model['calibration_sha256']
        assert cal['selected_checkpoint_sha256']==model['selected_checkpoint_sha256']
        assert cal['fit_partition']=='validation' and cal['test_data_used'] is False
        assert cal['training_seed']==se and cal['configuration']==m and cal['condition']==co
        assert cal['validation_scores_sha256']==model['validation_scores_sha256']
        assert set(cal['clients'])==set(CLIENTS) and cal['primary_quantile']==0.99
        for cl,cc in cal['clients'].items():
            assert cc['scale']>0 and math.isfinite(cc['scale']) and math.isfinite(cc['median'])
            assert cc['fit_partition']=='validation' and cc['alert_operator']=='>' and cc['quantile_method']=='linear'
            assert cc['sessions']==sum(r['partition']=='validation' and revmap[r['hospital_id']]==cl for r in labs)
            assert set(cc['thresholds'])==set(QS)
            assert cc['thresholds']['0.98']<=cc['thresholds']['0.99']<=cc['thresholds']['0.995']
            for q in QS:checks.eq((cc['thresholds'][q]-cc['median'])/cc['scale'],cc['calibrated_thresholds'][q],stem+'/'+cl+'/'+q,'calibration_transform')
            calibration_checks+=1
        for te in TESTS:
            name=f'{stem}_{te}';original=strict_json(R[f'cells/{name}.json'])
            assert original['checkpoint_sha256']==model['selected_checkpoint_sha256']
            assert original['calibration_sha256']==model['calibration_sha256']
            assert (original['configuration'],original['condition'],original['seed'],original['test'])==(m,co,se,te)
            rows=csv_rows(S[f'scores/{name}_scores.csv']);ids=[r['session_id'] for r in rows]
            assert len(ids)==len(set(ids)) and set(ids)==testids[te]
            for r in rows:
                sr=source_rows[r['session_id']]
                expected=(revmap[sr['hospital_id']],te,str({'TRUE':1,'FALSE':0}[sr['binary_label']]),sr['scenario_id'],sr['primary_class'])
                actual=tuple(r[k] for k in ('client','partition','label','scenario','primary_class'))
                assert actual==expected,(name,r['session_id'])
                assert (sr['primary_class']=='threat')==(r['label']=='1')
            total_rows+=len(rows)
            raw=np.array([float(r['loss']) for r in rows]);y=np.array([int(r['label']) for r in rows]);clients=np.array([r['client'] for r in rows])
            assert np.isfinite(raw).all() and (raw>=0).all()
            z=np.array([(x-cal['clients'][cl]['median'])/cal['clients'][cl]['scale'] for x,cl in zip(raw,clients)])
            assert np.isfinite(z).all()
            masks={cl:clients==cl for cl in CLIENTS}
            ranks={}
            for sc,scs,ix in [('global_calibrated',z,np.ones(len(y),bool)),('global_raw_secondary',raw,np.ones(len(y),bool))]+[(cl,raw,masks[cl]) for cl in CLIENTS]:
                yy,xx=y[ix],scs[ix];rnk=independent_ranking(yy,xx);ranks[sc]=rnk
                checks.eq(rnk['AP'],float(average_precision_score(yy,xx)),name+'/'+sc+'/sklearn_AP','independent_vs_sklearn_AP')
                checks.eq(rnk['AUROC'],float(roc_auc_score(yy,xx)),name+'/'+sc+'/sklearn_AUROC','independent_vs_sklearn_AUROC')
                ranking_rows.append({'configuration':m,'condition':co,'seed':se,'test':te,'scope':sc,'sessions':len(yy),'positives':int(yy.sum()),'negatives':int((yy==0).sum()),**rnk})
            for cl,ix in masks.items():
                rc=independent_ranking(y[ix],z[ix]);within_client_checks+=1
                checks.fields(rc,ranks[cl],name+'/'+cl+'/affine_invariance','affine_invariance')
            apmin=min(ranks[cl]['AP'] for cl in CLIENTS)
            worst=[cl for cl in CLIENTS if ranks[cl]['AP']==apmin]
            metricq={};pred_cache={}
            for q in QS:
                orig=original['metrics']['quantiles'][q]
                thresholds=np.array([cal['clients'][cl]['thresholds'][q] for cl in clients])
                pred=raw>thresholds;pred_cache[q]=pred
                cpred=z>np.array([cal['clients'][cl]['calibrated_thresholds'][q] for cl in clients])
                checks.eq(bool(np.array_equal(pred,cpred)),True,name+'/'+q+'/raw_z_decision_equivalence')
                glob={**counts(y,pred),**ranks['global_calibrated'],
                      'raw_score_AP_secondary':ranks['global_raw_secondary']['AP'],
                      'raw_score_AUROC_secondary':ranks['global_raw_secondary']['AUROC']}
                pc={cl:{**counts(y[ix],pred[ix]),**ranks[cl]} for cl,ix in masks.items()}
                checks.fields(glob,orig['global'],name+'/'+q+'/global','metric_report')
                for cl in CLIENTS:checks.fields(pc[cl],orig['per_client'][cl],name+'/'+q+'/'+cl,'metric_report')
                checks.eq(apmin,orig['worst_client_AP'],name+'/'+q+'/worst','metric_report')
                checks.eq(worst,orig['worst_client_ids'],name+'/'+q+'/worst_ids')
                grouped={}
                for i,r in enumerate(rows):grouped.setdefault(r['scenario'],[]).append(i)
                original_sc={r['scenario']:r for r in orig['scenario_counts']}
                assert set(grouped)==set(original_sc)
                for sid,ix in grouped.items():
                    cs=counts(y[ix],pred[ix]);scfields={k:cs[k] for k in ('sessions','positives','TP','FN','FP','TN')}
                    checks.fields(scfields,original_sc[sid],name+'/'+q+'/scenario/'+sid,'scenario_counts')
                    scenario_groups_checked+=1
                for sc,val in [('global',glob),*list(pc.items())]:
                    decision_rows.append({'configuration':m,'condition':co,'seed':se,'test':te,'quantile':q,'scope':sc,**val})
                metricq[q]={'global':glob,'per_client':pc,'worst_client_AP':apmin,'worst_client_ids':worst}
            assert not np.any(pred_cache['0.99'] & ~pred_cache['0.98'])
            assert not np.any(pred_cache['0.995'] & ~pred_cache['0.99'])
            cellrecord={'configuration':m,'condition':co,'seed':se,'test':te,'metrics':{'quantiles':metricq}}
            cells[(co,te,m,se)]=cellrecord
            audit_cells.append({'name':name,'score_rows':len(rows),'loss_min':float(raw.min()),'loss_max':float(raw.max()),
                                'score_sha256':sha(S[f'scores/{name}_scores.csv']),
                                'calibration_sha256':sha(cal_raw),'checkpoint_link_sha256':model['selected_checkpoint_sha256'],
                                'exact_membership_and_source_labels_match':True,
                                'global_primary_AP':ranks['global_calibrated']['AP'],'global_primary_AUROC':ranks['global_calibrated']['AUROC']})
            # Post-test descriptive diagnostics. Do not fit or choose thresholds or score direction.
            for field in ('primary_class','legitimate_subtype','threat_family'):
                groups={}
                for i,r in enumerate(rows):
                    val=source_rows[r['session_id']][field]
                    if not val:continue
                    groups.setdefault(val,[]).append(i)
                for val,ix in groups.items():
                    xx=raw[ix];zz=z[ix];cm=counts(y[ix],pred_cache['0.99'][ix])
                    group_rows.append({'configuration':m,'condition':co,'seed':se,'test':te,'group_field':field,'group':val,
                                       **cm,'raw_q25':float(np.quantile(xx,.25)),'raw_median':float(np.median(xx)),
                                       'raw_q75':float(np.quantile(xx,.75)),'calibrated_median':float(np.median(zz)),
                                       'mean_alert_rate':float(np.mean(pred_cache['0.99'][ix]))})
            for negclass in ('routine_legitimate','unusual_legitimate'):
                negix=np.array([r['primary_class']==negclass for r in rows]);both=(y==1)|negix
                if not negix.any():continue
                # AUROC numerator components expose ordering without negating scores.
                for sc,xx in [('global_calibrated',z),('global_raw_secondary',raw)]:
                    pos=xx[y==1];neg=np.sort(xx[negix]);l=np.searchsorted(neg,pos,'left');r=np.searchsorted(neg,pos,'right')
                    wins=int(l.sum());ties=int((r-l).sum());pairs=len(pos)*len(neg)
                    pair_diagnostics.append({'configuration':m,'condition':co,'seed':se,'test':te,'scope':sc,
                      'negative_class':negclass,'positives':len(pos),'negatives':len(neg),'positive_negative_pairs':pairs,
                      'positive_score_greater_pairs':wins,'equal_score_pairs':ties,
                      'positive_score_lower_pairs':pairs-wins-ties,'AUROC':(wins+.5*ties)/pairs,
                      'post_test_descriptive_not_new_primary_endpoint':True})
    checks.eq(total_rows,manifest['score_rows'],'score_rows')
    # Recompute predeclared seed summaries and conditional paired Student-t intervals.
    endpoints=['global_AP','worst_AP']+[cl+'_AP' for cl in CLIENTS]+['global_'+a for a in ('AUROC','precision','recall','F1','FPR','FNR','false_alerts_per_1000_legitimate')]
    report_summary=strict_json(R['paired_seed_analysis.json']);seed_summary=[];contrasts=[]
    for co in CONDS:
        for te in TESTS:
            orig=report_summary['analysis'][co+'/'+te]
            for ep in endpoints:
                method_values={}
                for m in 'ABC':
                    vals=[]
                    for se in SEEDS:
                        met=cells[(co,te,m,se)]['metrics']['quantiles']['0.99']
                        if ep=='worst_AP':val=met['worst_client_AP']
                        elif ep[:2] in CLIENTS:val=met['per_client'][ep[:2]]['AP']
                        else:val=met['global'][ep[7:]]
                        vals.append(val)
                    method_values[m]=np.array(vals);s=describe(vals)
                    for k,v in s.items():
                        if k=='values':
                            for i,x in enumerate(v):checks.eq(x,orig[ep]['by_method'][m][k][i],f'{co}/{te}/{ep}/{m}/{k}/{i}','summary')
                        else:checks.eq(v,orig[ep]['by_method'][m][k],f'{co}/{te}/{ep}/{m}/{k}','summary')
                    seed_summary.append({'condition':co,'test':te,'endpoint':ep,'configuration':m,**{k:v for k,v in s.items() if k!='values'}})
                for l,r in [('A','B'),('B','C')]:
                    diff=method_values[r]-method_values[l];s=describe(diff);name=r+'_minus_'+l
                    for k,v in s.items():
                        if k=='values':
                            for i,x in enumerate(v):checks.eq(x,orig[ep][name][k][i],f'{co}/{te}/{ep}/{name}/{k}/{i}','paired_summary')
                        else:checks.eq(v,orig[ep][name][k],f'{co}/{te}/{ep}/{name}/{k}','paired_summary')
                    interval=None
                    if ep in ('global_AP','worst_AP'):
                        width=float(t.ppf(.975,4))*s['sample_sd']/math.sqrt(5)
                        interval=[s['mean']-width,s['mean']+width]
                        for k in (0,1):checks.eq(interval[k],orig[ep][name]['interval_95'][k],f'{co}/{te}/{ep}/{name}/CI/{k}','paired_CI')
                    contrasts.append({'condition':co,'test':te,'endpoint':ep,'contrast':name,**{k:v for k,v in s.items() if k!='values'},
                                      'ci95_low':interval[0] if interval else None,'ci95_high':interval[1] if interval else None,
                                      **{f'difference_seed_{se}':float(v) for se,v in zip(SEEDS,diff)}})
    # Tables are derived audit outputs, not a replacement campaign or a rescore.
    for fname,rows in [('recomputed_rank_metrics.csv',ranking_rows),('recomputed_decision_metrics.csv',decision_rows),
                       ('seed_summary.csv',seed_summary),('paired_contrasts.csv',contrasts),
                       ('descriptive_class_score_summary.csv',group_rows),('descriptive_pairwise_ordering.csv',pair_diagnostics)]:
        pd.DataFrame(rows).to_csv(out/fname,index=False)
    summaryframe=pd.DataFrame(seed_summary)
    pframe=pd.DataFrame(pair_diagnostics)
    gframe=pd.DataFrame(group_rows)
    # Presentation tables, retaining all methods/conditions/tests (no favourable selection).
    globaltable=[]
    for te in TESTS:
        for co in CONDS:
            for m in 'ABC':
                values=summaryframe[(summaryframe.test==te)&(summaryframe.condition==co)&(summaryframe.configuration==m)].set_index('endpoint')
                record={'test':te,'condition':co,'configuration':m}
                for ep in ['global_AP','global_AUROC','global_precision','global_recall','global_F1','global_FPR','global_FNR','global_false_alerts_per_1000_legitimate','worst_AP']:
                    record[ep+'_mean']=float(values.loc[ep,'mean']);record[ep+'_sd']=float(values.loc[ep,'sample_sd'])
                globaltable.append(record)
    pd.DataFrame(globaltable).to_csv(out/'global_results_table.csv',index=False)
    sums=[]
    for key,g in gframe.groupby(['test','condition','configuration','group_field','group'],sort=True):
        rec=dict(zip(['test','condition','configuration','group_field','group'],key));rec['n_seeds']=len(g)
        assert len(g)==5 and g.sessions.nunique()==1
        rec['sessions_per_model']=int(g.sessions.iloc[0])
        for col in ['TP','TN','FP','FN','mean_alert_rate','raw_median','calibrated_median']:
            rec[col+'_mean_across_seeds']=float(g[col].mean());rec[col+'_sd_across_seeds']=float(g[col].std(ddof=1))
        sums.append(rec)
    pd.DataFrame(sums).to_csv(out/'descriptive_class_error_summary.csv',index=False)
    pairsummary=pframe.groupby(['test','condition','configuration','scope','negative_class'])['AUROC'].agg(['mean','std','min','max']).reset_index()
    pairsummary.to_csv(out/'descriptive_pairwise_ordering_summary.csv',index=False)
    mixture_rows=[]
    for (co,m,se,sc),g in pframe[pframe.test=='test_held_out'].groupby(['condition','configuration','seed','scope']):
        assert len(g)==2
        weighted=float(np.dot(g.AUROC.to_numpy(),g.negatives.to_numpy())/g.negatives.sum())
        target_rank=next(r for r in ranking_rows if (r['condition'],r['configuration'],r['seed'],r['test'],r['scope'])==(co,m,se,'test_held_out',sc))
        checks.eq(weighted,target_rank['AUROC'],f'mixture/{co}/{m}/{se}/{sc}','AUROC_mixture_identity')
        mixture_rows.append({'condition':co,'configuration':m,'seed':se,'scope':sc,'weighted_component_AUROC':weighted,'original_global_AUROC':target_rank['AUROC']})
    pd.DataFrame(mixture_rows).to_csv(out/'descriptive_AUROC_mixture_checks.csv',index=False)
    summ['AUROC_negative_class_mixture_checks']=len(mixture_rows)
    summ.update({
      'counts':{'score_files':60,'calibration_files':30,'model_client_calibrations':calibration_checks,
                'score_rows':total_rows,'distinct_sessions':len(testids[TESTS[0]]|testids[TESTS[1]]),
                'distinct_sessions_per_test':{k:len(v) for k,v in testids.items()},
                'source_label_rows':len(labs),'source_matched_score_rows':total_rows,
                'unique_ranking_groups_compared_to_report':len(ranking_rows),
                'unique_AP_and_AUROC_values_compared_to_report':2*len(ranking_rows),
                'per_client_affine_ranking_invariance_checks':within_client_checks,
                'decision_metric_groups_recomputed':len(decision_rows),'scenario_count_groups_recomputed':scenario_groups_checked,
                'method_endpoint_seed_summaries':len(seed_summary),'paired_endpoint_summaries':len(contrasts),
                'exploratory_paired_intervals':sum(x['ci95_low'] is not None for x in contrasts)},
      'comparison':{'scalar_checks':checks.n,'absolute_max_differences':dict(checks.maxima),
                    'failures':checks.failures,'failures_count':len(checks.failures)},
      'cells':audit_cells,'summary_global_results':globaltable,
      'source_labels_audit':{'session_membership_exact_all_models':True,'duplicates_within_score_files':0,
         'test_partitions_disjoint':True,'test_and_development_session_ids_disjoint':True,
         'TRUE_mapped_to_1_threat':True,'FALSE_mapped_to_0_legitimate':True,
         'client_partition_scenario_and_primary_class_exact_match':True,
         'source_is_synthetic_not_external_clinical_validation':True},
      'limits':[
        'Verifies arithmetic from supplied stored scores, not a new inference from the selected weight files.',
        'No checkpoint, event tensors or validation-session score arrays were loaded or recomputed.',
        'The raw score files were not themselves included in the original final-report hash manifest. Their hashes are fixed by this export; agreement with originally recorded metrics is checked here.',
        'Calibrations match the pre-evaluation completion manifest; quantiles, medians and scale estimates were not refitted because original validation scores are not in this export.',
        'Source label and membership consistency is with the supplied synthetic snapshot, not external verification of label realism.',
        'No general claim of absence of leakage, privacy certification or real-hospital deployment readiness follows from this audit.',
        'Ninety thousand score rows are repeated scoring of 3009 distinct sessions across 30 models, not 90270 independent test observations.',
        'Group score distributions and within-negative-class pairwise comparisons are post-test descriptive diagnostics, not newly selected primary endpoints or retuning.',
        'Five-seed intervals are exploratory and conditional on a fixed benchmark. No equivalence test or p-values were added.'
      ]})
    assert {k:sha(p.read_bytes()) for k,p in paths.items()}==hashes_before
    summ['all_input_archive_bytes_unchanged_after_audit']=True
    summ['status']='ARITHMETIC_AUDIT_PASS_NO_MODEL_RESCORE_NO_RETUNING' if not checks.failures else 'ARITHMETIC_AUDIT_MISMATCH_REVIEW_REQUIRED'
    (out/'Final_Score_Arithmetic_Audit.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False)+'\n')
    print(summ['status']);print(json.dumps(summ['counts'],indent=2));print(json.dumps(summ['comparison'],indent=2)[:5000])
    return 0 if not checks.failures else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('scores','report','completion','source','benchmark','output'):p.add_argument('--'+a,type=Path,required=True)
    args=p.parse_args()
    raise SystemExit(main(args))
