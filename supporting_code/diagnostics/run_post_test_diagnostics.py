"""Post-test diagnostics for revision v1.18; fixed snapshot/checkpoints, no fitting.
All outputs are aggregate or anonymous row ordinals. No raw entity/session IDs,
raw benchmark rows, trained weights, or new deployment-ready detector are exported.
"""
from pathlib import Path
import sys, json, csv, io, zipfile, hashlib, time, math, itertools, collections, platform
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path("/mnt/c/Users/RNO/Desktop/Coureses/P1/report_revision_v1_18_20261018")
OUT=ROOT/"evidence/diagnostics"
OUT.mkdir(parents=True,exist_ok=True)
RUN=Path("/home/mai/ehr_me06_reproduction_20261011_v1")
SOURCE=RUN/"source"
sys.path[:0]=[str(SOURCE),str(SOURCE/"app")]
from main_study.test_inputs import load_final_inputs
from ehr_preprocessing.dataset import collate_sessions
from ehr_fedavg.task import setup_torch, model_factory
from pilot_core.model import session_losses, state_digest
setup_torch()
torch.set_num_threads(1)
assert torch.get_num_interop_threads()==1
torch.use_deterministic_algorithms(True)
BENCH=Path("/mnt/c/Users/RNO/Downloads/review_FL/04_EHR_FL_Benchmark_Release_Candidate_v1_0.zip")
SCORES=Path("/mnt/c/Users/RNO/Downloads/review_FL/01_Saved_Score_Audit_Inputs_Original.zip")
HIST=Path("/mnt/c/Users/RNO/Downloads/review_FL/03_Training_Completion_Original.zip")
manifest=json.loads((RUN/"campaign/selected_models_manifest.json").read_text())
inventory=[]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def identity(path,role):
 p=Path(path);d={"path":str(p),"role":role,"bytes":p.stat().st_size,"sha256":sha(p)};inventory.append(d);return d["sha256"]
def write(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
def rho(a,b):
 if len(a)<3 or len(set(a))<2 or len(set(b))<2:return None
 return float(spearmanr(a,b).statistic)
def desc(v):
 a=np.asarray(v,dtype=float)
 return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"min":float(a.min()),"max":float(a.max())}
def metrics(y,s):
 return {"n":len(y),"threats":int(sum(y)),"AP":float(average_precision_score(y,s)),"AUROC":float(roc_auc_score(y,s))} if len(set(y))==2 else {"n":len(y),"status":"one_class"}
for p,role in [(BENCH,"raw_archive"),(SCORES,"original_saved_score_archive"),(HIST,"original_selected_manifest_archive"),(RUN/"campaign/selected_models_manifest.json","replay_selected_manifest"),(ROOT/"evidence/Report_Model.json","current_report_model"),(ROOT/"evidence/Reference_Model.json","reference_report_model"),(Path(__file__),"new_post_test_code")]:identity(p,role)
for rel in ["app/pilot_core/model.py","app/ehr_fedavg/task.py","app/ehr_preprocessing/dataset.py","app/ehr_preprocessing/pipeline.py","main_study/test_inputs.py","main_study/final_evaluator.py"]:
 identity(SOURCE/rel,"inspected_executed_kernel")
schema=json.loads((RUN/"campaign/frozen/development/encoding_schema.json").read_text())
identity(RUN/"campaign/frozen/development/encoding_schema.json","fixed_schema")
identity(RUN/"campaign/frozen/development/relative_time_scaler.json","fixed_scaler")
fields=[f["name"] for f in schema["categorical_fields"]]
known={f["name"]:set(f["categories"])-{"__UNK__","__MISSING__"} for f in schema["categorical_fields"]}
CLIENT={"H1":"C1","H2":"C2","H3":"C3"}
# Actual hospital mapping is read from the retained preprocessing implementation.
from ehr_preprocessing.pipeline import CLIENTS as HOSPITALS
CLIENT={v:k for k,v in HOSPITALS.items()}
class RawZip:
 def __init__(self,path):
  self.z=zipfile.ZipFile(path)
  self.names={Path(n).name:n for n in self.z.namelist() if "/data/" in n and n.endswith(".csv")}
 def text(self,name):return io.TextIOWrapper(self.z.open(self.names[name]),encoding="utf-8-sig",newline="")
source=RawZip(BENCH)
labels={r["session_id"]:r for r in csv.DictReader(source.text("labels.csv"))}
if len(labels)!=17464:raise AssertionError("Unexpected label count")
sets=collections.defaultdict(lambda:{"sessions":set(),"actors":set(),"patients":set()})
session={}
for sid,l in labels.items():
 c=CLIENT[l["hospital_id"]];session[sid]={"partition":l["partition"],"class":l["primary_class"],"client":c,"actors":set(),"patients":set(),"length":0,"excluded":0,"excluded_fields":collections.Counter(),"patient_event_ids":[]}
 for ckey in [c,"pooled"]:sets[(l["partition"],ckey)]["sessions"].add(sid)
duplicates=0
for row in csv.DictReader(source.text("events.csv")):
 s=session[row["session_id"]];s["length"]+=1
 s["actors"].add(row["actor_id"])
 if row["patient_id"]:s["patients"].add(row["patient_id"]);s["patient_event_ids"].append(row["patient_id"])
 for ckey in [s["client"],"pooled"]:
  sets[(s["partition"],ckey)]["actors"].add(row["actor_id"])
  if row["patient_id"]:sets[(s["partition"],ckey)]["patients"].add(row["patient_id"])
 for f in fields:
  if row[f] not in known[f]:s["excluded"]+=1;s["excluded_fields"][f]+=1
 duplicates+=row["department"]==row["ward_or_location"]
for name in ["events.csv","labels.csv"]:
 n=source.names[name];b=source.z.read(n);inventory.append({"path":str(BENCH)+"!/"+n,"role":"raw_member","bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()})
set_sizes=[{"partition":p,"client":c,**{k:len(v) for k,v in st.items()}} for (p,c),st in sorted(sets.items())]
intersections=[]
for (p,c),(q,d) in itertools.combinations(sorted(sets),2):
 if p==q:continue
 a,b=sets[(p,c)],sets[(q,d)]
 intersections.append({"partition_a":p,"client_a":c,"partition_b":q,"client_b":d,**{k:len(a[k]&b[k]) for k in a}})
class_descriptive=[]
for (part,cl,c),ss in itertools.groupby(sorted(session.values(),key=lambda s:(s["partition"],s["class"],s["client"])),key=lambda s:(s["partition"],s["class"],s["client"])):
 ss=list(ss);train=sets[("training",c)]
 for s in ss:
  s["actor_all_seen_training"]=s["actors"].issubset(train["actors"])
  s["patient_any_unseen_training"]=bool(s["patients"]-train["patients"])
  s["patient_seen_event_fraction"]=sum(x in train["patients"] for x in s["patient_event_ids"])/len(s["patient_event_ids"]) if s["patient_event_ids"] else 1.
  s["excluded_target_fraction"]=s["excluded"]/(13*s["length"])
 class_descriptive.append({"partition":part,"class":cl,"client":c,"sessions":len(ss),"events":sum(s["length"] for s in ss),"length":desc([s["length"] for s in ss]),"all_actor_ids_seen_in_same_client_training_sessions":sum(s["actor_all_seen_training"] for s in ss),"sessions_with_any_patient_not_seen_in_same_client_training":sum(s["patient_any_unseen_training"] for s in ss),"patient_seen_event_fraction":desc([s["patient_seen_event_fraction"] for s in ss]),"excluded_targets":sum(s["excluded"] for s in ss),"possible_targets":13*sum(s["length"] for s in ss),"excluded_by_field":dict(sum((s["excluded_fields"] for s in ss),collections.Counter()))})
length_only=[]
for part in ["test_held_out","test_known"]:
 for subset in ["all","threat_vs_unusual"]:
  ss=[s for s in session.values() if s["partition"]==part and (subset=="all" or s["class"]!="routine_legitimate")]
  y=[int(s["class"]=="threat") for s in ss];lens=[s["length"] for s in ss]
  length_only.append({"test":part,"subset":subset,"negative_event_count":metrics(y,[-x for x in lens]),"positive_event_count":metrics(y,lens),"selection":"Both signed descriptors reported; neither selected or fitted as a replacement detector."})
write("Raw_Diagnostics.json",{"post_test":True,"events":sum(s["length"] for s in session.values()),"sessions":len(session),"duplicate_department_ward_events":duplicates,"entity_set_sizes":set_sizes,"entity_intersections":intersections,"class_client_descriptive":class_descriptive,"length_only":length_only,"limits":["Counts describe the fixed supplied snapshot, not the original construction algorithm.","Identifiers excluded from model inputs; recurrence does not demonstrate memorisation.","Length-only ranking is post-test descriptive; no model/threshold selection or length-matched retraining."]})
print("RAW_DIAGNOSTICS_COMPLETE",flush=True)
samples,_,coverage=load_final_inputs(source,RUN/"campaign")
original_zip=zipfile.ZipFile(SCORES)
with zipfile.ZipFile(HIST) as z:hist=json.loads(z.read("selected_models_manifest.json"))
histmap={(m["configuration"],m["condition"],m["seed"]):m for m in hist["models"]}
cells=[];groups=[];associations=[];alternative_scores=[];cases=[]
max_recomposition=0.;start=time.monotonic()
for row in manifest["models"]:
 key=(row["configuration"],row["condition"],row["seed"]);h=histmap[key]
 cp=RUN/"campaign"/row["checkpoint"];calpath=RUN/"campaign"/row["calibration"]
 cp_hash=identity(cp,"fixed_selected_checkpoint");calhash=identity(calpath,"fixed_validation_calibration")
 if cp_hash!=row["selected_checkpoint_sha256"] or cp_hash!=h["selected_checkpoint_sha256"]:raise AssertionError("Checkpoint identity mismatch")
 if calhash!=row["calibration_sha256"] or calhash!=h["calibration_sha256"]:raise AssertionError("Calibration identity mismatch")
 model=model_factory();model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True),strict=True);model.eval()
 before=state_digest(model);cal=json.loads(calpath.read_text())
 for part in ["test_held_out","test_known"]:
  ss=samples[part];values=[];comps=[];fixed13=[];omitward=[]
  for lo in range(0,len(ss),64):
   batch=collate_sessions(ss[lo:lo+64])
   with torch.no_grad():
    outputs=model(batch["x"],batch["lengths"])
    loss,ev,mask=session_losses(outputs,batch["categorical"],batch["binary"],batch["temporal"],batch["lengths"],batch["categorical_valid"])
    cats,bb,tt=outputs;valid=batch["categorical_valid"];ncat=valid.sum(-1);ngroup=2+(ncat>0).to(bb.dtype)
    terms=[]
    for j,logits in enumerate(cats):
     target=batch["categorical"][:,:,j].clone().masked_fill(~valid[:,:,j],-100)
     terms.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction="none",ignore_index=-100,label_smoothing=0).reshape_as(ncat))
    terms=torch.stack(terms,-1)
    bl=F.binary_cross_entropy_with_logits(bb,batch["binary"],reduction="none")
    tl=F.smooth_l1_loss(tt,batch["temporal"],reduction="none",beta=1.)
    event_terms=torch.cat([terms/ncat.clamp_min(1)[:,:,None],bl/3.,tl/6.],-1)/ngroup[:,:,None]
    session_terms=(event_terms*mask[:,:,None]).sum(1)/batch["lengths"][:,None]
    rec=session_terms.sum(-1);max_recomposition=max(max_recomposition,float((rec-loss).abs().max()))
    cat_sum=terms.sum(-1)
    f13=(cat_sum/13+bl.mean(-1)+tl.mean(-1))/3
    nw=ncat-valid[:,:,12].to(ncat.dtype)
    ow=((cat_sum-terms[:,:,12])/nw.clamp_min(1)+bl.mean(-1)+tl.mean(-1))/(2+(nw>0).to(bb.dtype))
    values.extend(loss.numpy().tolist());comps.extend(session_terms.numpy().astype(float).tolist())
    fixed13.extend(((f13*mask).sum(-1)/batch["lengths"]).numpy().tolist())
    omitward.extend(((ow*mask).sum(-1)/batch["lengths"]).numpy().tolist())
  values=np.asarray(values);comps=np.asarray(comps);fixed13=np.asarray(fixed13);omitward=np.asarray(omitward)
  basename=f"{key[0]}_{key[1]}_{key[2]}_{part}_scores.csv";member=next(n for n in original_zip.namelist() if n.endswith(basename))
  b=original_zip.read(member);inventory.append({"path":str(SCORES)+"!/"+member,"role":"original_saved_scores","bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()})
  original={x["session_id"]:float(x["loss"]) for x in csv.DictReader(io.StringIO(b.decode("utf-8-sig")))}
  expected=np.asarray([original[s["session_id"]] for s in ss])
  if not np.array_equal(values,expected):raise AssertionError(f"Original score mismatch {basename}: {np.max(np.abs(values-expected))}")
  ann=[session[s["session_id"]] for s in ss]
  y=np.asarray([int(a["class"]=="threat") for a in ann])
  pred=np.asarray([v>cal["clients"][a["client"]]["thresholds"]["0.99"] for v,a in zip(values,ann)])
  cellbase={"configuration":key[0],"condition":key[1],"seed":key[2],"test":part}
  cells.append({**cellbase,"sessions":len(ss),"original_scores_exact":True,"max_abs_score_delta":float(np.abs(values-expected).max()),"checkpoint_unchanged":True,"weights_not_fitted":True})
  for cl in sorted(set(a["class"] for a in ann)):
   idx=np.asarray([i for i,a in enumerate(ann) if a["class"]==cl]);cc=comps[idx];mean_score=float(values[idx].mean())
   sums=[float(cc[:,:13].sum(-1).mean()),float(cc[:,13:16].sum(-1).mean()),float(cc[:,16:].sum(-1).mean())]
   groups.append({**cellbase,"class":cl,"sessions":len(idx),"mean_original_raw_loss":mean_score,"categorical_contribution":sums[0],"binary_contribution":sums[1],"temporal_contribution":sums[2],"categorical_share":sums[0]/mean_score,"binary_share":sums[1]/mean_score,"temporal_share":sums[2]/mean_score,"department_and_ward_share":float(cc[:,[2,12]].sum(-1).mean())/mean_score,"mean_field_contributions":{f:float(cc[:,j].mean()) for j,f in enumerate(fields+schema["binary_fields"]+schema["temporal_fields"])}})
   alternative_scores.append({**cellbase,"class":cl,"sessions":len(idx),"fixed13_mean_raw_loss":float(fixed13[idx].mean()),"fixed13_mean_loss_delta":float((fixed13-values)[idx].mean()),"fixed13_percent_mean_loss_change":100*float((fixed13-values)[idx].mean())/mean_score,"omit_ward_head_mean_raw_loss":float(omitward[idx].mean()),"omit_ward_head_mean_loss_delta":float((omitward-values)[idx].mean()),"omit_ward_head_percent_mean_loss_change":100*float((omitward-values)[idx].mean())/mean_score})
   for c in ["C1","C2","C3"]:
    jj=np.asarray([i for i in idx if ann[i]["client"]==c])
    for predictor in ["length","excluded_target_fraction","patient_seen_event_fraction"]:
     x=[ann[i][predictor] for i in jj]
     associations.append({**cellbase,"class":cl,"client":c,"predictor":predictor,"sessions":len(jj),"rho_raw_loss":rho(x,values[jj].tolist()),"predictor_min":min(x),"predictor_max":max(x)})
  if key[0]=="C" and key[2]==11:
   # Deterministic middle-scoring example within each label/decision cell. No IDs exported.
   for label in [0,1]:
    for decision in [0,1]:
     jj=[i for i in range(len(ss)) if y[i]==label and pred[i]==decision]
     if not jj:continue
     ix=sorted(jj,key=lambda i:(values[i],i))[len(jj)//2];a=ann[ix]
     cases.append({**cellbase,"anonymous_sorted_test_row_ordinal":ix+1,"classification":("TP" if label else "FP") if decision else ("FN" if label else "TN"),"source_class":a["class"],"client":a["client"],"length":a["length"],"excluded_targets":a["excluded"],"possible_targets":13*a["length"],"patient_seen_event_fraction":a["patient_seen_event_fraction"],"raw_loss":float(values[ix]),"fixed_validation_threshold":cal["clients"][a["client"]]["thresholds"]["0.99"],"categorical_contribution":float(comps[ix,:13].sum()),"binary_contribution":float(comps[ix,13:16].sum()),"temporal_contribution":float(comps[ix,16:].sum()),"selection":"Middle original raw score in this source-label/unchanged-decision cell; illustrative, not representative sampling."})
  print("DIAGNOSTIC_SCORED",basename,"exact",len(ss),flush=True)
 if before!=state_digest(model) or any(p.grad is not None for p in model.parameters()):raise AssertionError("Model changed")
 if sha(cp)!=cp_hash or sha(calpath)!=calhash:raise AssertionError("Input file altered")
 if time.monotonic()-start>900:raise RuntimeError("Bounded 900-second diagnostic budget exceeded")
def across_seeds(rows,keys,metrics):
 out=[]
 for k,rr in itertools.groupby(sorted(rows,key=lambda r:tuple(r[x] for x in keys)),key=lambda r:tuple(r[x] for x in keys)):
  rr=list(rr);v=dict(zip(keys,k));v["training_seeds"]=len(rr)
  for name in metrics:
   a=[r[name] for r in rr if r[name] is not None]
   v[name]={"mean":float(np.mean(a)) if a else None,"sample_SD":float(np.std(a,ddof=1)) if len(a)>1 else None,"finite_seeds":len(a)}
  out.append(v)
 return out
write("Group_Loss_Diagnostics.json",{"post_test":True,"fixed_model_cells":cells,"score_recomposition_max_abs_delta":max_recomposition,"per_seed_class_groups":groups,"seed_summaries":across_seeds(groups,["configuration","condition","test","class"],["mean_original_raw_loss","categorical_share","binary_share","temporal_share","department_and_ward_share"]),"fixed_output_normalisation_and_head_diagnostics":alternative_scores,"alternative_seed_summaries":across_seeds(alternative_scores,["configuration","condition","test","class"],["fixed13_percent_mean_loss_change","omit_ward_head_percent_mean_loss_change"]),"within_class_client_associations":associations,"association_seed_summaries":across_seeds(associations,["configuration","condition","test","class","client","predictor"],["rho_raw_loss"]),"anonymous_illustrative_cases":cases,"controls":["All 30 checkpoint and calibration file hashes match original and replay manifests.","All 90,270 newly inferred raw scores exactly match original saved scores using unchanged batch order and loss.","No optimiser, gradient calculation, retraining, threshold refitting, score reversal or representation selection.","Component shares describe the adopted loss algebra; they are not input importance, SHAP values or causal effects.","Fixed-13 normalisation changes only the denominator for valid categorical losses; omitted targets still contribute zero.","Omitting ward output at fixed predictions is score arithmetic, not removal of duplicated inputs or retraining.","Summaries across five seeds condition on the same sessions; no independent-dataset uncertainty is established."]})
write("Input_Inventory.json",inventory)
write("Execution_Receipt.json",{"status":"POST_TEST_DIAGNOSTICS_COMPLETE_NO_FITTING","elapsed_model_seconds":time.monotonic()-start,"python":sys.version,"python_executable":sys.executable,"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"models":30,"test_cells":60,"score_rows":sum(c["sessions"] for c in cells),"distinct_test_sessions":len(samples["test_held_out"])+len(samples["test_known"]),"all_original_raw_scores_exact":all(c["original_scores_exact"] for c in cells),"score_recomposition_max_abs_delta":max_recomposition,"raw_ids_exported":False,"code_sha256":sha(Path(__file__))})
print("POST_TEST_DIAGNOSTICS_COMPLETE",time.monotonic()-start,flush=True)

