"""Approved 119-wide event encoding and train-only relative-time scaling.

Only training and validation events are numerically interpreted. Test rows share
one raw file: their bytes are read to verify the source hash and their session IDs
are inspected to skip them; their feature values are not encoded or analysed.
This is an offline development builder, NOT a Flower workflow or secure statistics
protocol. Client summary combination is plaintext, as stated in the pilot policy.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from zoneinfo import ZoneInfo
import numpy as np
from pilot_core.core import validate_schema
from .io_utils import BenchmarkSource, id_set_hash, sha256_file, write_csv, write_json, seal_artefacts

SCHEMA_SHA = '4cca92b4ea5aa50474aeb305e127d6136e4f0920843b40e8ba51d4dbb2ff9792'
ALLOWED_PARTITIONS = ('training', 'validation')
KNOWN_PARTITIONS = {'training', 'validation', 'test_known', 'test_held_out'}
CLIENTS = {
    'C1': 'community_and_emergency_hospital',
    'C2': 'large_tertiary_general_hospital',
    'C3': 'specialist_referral_hospital',
}
EXPECTED = {
    ('C1','training'): (3609,36000), ('C1','validation'): (1214,12004),
    ('C2','training'): (3548,36004), ('C2','validation'): (1198,12003),
    ('C3','training'): (3676,36001), ('C3','validation'): (1210,12006),
}
LABEL_COLUMNS_READ = ['session_id','hospital_id','partition','primary_class','binary_label']
LINKAGE_COLUMNS = ['session_id','hospital_id','event_id','sequence_position','timestamp_utc']
FORBIDDEN_TENSOR_COLUMNS = [
    'event_id','session_id','actor_id','patient_id','encounter_id','shift_id','resource_id',
    'hospital_id','event_code','recorded_at_utc','binary_label','primary_class','threat_family',
    'legitimate_subtype','affected_event_ids','scenario_id','template_id','generator_seed',
    'partition','injection_start_utc','injection_end_utc','campaign_id','pair_id',
    'campaign_day_count','campaign_day_index',
]


@dataclass
class Session:
    session_id: str
    client: str
    hospital_id: str
    partition: str
    event_ids: list[str]
    timestamp_us: np.ndarray
    categorical: np.ndarray
    categorical_valid: np.ndarray
    binary: np.ndarray
    raw_time: np.ndarray


def approved_schema(path: Path) -> dict:
    s = validate_schema(Path(path), SCHEMA_SHA)
    offset = 0
    for f in s['categorical_fields']:
        if f['one_hot_start_inclusive'] != offset:
            raise ValueError('Noncontiguous categorical slices')
        offset += len(f['categories'])
        if f['one_hot_stop_exclusive'] != offset:
            raise ValueError('Wrong categorical slice stop')
        for key, token in [('unknown_index','__UNK__'),('missing_index','__MISSING__')]:
            if f['categories'][f[key]] != token:
                raise ValueError('Reserved category index disagrees with schema')
        if set(f['masked_target_indices']) != {f['unknown_index'], f['missing_index']}:
            raise ValueError('Unrecognised target masking policy')
    if offset != 110 or s['input_dim'] != 119:
        raise ValueError('This release expects approved schema 1.0.0 with 119 dimensions')
    return s


def parse_boolean(value: str) -> float:
    # Case-insensitive Boolean serialisations and explicit 0/1 only; no imputation.
    if value in ('True','TRUE','true','1'):
        return 1.0
    if value in ('False','FALSE','false','0'):
        return 0.0
    raise ValueError('Missing or invalid Boolean; no silent imputation')


def parse_timestamp(value: str) -> datetime:
    # Exact source timestamps have microsecond precision and an explicit UTC offset.
    if not isinstance(value,str) or not re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})', value):
        raise ValueError('Timestamp must be explicit-offset ISO 8601 with <=6 fractional digits')
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError('Naive timestamp')
    return dt.astimezone(timezone.utc)


def epoch_us(dt: datetime) -> int:
    d = dt - datetime(1970,1,1,tzinfo=timezone.utc)
    return (d.days*86400 + d.seconds)*1_000_000 + d.microseconds


def format_us(us: int) -> str:
    from datetime import timedelta
    return (datetime(1970,1,1,tzinfo=timezone.utc) + timedelta(microseconds=int(us))).isoformat(timespec='microseconds')


def temporal_values(times: list[datetime], calendar_timezone: str) -> tuple[np.ndarray,np.ndarray]:
    if not times:
        raise ValueError('Empty session')
    us = np.asarray([epoch_us(t) for t in times], dtype='<i8')
    gaps_us = np.concatenate([np.array([0],dtype=np.int64), np.diff(us)])
    if np.any(gaps_us < 0):
        raise ValueError('Negative elapsed time after sequence ordering')
    local = [t.astimezone(ZoneInfo(calendar_timezone)) for t in times]
    hours = np.asarray([t.hour+t.minute/60+t.second/3600+t.microsecond/3_600_000_000 for t in local], dtype=np.float64)
    days = np.asarray([t.weekday() for t in local], dtype=np.float64)
    out = np.column_stack([
        np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24),
        np.sin(2*np.pi*days/7), np.cos(2*np.pi*days/7),
        np.log1p(gaps_us.astype(np.float64)/1_000_000),
        np.log1p((us-us[0]).astype(np.float64)/1_000_000),
    ]).astype('<f8')
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite temporal values')
    return us, out


def encode_category(value: str, field: dict, mapping: dict[str,int]) -> tuple[int,bool]:
    if not isinstance(value, str):
        raise ValueError('Malformed CSV categorical value, expected a literal string')
    if value in ('__UNK__','__MISSING__'):
        raise ValueError('Raw literal conflicts with reserved synthetic encoding token')
    idx = field['missing_index'] if value == '' else mapping.get(value, field['unknown_index'])
    return idx, idx not in field['masked_target_indices']


def load_allowed(source: BenchmarkSource, schema: dict) -> tuple[dict[str,Session],dict]:
    """Read label metadata for partition assignment; skip test features before parsing."""
    all_assignments, selected = {}, {}
    hospital_to_client = {v:k for k,v in CLIENTS.items()}
    with source.text('labels.csv') as f:
        reader = csv.DictReader(f)
        if not set(LABEL_COLUMNS_READ).issubset(reader.fieldnames or []):
            raise ValueError('Missing partition metadata columns')
        for row in reader:
            sid, part = row['session_id'], row['partition']
            if not sid or sid in all_assignments or part not in KNOWN_PARTITIONS:
                raise ValueError('Invalid/duplicated session assignment or unknown partition')
            all_assignments[sid] = part
            if part not in ALLOWED_PARTITIONS:
                continue  # no test class or feature analysis
            hid = row['hospital_id']
            if hid not in hospital_to_client:
                raise ValueError('Unknown client')
            if row['primary_class'] != 'routine_legitimate' or parse_boolean(row['binary_label']) != 0:
                raise ValueError('Only routine legitimate sessions permitted in development partitions')
            selected[sid] = (hospital_to_client[hid], hid, part)
    training_ids = [sid for sid,(_,_,p) in selected.items() if p=='training']
    if id_set_hash(training_ids) != schema['training_selection']['session_id_set_sha256']:
        raise ValueError('Training membership mismatch')
    cat_fields = schema['categorical_fields']
    cat_names = [f['name'] for f in cat_fields]
    keep = LINKAGE_COLUMNS + cat_names + schema['binary_fields']
    if set(cat_names + schema['binary_fields']) & set(FORBIDDEN_TENSOR_COLUMNS):
        raise ValueError('Forbidden feature in schema')
    groups = {sid:[] for sid in selected}
    seen_events, total, skipped = set(), 0, 0
    with source.text('events.csv') as f:
        reader = csv.DictReader(f)
        if not set(keep).issubset(reader.fieldnames or []):
            raise ValueError('Missing source features')
        for row in reader:
            total += 1
            sid = row['session_id']
            if sid not in all_assignments:
                raise ValueError('Unlabelled event session')
            if sid not in selected:
                skipped += 1
                continue  # drop before timestamp, category, Boolean or event-code processing
            if row['hospital_id'] != selected[sid][1]:
                raise ValueError('Event/label client mismatch')
            eid = row['event_id']
            if not eid or eid in seen_events:
                raise ValueError('Duplicated or empty selected event ID')
            seen_events.add(eid)
            groups[sid].append({k:row[k] for k in keep})
    if total != 181805 or len(all_assignments) != 17464:
        raise ValueError('Unexpected source dimensions')
    maps = [{cat:i for i,cat in enumerate(f['categories'])} for f in cat_fields]
    result, training_seen = {}, [set() for _ in cat_fields]
    # Explicit stable ordering defines floating-point summation and artefact row order.
    for sid in sorted(groups, key=lambda s:(selected[s][0],selected[s][2],s)):
        rows = groups[sid]
        if not rows:
            raise ValueError('Selected session without events')
        for row in rows:
            raw_pos = row['sequence_position']
            if not re.fullmatch(r'0|[1-9][0-9]*', raw_pos):
                raise ValueError('Noncanonical sequence position')
            row['_pos'] = int(raw_pos)
            row['_time'] = parse_timestamp(row['timestamp_utc'])
        rows.sort(key=lambda r:(r['_pos'],r['_time'],r['event_id']))
        if [r['_pos'] for r in rows] != list(range(len(rows))):
            raise ValueError('Sequence positions must be unique, contiguous, and start at zero')
        n = len(rows)
        cat = np.empty((n,13),dtype='<i8')
        valid = np.empty((n,13),dtype=np.bool_)
        binary = np.empty((n,3),dtype='<f4')
        for i,row in enumerate(rows):
            for j,field in enumerate(cat_fields):
                cat[i,j], valid[i,j] = encode_category(row[field['name']],field,maps[j])
                if selected[sid][2] == 'training' and row[field['name']] != '':
                    training_seen[j].add(row[field['name']])
            for j,name in enumerate(schema['binary_fields']):
                binary[i,j] = parse_boolean(row[name])
        us,time = temporal_values([r['_time'] for r in rows], schema['preprocessing_policy_reference']['calendar_timezone'])
        client,hid,part = selected[sid]
        result[sid] = Session(sid,client,hid,part,[r['event_id'] for r in rows],us,cat,valid,binary,time)
    for j,field in enumerate(cat_fields):
        original = set(field['categories'])-{'__UNK__','__MISSING__'}
        if training_seen[j] != original:
            raise ValueError(f"Training vocabulary mismatch for {field['name']}")
    for key,expected in EXPECTED.items():
        subset = [s for s in result.values() if (s.client,s.partition)==key]
        if (len(subset),sum(len(s.event_ids) for s in subset)) != expected:
            raise ValueError(f'Selected source count mismatch: {key}')
    for client in CLIENTS:
        tr = [s for s in result.values() if s.client==client and s.partition=='training']
        va = [s for s in result.values() if s.client==client and s.partition=='validation']
        if max(int(s.timestamp_us[-1]) for s in tr) >= min(int(s.timestamp_us[0]) for s in va):
            raise ValueError('Training/validation chronological boundary overlap')
    return result, {
        'raw_rows_streamed':total, 'selected_rows':len(seen_events), 'skipped_test_rows':skipped,
        'selected_sessions':len(result), 'skipped_test_sessions':len(all_assignments)-len(result),
        'test_feature_rows_encoded':0, 'test_features_used_for_fit':False,
        'test_label_values_analysed':False, 'approved_training_vocabulary_exact_match':True,
        'training_membership_sha256':id_set_hash(training_ids),
        'source_label_columns_used':LABEL_COLUMNS_READ,
        'source_feature_columns_used':cat_names+schema['binary_fields']+['timestamp_utc'],
        'linkage_order_only_columns':LINKAGE_COLUMNS,
        'raw_test_bytes_read_for_hashing_and_stream_filtering':True,
        'selected_row_filtering_or_imputation': 'none; invalid development rows cause failure',
    }


def summarise(values: np.ndarray) -> dict:
    x = np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or x.shape[0]<1 or not np.isfinite(x).all():
        raise ValueError('Need finite nonempty (n,2) relative-time training values')
    mean = x.mean(axis=0,dtype=np.float64)
    m2 = np.square(x-mean).sum(axis=0,dtype=np.float64)
    return {'count':int(x.shape[0]),'mean':mean.tolist(),'M2':m2.tolist()}


def combine_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError('No training summaries')
    n,mean,m2 = 0,np.zeros(2,dtype=np.float64),np.zeros(2,dtype=np.float64)
    for s in summaries:
        k = s['count']
        mu,mm = np.asarray(s['mean'],dtype=np.float64),np.asarray(s['M2'],dtype=np.float64)
        if type(k) is not int or k<1 or mu.shape!=(2,) or mm.shape!=(2,) or not np.isfinite(mu).all() or not np.isfinite(mm).all() or np.any(mm<0):
            raise ValueError('Invalid client summary')
        delta = mu-mean
        total = n+k
        m2 = m2 + mm + delta**2 * (float(n)*float(k)/total)
        mean = mean + delta*(k/total)
        n = total
    variance = m2/n  # approved population variance, ddof=0
    if not np.isfinite(variance).all() or np.any(variance<0):
        raise ValueError('Invalid variance')
    scale = np.sqrt(variance)
    constant = scale==0
    scale[constant] = 1.0
    return {'count':n,'mean':mean.tolist(),'M2':m2.tolist(),'variance':variance.tolist(),
            'scale':scale.tolist(),'constant_columns':constant.tolist()}


def fit_scaler(sessions: dict[str,Session], schema_hash: str) -> tuple[dict,dict]:
    chunks,client_stats = [],[]
    for client in CLIENTS:
        tr = sorted((s for s in sessions.values() if s.client==client and s.partition=='training'),key=lambda s:s.session_id)
        if not tr:
            raise ValueError('Missing client training data')
        x = np.concatenate([s.raw_time[:,4:] for s in tr],axis=0).astype(np.float64)
        chunks.append(x)
        client_stats.append({'client':client,'hospital_id':CLIENTS[client],
                             'session_count':len(tr),'session_id_set_sha256':id_set_hash(s.session_id for s in tr),
                             **summarise(x)})
    merged = combine_summaries(client_stats)
    # Independent pooled check in the offline builder; NOT network transfer or a deployment claim.
    pooled = np.concatenate(chunks,axis=0)
    pooled_mean,pooled_var = pooled.mean(axis=0,dtype=np.float64),pooled.var(axis=0,ddof=0,dtype=np.float64)
    mean_error=float(np.max(np.abs(pooled_mean-merged['mean'])))
    var_error=float(np.max(np.abs(pooled_var-merged['variance'])))
    if not np.allclose(merged['mean'],pooled_mean,atol=1e-12,rtol=1e-12) or not np.allclose(merged['variance'],pooled_var,atol=1e-12,rtol=1e-12):
        raise ValueError('Client summaries disagree with pooled numerical check')
    fitted = {
        'version':'1.0.0', 'fitted':True, 'approved_schema_sha256':schema_hash,
        'features':['log1p_seconds_since_previous_event','log1p_seconds_since_session_start'],
        'fit_partition':'training', 'training_sessions':sum(c['session_count'] for c in client_stats), 'training_events':merged['count'],
        'dtype':'float64', 'ddof':0, 'local_order':'lexicographic session_id, then sequence_position',
        'client_merge_order':list(CLIENTS),
        'method':'client count/mean/M2, merged with fixed-order parallel variance formula',
        'summary_privacy':'plaintext summaries; not SecAgg, not differential privacy',
        'execution':'offline builder with client-separated calculations; no network protocol executed',
        'reused_across':['A','B','C','reference','noniid'],
        'schema_file_modified':False,
        **merged,
        'client_summaries':client_stats,
    }
    checks = {'pooled_check':True,'max_abs_mean_error':mean_error,'max_abs_variance_error':var_error,
              'tolerance':{'atol':1e-12,'rtol':1e-12}}
    return fitted,checks


def event_features(s: Session, schema: dict, scaler: dict) -> np.ndarray:
    n = len(s.event_ids)
    x = np.zeros((n,schema['input_dim']),dtype='<f4')
    ii = np.arange(n)
    for j,field in enumerate(schema['categorical_fields']):
        x[ii,field['one_hot_start_inclusive']+s.categorical[:,j]] = 1.0
    x[:,110:113] = s.binary
    time = s.raw_time.copy()
    time[:,4:] = (time[:,4:]-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])
    x[:,113:119] = time.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite encoded tensor')
    return x


def build(source_path: Path, package_root: Path, output: Path) -> dict:
    package_root,output = Path(package_root).resolve(),Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Output already exists: use a new output path; no silent overwrite')
    schema_path=package_root/'config/encoding_schema.json'
    schema=approved_schema(schema_path)
    cfg=json.loads((package_root/'config/pilot_settings_schema_bound_v1_2.json').read_text())
    if cfg['schema']['sha256'] != SCHEMA_SHA or cfg['clients'] != CLIENTS:
        raise ValueError('Pilot/schema/client binding mismatch')
    if cfg['preprocessing'] != schema['preprocessing_policy_reference']:
        raise ValueError('Preprocessing policy differs from approved schema')
    with BenchmarkSource(source_path,package_root/'manifests/source_file_manifest.csv') as source:
        verified = source.verify()
        sessions, selection = load_allowed(source,schema)
        scaler, scaler_checks = fit_scaler(sessions,SCHEMA_SHA)
        output.parent.mkdir(parents=True,exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix='.'+output.name+'-building-',dir=output.parent))
        try:
            write_json(temp/'relative_time_scaler.json',scaler)
            scaler_sha=sha256_file(temp/'relative_time_scaler.json')
            shutil.copy2(schema_path,temp/'encoding_schema.json')
            write_json(temp/'source_verification.json',{'files':verified,'schema_sha256':SCHEMA_SHA})
            fields_manifest=[]
            for f in schema['categorical_fields']:
                for j,c in enumerate(f['categories']):
                    fields_manifest.append({'index':f['one_hot_start_inclusive']+j,'group':'categorical',
                                            'source_or_derived_field':f['name'],'value':c})
            for j,f in enumerate(schema['binary_fields']):
                fields_manifest.append({'index':110+j,'group':'binary','source_or_derived_field':f,'value':'0/1'})
            for j,f in enumerate(schema['temporal_fields']):
                fields_manifest.append({'index':113+j,'group':'temporal','source_or_derived_field':f,
                                        'value':'deterministic cyclic' if j<4 else 'train-standardised log1p seconds'})
            write_csv(temp/'feature_index.csv',fields_manifest,['index','group','source_or_derived_field','value'])
            manifest, linkage, summaries, coverage = [],[],[],[]
            membership_index={}
            for client in CLIENTS:
                for part in ALLOWED_PARTITIONS:
                    group=sorted((s for s in sessions.values() if s.client==client and s.partition==part),key=lambda s:s.session_id)
                    out_dir=temp/client/part
                    out_dir.mkdir(parents=True)
                    length=np.asarray([len(s.event_ids) for s in group],dtype='<i8')
                    offsets=np.concatenate([np.array([0],dtype='<i8'),np.cumsum(length,dtype=np.int64)])
                    x=np.concatenate([event_features(s,schema,scaler) for s in group],axis=0)
                    cat=np.concatenate([s.categorical for s in group],axis=0)
                    valid=np.concatenate([s.categorical_valid for s in group],axis=0)
                    np.save(out_dir/'features.npy',x,allow_pickle=False)
                    np.save(out_dir/'categorical_targets.npy',cat,allow_pickle=False)
                    np.save(out_dir/'categorical_valid.npy',valid,allow_pickle=False)
                    np.save(out_dir/'offsets.npy',offsets,allow_pickle=False)
                    np.save(out_dir/'lengths.npy',length,allow_pickle=False)
                    for j,s in enumerate(group):
                        item={'session_id':s.session_id,'client':client,'hospital_id':s.hospital_id,'partition':part,
                              'session_index':j,'event_offset':int(offsets[j]),'length':int(length[j]),
                              'start_utc':format_us(s.timestamp_us[0]),'end_utc':format_us(s.timestamp_us[-1]),
                              'features_path':f'{client}/{part}/features.npy'}
                        manifest.append(item); membership_index[s.session_id]=item
                        for pos,(eid,us) in enumerate(zip(s.event_ids,s.timestamp_us)):
                            linkage.append({'event_id':eid,'session_id':s.session_id,'client':client,'partition':part,
                                            'session_index':j,'sequence_position':pos,'event_offset':int(offsets[j])+pos,
                                            'timestamp_utc':format_us(us)})
                    for j,f in enumerate(schema['categorical_fields']):
                        coverage.append({'client':client,'partition':part,'field':f['name'],'events':len(x),
                                         'unknown':int(np.sum(cat[:,j]==f['unknown_index'])),
                                         'missing':int(np.sum(cat[:,j]==f['missing_index'])),
                                         'valid_categorical_targets':int(valid[:,j].sum())})
                    summary={'client':client,'partition':part,'sessions':len(group),'events':int(len(x)),
                             'feature_shape':list(x.shape),'dtype':str(x.dtype),
                             'min_length':int(length.min()),'max_length':int(length.max()),
                             'schema_sha256':SCHEMA_SHA,'scaler_sha256':scaler_sha}
                    summaries.append(summary)
            write_csv(temp/'session_manifest.csv',manifest,list(manifest[0]))
            write_csv(temp/'event_linkage.csv',linkage,list(linkage[0]))
            write_csv(temp/'category_coverage.csv',coverage,list(coverage[0]))
            (temp/'pilot_membership').mkdir()
            membership_checks=[]
            for name in ['pilot_membership.csv','overfit_32_membership.csv']:
                src=package_root/'manifests/pilot'/name
                with src.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
                if len({r['session_id'] for r in rows}) != len(rows):
                    raise ValueError('Duplicate pilot membership')
                for r in rows:
                    if r['session_id'] not in membership_index:
                        raise ValueError('Pilot session absent or forbidden')
                    m=membership_index[r['session_id']]
                    if any(r[k]!=m[k] for k in ['client','hospital_id','partition']):
                        raise ValueError('Pilot membership changed')
                shutil.copy2(src,temp/'pilot_membership'/name)
                membership_checks.append({'file':name,'rows':len(rows),'sha256':sha256_file(src),'all_resolved':True})
            if source.verify()!=verified or sha256_file(schema_path)!=SCHEMA_SHA:
                raise ValueError('Source changed during build')
            status={
                'status':'PREPROCESSING_BUILT_VALIDATION_REQUIRED_NOT_MAIN_FROZEN',
                'pipeline_version':'1.0.0','schema_version':schema['schema_version'],
                'schema_sha256':SCHEMA_SHA,'scaler_sha256':scaler_sha,
                'source_policy':'original eleven-file benchmark; not the two Kaggle EHR source CSVs',
                'selection':selection,'groups':summaries,'scaler_checks':scaler_checks,
                'pilot_membership_checks':membership_checks,
                'source_csvs_changed':False,'schema_changed':False,'data_augmented':False,
                'training_run':False,'test_tensors_created':False,'test_metrics_computed':False,
                'flower_executed':False,'secagg_executed':False,'main_experiments_frozen':False,
                'forbidden_tensor_columns':FORBIDDEN_TENSOR_COLUMNS,
                'linkage_note':'Identifiers are stored only in external linkage manifests, never numeric input arrays',
                'limits':[
                    'Structural/encoding validity, not clinical realism or independent ground-truth validation.',
                    'Train/validation partitions contain routine legitimate sessions only; no test tensors or threat scores here.',
                    'Earlier test-data exposure elsewhere in the project is not erased by this stage.',
                    'Client summary statistics are not cryptographically protected in this offline builder.',
                    'No inter-client network privacy or Flower execution is demonstrated by this step.',
                ],
            }
            write_json(temp/'preprocessing_summary.json',status)
            seal_artefacts(temp)
            temp.rename(output)
            return status
        except BaseException:
            # Keep diagnostics at the caller; partial tensors must not look like a successful dataset.
            shutil.rmtree(temp,ignore_errors=True)
            raise
