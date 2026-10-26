"""Prespecified public conformance and in-memory freshness diagnostics.

No benchmark imports, training/inference, writes to retained environment, or
exports of generated keys/seeds/shares/masks. Case outputs are deliberately small.
"""
from __future__ import annotations
import ast, base64, csv, hashlib, hmac, importlib, importlib.metadata, inspect
import io, json, math, os, struct, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONTRACT=json.loads((HERE/'contract.json').read_text())
BASE=Path('/home/mai/ehr_me06_reproduction_20261011_v1')
SOURCE=BASE/'source'
sys.dont_write_bytecode=True
sys.path.insert(0,str(SOURCE/'app'))
sys.path.insert(0,str(SOURCE/'validation_support'))
RESULTS=[]
START=datetime.now(timezone.utc).isoformat()

def sha(b): return hashlib.sha256(b).hexdigest()
def require(value):
    if not value: raise AssertionError('Public contract assertion failed')
def case(name,group,fn):
    start=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
    try:
        public=fn() or {}
        rec={'case':name,'group':group,'status':'PASS','evidence':public}
    except Exception as exc:
        rec={'case':name,'group':group,'status':'FAIL','error_type':type(exc).__name__,
             'error_category':'contract assertion or harness exception; no private values exported'}
    rec.update(start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-t)
    RESULTS.append(rec)
    print(json.dumps({'case':name,'group':group,'status':rec['status']},sort_keys=True),flush=True)

helper=importlib.import_module('flwr.common.secure_aggregation.secaggplus_utils')
client=importlib.import_module('flwr.client.mod.secure_aggregation.secaggplus_mod')
server=importlib.import_module('flwr.server.workflow.secure_aggregation.secaggplus_workflow')
fn=helper.pseudo_rand_gen
import numpy as np

def identity():
    import mai_runtime_identity as ident
    r=ident.binding_identity()
    require(r['version']==CONTRACT['expected_flower_version'])
    require(sha(Path(helper.__file__).read_bytes())==CONTRACT['expected_helper_sha256'])
    require(sha(inspect.getsource(fn).encode())==CONTRACT['expected_function_sha256'])
    require(client.pseudo_rand_gen is fn and server.pseudo_rand_gen is fn)
    require(Path(helper.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()))
    return {'version_matches':True,'helper_file_matches':True,'function_matches':True,
            'both_import_bindings_match':True,'helper_inside_environment':True,
            'environment_python':sys.version.split()[0],'remote_attestation':False}
case('SV01.identity.installed_helper_and_bindings','SV-01',identity)

def wheel_identity():
    wheel=BASE/'wheelhouse'/f"flwr-{CONTRACT['expected_flower_version']}-py3-none-any.whl"
    raw=wheel.read_bytes();require(sha(raw)==CONTRACT['expected_candidate_wheel_sha256'])
    dist=importlib.metadata.distribution('flwr');checked=0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members={n:z.read(n) for n in z.namelist() if not n.endswith('/')}
        records=[n for n in members if n.endswith('.dist-info/RECORD')];require(len(records)==1)
        for name,hash_field,size in csv.reader(io.StringIO(members[records[0]].decode())):
            if name==records[0]:require(hash_field=='' and size=='');continue
            require(name in members)
            require(hash_field=='sha256='+base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest()).decode().rstrip('='))
            require(size==str(len(members[name])))
        for n,b in members.items():
            if n.startswith('flwr/'):
                require(Path(dist.locate_file(n)).read_bytes()==b);checked+=1
        meta=[n for n in members if n.endswith('.dist-info/METADATA')]
        require(len(meta)==1 and Path(dist.locate_file(meta[0])).read_bytes()==members[meta[0]])
    return {'candidate_wheel_hash_matches':True,'wheel_record_integrity_matches':True,
            'installed_runtime_files_match_wheel':checked,'installed_metadata_matches_wheel':True}
case('SV01.identity.wheel_record_and_installed_payload','SV-01',wheel_identity)

def middleware_identity():
    import mai_runtime_identity as ident
    from flwr.common import Message, RecordDict, ConfigRecord, Context
    called=0
    def callback(msg,ctxt):
        nonlocal called
        called+=1
        return Message(content=RecordDict(),reply_to=msg)
    def msg(bid,kind='train'):
        return Message(content=RecordDict({ident.ID_RECORD:ConfigRecord({'build_id':bid})}),dst_node_id=11,message_type=kind)
    ctx=Context(run_id=1,node_id=11,node_config={'client-id':'C1'},state=RecordDict(),run_config={})
    for bad in [ident.OLD_BUILD_ID,'wrong-public-build-id']:
        try: ident.implementation_identity_mod(msg(bad),ctx,callback)
        except ident.IdentityMismatch: pass
        else: raise AssertionError('Bad identity accepted')
    require(called==0)
    reply=ident.implementation_identity_mod(msg(ident.EXPECTED_BUILD_ID),ctx,callback)
    ident.require_reply(dict(reply.content.config_records[ident.ID_RECORD]))
    require(called==1)
    for advertised,want in [(ident.OLD_BUILD_ID,False),(ident.EXPECTED_BUILD_ID,True)]:
        out=ident.implementation_identity_mod(msg(advertised,ident.QUERY_TYPE),ctx,callback)
        record=dict(out.content.config_records[ident.PROBE_RECORD])
        require(record['accepted'] is want and record['data_callback_called'] is False)
    require(called==1)
    for bad in [None,{}, {'build_id':ident.EXPECTED_BUILD_ID,'extra':True}]:
        try:ident.require_request(bad)
        except ident.IdentityMismatch:pass
        else:raise AssertionError('Malformed identity accepted')
    return {'mismatched_training_identity_rejected_before_callback':True,
            'valid_training_identity_accepted':True,'identity_queries_avoid_callback':True,
            'missing_extra_identity_fields_rejected':True,'valid_callback_count':1,
            'transport':'direct middleware public fixture; no network deployment','training_executed':False}
case('SV01.identity.middleware_positive_negative','SV-01',middleware_identity)

def scalar_reference(seed,modulus,shapes):
    width=((modulus-1).bit_length()+7)//8
    need=sum(math.prod(s) for s in shapes)*width
    stream=b''.join(hmac.new(seed,struct.pack('<Q',i),hashlib.sha256).digest() for i in range((need+31)//32))
    out=[];at=0
    for shape in shapes:
        vals=[]
        for _ in range(math.prod(shape)):
            vals.append(int.from_bytes(stream[at:at+width],'big')&(modulus-1));at+=width
        out.append(np.array(vals,dtype=np.int64).reshape(shape))
    return out
def same(a,b):return len(a)==len(b) and all(np.array_equal(x,y) for x,y in zip(a,b))
def state_same(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
PUBLIC_KEY=bytes(range(32))
SHAPES=[tuple(s) for s in CONTRACT['primitive_shapes']]

def rfc_vector():
    require(hmac.new(bytes.fromhex('0b'*20),b'Hi There',hashlib.sha256).hexdigest()==
      'b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7')
    return {'RFC4231_HMAC_SHA256_test_case_1_matches':True,'public_vector':True}
case('SV01.primitive.RFC4231_HMAC_SHA256_case1','SV-01',rfc_vector)

for exp in CONTRACT['primitive_moduli_exponents']:
    def decoder(exp=exp):
        mod=2**exp;arr=fn(PUBLIC_KEY,mod,SHAPES)
        require(same(arr,scalar_reference(PUBLIC_KEY,mod,SHAPES)))
        require(all(a.dtype==np.int64 and a.shape==s and (not a.size or (int(a.min())>=0 and int(a.max())<mod)) for a,s in zip(arr,SHAPES)))
        return {'public_scalar_reference_matches':True,'shape_dtype_range_match':True,'modulus_exponent':exp,'shape_cases':len(SHAPES)}
    case(f'SV01.primitive.scalar_reference_modulus_2^{exp}','SV-01',decoder)

def determinism_numpy():
    saved=np.random.get_state()
    try:
        np.random.seed(177);before=np.random.get_state();a=fn(PUBLIC_KEY,2**31,SHAPES);after=np.random.get_state()
        np.random.seed(91);b=fn(PUBLIC_KEY,2**31,SHAPES)
        require(state_same(before,after));require(same(a,b));require(same(a,fn(PUBLIC_KEY,2**31,SHAPES)))
    finally:np.random.set_state(saved)
    return {'same_public_seed_deterministic':True,'does_not_consume_numpy_rng':True,'independent_of_public_numpy_seed':True}
case('SV01.primitive.determinism_and_numpy_independence','SV-01',determinism_numpy)

for mod in CONTRACT['invalid_moduli']:
    def invalid(mod=mod):
        try:fn(PUBLIC_KEY,mod,[(2,)])
        except ValueError:return {'invalid_modulus_rejected':True,'public_invalid_modulus':mod}
        raise AssertionError('Invalid modulus accepted')
    case(f'SV01.primitive.reject_modulus_{mod}','SV-01',invalid)

def collision():
    p=SOURCE/'validation_support/sources/flower_v1_22_0_secaggplus_utils.py'
    raw=p.read_bytes();require(sha(raw)==CONTRACT['expected_legacy_source_sha256'])
    text=raw.decode();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='pseudo_rand_gen'];require(len(nodes)==1)
    ns={'np':np};exec(compile('from __future__ import annotations\n'+ast.get_source_segment(text,nodes[0]),str(p),'exec'),ns)
    old=ns['pseudo_rand_gen'];n=CONTRACT['legacy_constructed_collision_inputs']
    public_keys=[i.to_bytes(4,'little')*2+b'\0'*24 for i in range(n)]
    old_out=[old(k,2**31,[(16,)])[0] for k in public_keys]
    new_out=[fn(k,2**31,[(16,)])[0] for k in public_keys]
    require(all(np.array_equal(a,old_out[0]) for a in old_out))
    require(len({a.tobytes() for a in new_out})==n)
    return {'constructed_public_inputs':n,'legacy_fold_collision_reproduced':True,
            'candidate_distinguishes_constructed_inputs':True,'cryptographic_collision_resistance_proven':False}
case('SV01.primitive.legacy_constructed_fold_collision_control','SV-01',collision)

def tensor_model():
    require(np.array_equal(fn(PUBLIC_KEY,2**31,[(19,)])[0],np.concatenate(fn(PUBLIC_KEY,2**31,[(7,),(12,)]))))
    require(fn(PUBLIC_KEY,2**31,[])==[])
    n=CONTRACT['model_sized_public_array_coordinates']
    require(same(fn(PUBLIC_KEY,2**31,[(n,)]),scalar_reference(PUBLIC_KEY,2**31,[(n,)])))
    return {'tensor_boundary_stream_matches':True,'empty_tensor_list_matches':True,
            'public_model_sized_scalar_reference_matches':True,'public_coordinates':n}
case('SV01.primitive.tensor_boundaries_and_model_size','SV-01',tensor_model)

def freshness():
    from flwr.common import ConfigRecord, ndarrays_to_parameters
    from flwr.common.secure_aggregation.secaggplus_constants import Key
    from flwr.common.secure_aggregation.crypto.symmetric_encryption import bytes_to_private_key,bytes_to_public_key,generate_shared_key
    conf=CONTRACT['freshness_settings'];nids=[11,22,33]
    public_values=[np.array([-1.,0.,1.,2.],np.float32),np.array([2.,1.,-2.,0.],np.float32),np.array([0.,3.,1.,-1.],np.float32)]
    seen_pk=set();seen_sk=set();seen_seed=set();seen_pair=set();seen_enc=set();seen_reply=set()
    setup_share_numpy_unchanged=True;endpoint_matches=True
    saved=np.random.get_state()
    try:
        for session in range(CONTRACT['freshness_independent_sessions']):
            np.random.seed(CONTRACT['freshness_public_numpy_seed_reset_each_session'])
            before=np.random.get_state()
            states={nid:client.SecAggPlusState(nid=nid) for nid in nids}
            params=ConfigRecord({Key.SAMPLE_NUMBER:3,Key.SHARE_NUMBER:conf['num_shares'],Key.THRESHOLD:conf['reconstruction_threshold'],Key.CLIPPING_RANGE:conf['clipping_range'],Key.TARGET_RANGE:conf['quantization_range'],Key.MOD_RANGE:conf['modulus_range'],Key.MAX_WEIGHT:conf['max_weight']})
            setup={nid:client._setup(states[nid],params) for nid in nids}
            public_dict=ConfigRecord({str(nid):[setup[nid][Key.PUBLIC_KEY_1],setup[nid][Key.PUBLIC_KEY_2]] for nid in nids})
            shares={nid:client._share_keys(states[nid],public_dict) for nid in nids}
            setup_share_numpy_unchanged &= state_same(before,np.random.get_state())
            for nid in nids:
                st=states[nid];seen_pk.update((st.pk1,st.pk2));seen_sk.update((st.sk1,st.sk2));seen_seed.add(st.rd_seed)
            for i,a in enumerate(nids):
                for b in nids[i+1:]:
                    ka=generate_shared_key(bytes_to_private_key(states[a].sk1),bytes_to_public_key(states[b].pk1))
                    kb=generate_shared_key(bytes_to_private_key(states[b].sk1),bytes_to_public_key(states[a].pk1))
                    endpoint_matches &= ka==kb
                    seen_pair.add(ka)
                    endpoint_matches &= states[a].ss2_dict[b]==states[b].ss2_dict[a]
                    seen_enc.add(states[a].ss2_dict[b])
            for idx,nid in enumerate(nids):
                src=[];cts=[]
                for sender in nids:
                    for dest,ct in zip(shares[sender][Key.DESTINATION_LIST],shares[sender][Key.CIPHERTEXT_LIST]):
                        if dest==nid:src.append(sender);cts.append(ct)
                reply=client._collect_masked_vectors(states[nid],ConfigRecord({Key.SOURCE_LIST:src,Key.CIPHERTEXT_LIST:cts}),256,ndarrays_to_parameters([public_values[idx].copy()]))
                seen_reply.add(tuple(reply[Key.MASKED_PARAMETERS]))
            states.clear();setup.clear();shares.clear()
        sessions=CONTRACT['freshness_independent_sessions'];contexts=sessions*3
        require(len(seen_pk)==2*contexts and len(seen_sk)==2*contexts and len(seen_seed)==contexts)
        require(len(seen_pair)==sessions*3 and len(seen_enc)==sessions*3 and len(seen_reply)==contexts)
        require(setup_share_numpy_unchanged and endpoint_matches)
        counts={'sessions':sessions,'client_contexts':contexts,'unique_public_keys':len(seen_pk),
          'unique_private_key_contexts':len(seen_sk),'unique_private_seed_contexts':len(seen_seed),
          'unique_pairwise_mask_key_contexts':len(seen_pair),'unique_pairwise_share_key_contexts':len(seen_enc),
          'unique_masked_reply_contexts':len(seen_reply)}
        return {**counts,'within_category_no_reuse_observed':True,'both_endpoints_derive_matching_ECDH_HKDF_keys':True,
          'setup_share_do_not_consume_numpy_rng':True,'same_numpy_seed_reset_every_session':True,
          'installed_client_setup_share_collect_functions_executed':True,
          'secret_material_and_digests_exported':False,'crypto_entropy_or_unpredictability_proven':False,
          'unmask_executed_in_this_group':False,'transport':'direct real installed client-stage functions, in memory public fixtures'}
    finally:
        np.random.set_state(saved)
        for values in (seen_pk,seen_sk,seen_seed,seen_pair,seen_enc,seen_reply):values.clear()
case('SV06.freshness.twenty_three_client_real_generator_sessions','SV-06',freshness)

END=datetime.now(timezone.utc).isoformat()
OUT={'groups':['SV-01','SV-06'],'status':'PASS' if all(r['status']=='PASS' for r in RESULTS) else 'FAIL',
  'cases_count':len(RESULTS),'pass_count':sum(r['status']=='PASS' for r in RESULTS),
  'fail_count':sum(r['status']=='FAIL' for r in RESULTS),'not_run_count':0,
  'start_utc':START,'end_utc':END,'contract_sha256':sha((HERE/'contract.json').read_bytes()),
  'script_sha256':sha(Path(__file__).read_bytes()),'results':RESULTS,
  'benchmark_data_read':False,'training_executed':False,'inference_executed':False,
  'original_source_runtime_campaign_modified':False,'private_material_exported':False,
  'formal_security_or_randomness_certification':False}
(HERE/'results_attempt_01.json').write_text(json.dumps(OUT,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:OUT[k] for k in ['status','cases_count','pass_count','fail_count','not_run_count']},sort_keys=True),flush=True)
sys.exit(0 if OUT['status']=='PASS' else 1)
