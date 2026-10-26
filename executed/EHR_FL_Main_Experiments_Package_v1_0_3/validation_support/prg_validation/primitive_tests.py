"""Deterministic public-input tests, not a cryptographic certification."""
from __future__ import annotations
import ast,hashlib,hmac,math,struct
import numpy as np
from .build import ROOT,check_sources,function_source

def source_function(filename):
    raw=(ROOT/'sources'/filename).read_bytes()
    namespace={'np':np,'hashlib':hashlib,'hmac':hmac,'struct':struct}
    exec(compile('from __future__ import annotations\n'+function_source(raw),str(ROOT/'sources'/filename),'exec'),namespace)
    return namespace['pseudo_rand_gen']

def scalar_reference(seed,modulus,shapes):
    width=((modulus-1).bit_length()+7)//8
    need=sum(math.prod(s) for s in shapes)*width
    stream=b''.join(hmac.new(seed,struct.pack('<Q',i),hashlib.sha256).digest() for i in range((need+31)//32))
    result=[];at=0
    for shape in shapes:
        vals=[]
        for _ in range(math.prod(shape)):
            vals.append(int.from_bytes(stream[at:at+width],'big')&(modulus-1));at+=width
        result.append(np.array(vals,dtype=np.int64).reshape(shape))
    return result

def same(a,b):return len(a)==len(b) and all(np.array_equal(x,y) for x,y in zip(a,b))
def state_same(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]

def run_checks(fn=None):
    check_sources()
    if fn is None:
        from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen
        fn=pseudo_rand_gen
    key=bytes(range(32));shape=[(10,),(2,3),(),(0,),(3,0,2)];checks=[]
    def ok(value,name):
        if not value:raise AssertionError(name)
        checks.append({'name':name,'status':'PASS'})
    expected='b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7'
    ok(hmac.new(bytes.fromhex('0b'*20),b'Hi There',hashlib.sha256).hexdigest()==expected,'RFC4231_HMAC_SHA256_case_1')
    for k in [1,8,9,16,17,22,24,25,31,32,33,40,48,56,63]:
        arr=fn(key,2**k,shape)
        ok(same(arr,scalar_reference(key,2**k,shape)),f'independent_scalar_decoder_modulus_2^{k}')
        ok(all(a.dtype==np.int64 and a.shape==s and (not a.size or (a.min()>=0 and int(a.max())<2**k)) for a,s in zip(arr,shape)),f'shape_dtype_range_2^{k}')
    ok(same(fn(key,2**31,shape),fn(key,2**31,shape)),'deterministic_for_same_secret')
    saved=np.random.get_state()
    try:
        np.random.seed(177);before=np.random.get_state();a=fn(key,2**31,shape);after=np.random.get_state()
        np.random.seed(91);b=fn(key,2**31,shape)
        ok(state_same(before,after),'does_not_consume_global_numpy_rng')
        ok(same(a,b),'independent_of_public_numpy_training_seed')
    finally:np.random.set_state(saved)
    for mod in [0,1,3,100,2**63+1,2**64]:
        try:fn(key,mod,[(2,)])
        except ValueError:ok(True,f'reject_invalid_modulus_{mod}')
        else:raise AssertionError('Invalid modulus accepted')
    keys=[i.to_bytes(4,'little')*2+b'\0'*24 for i in range(16)]
    old=source_function('flower_v1_22_0_secaggplus_utils.py')
    old_results=[old(k,2**31,[(16,)])[0] for k in keys]
    new_results=[fn(k,2**31,[(16,)])[0] for k in keys]
    ok(all(np.array_equal(a,old_results[0]) for a in old_results),'legacy_constructed_fold_collision_reproduced')
    ok(len({a.tobytes() for a in new_results})==16,'candidate_distinguishes_16_constructed_public_inputs')
    single=fn(key,2**31,[(19,)])[0];split=fn(key,2**31,[(7,),(12,)])
    ok(np.array_equal(single,np.concatenate(split)),'consistent_stream_across_tensor_boundaries')
    ok(fn(key,2**31,[])==[],'empty_tensor_list')
    ok(same(fn(key,2**31,[(100_000,)]),scalar_reference(key,2**31,[(100_000,)])),'model_sized_public_array')
    # Secret-shape pairwise cancellation algebra over a public toy, not a protocol attack.
    q=2**31;public=[np.arange(10,dtype=np.int64)+j for j in (1,2,3)]
    pair={(i,j):fn(bytes([i+j])*32,q,[(10,)])[0] for i,j in [(0,1),(0,2),(1,2)]}
    masked=[]
    for i in range(3):
        a=public[i].copy()
        for j in range(3):
            if j!=i:a=(a+(1 if i>j else -1)*pair[tuple(sorted((i,j)))])%q
        masked.append(a)
    ok(np.array_equal(sum(masked)%q,sum(public)%q),'public_pairwise_mask_cancellation')
    return {'status':'PASS','checks_count':len(checks),'checks':checks,'public_test_inputs_only':True,
            'cryptographic_security_proven':False,'benchmark_data_read':False}
