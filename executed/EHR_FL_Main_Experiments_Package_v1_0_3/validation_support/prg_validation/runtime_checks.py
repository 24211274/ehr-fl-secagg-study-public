"""Run only in the newly created validation interpreters. No datasets or listeners."""
from __future__ import annotations
import argparse,ast,hashlib,hmac,importlib,importlib.metadata,inspect,json,os,sys
from pathlib import Path
from .build import manifest,sha,read_wheel
from .identity import binding_identity

def secret_source_check():
    import numpy as np
    mod=importlib.import_module('flwr.client.mod.secure_aggregation.secaggplus_mod')
    sym=importlib.import_module('flwr.common.secure_aggregation.crypto.symmetric_encryption')
    setup=inspect.getsource(mod._setup);share=inspect.getsource(mod._share_keys)
    gen=inspect.getsource(sym.generate_key_pairs);derive=inspect.getsource(sym.generate_shared_key)
    if 'os.urandom(32)' not in share or 'generate_key_pairs()' not in setup or 'ec.generate_private_key(' not in gen:
        raise ValueError('Unexpected secret generation source; review required')
    if 'HKDF(' not in derive or 'length=32' not in derive or 'urlsafe_b64encode' not in derive:
        raise ValueError('Unexpected shared-key derivation')
    saved=np.random.get_state()
    try:
        np.random.seed(177);k1,p1=sym.generate_key_pairs()
        np.random.seed(177);k2,p2=sym.generate_key_pairs()
        s1=sym.generate_shared_key(k1,p2);s2=sym.generate_shared_key(k2,p1)
        if sym.public_key_to_bytes(p1)==sym.public_key_to_bytes(p2) or s1!=s2 or len(s1)!=44:
            raise ValueError('Fresh key/agreement sanity check failed')
        # Do not export private/public keys, derived keys or their fingerprints.
        return {'status':'PASS','private_mask_seed_source':'os.urandom(32)',
                'pairwise_key_source':'upstream EC key generation + ECDH + HKDF-SHA256',
                'pairwise_prg_input':'44-byte URL-safe base64 of 32-byte derived key; unchanged',
                'repeating_public_numpy_seed_did_not_repeat_generated_keypairs_in_this_check':True,
                'key_agreement_sanity_pass':True,'secret_material_exported':False,
                'entropy_or_cryptographic_security_proof':False,
                'source_sha256':{n:sha(s.encode()) for n,s in [('setup',setup),('share_keys',share),('key_pairs',gen),('derive',derive)]}}
    finally:np.random.set_state(saved)

def installed_wheel_check(wheel:Path,want_hash:str):
    content=read_wheel(wheel,want_hash)
    dist=importlib.metadata.distribution('flwr');checked=0
    for name,data in content.items():
        if name.startswith('flwr/'):
            path=Path(dist.locate_file(name)).resolve()
            if not path.is_relative_to(Path(sys.prefix).resolve()) or not path.is_file() or sha(path.read_bytes())!=sha(data):
                raise ValueError('Installed Flower differs from selected wheel')
            checked+=1
    return {'status':'PASS','flower_runtime_files_checked':checked,'outside_venv_paths':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reference','candidate'],required=True)
    p.add_argument('--wheel');p.add_argument('--wheel-sha256');p.add_argument('--out',required=True)
    p.add_argument('--identity-only',action='store_true');a=p.parse_args()
    identity=binding_identity(a.mode)
    report={'status':'PASS','identity':identity,'python':sys.version.split()[0],
            'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'benchmark_data_read':False,'deployment_engine_used':False,'main_experiments_frozen':False,
            'privacy_claim_validated':False}
    if not a.identity_only:
        report['installed_wheel']=installed_wheel_check(Path(a.wheel),a.wheel_sha256)
        report['secret_sources']=secret_source_check()
        from .public_protocol import run_oracle
        report['public_protocol']=run_oracle(a.mode)
        if a.mode=='candidate':
            from .primitive_tests import run_checks
            report['primitive_checks']=run_checks()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
if __name__=='__main__':main()
