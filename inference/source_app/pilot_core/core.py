"""Deterministic pilot helpers. No cryptographic protocol is implemented here."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence
import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def derived_seed(*parts: object) -> int:
    b = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'little') % 2**32


def validate_schema(path: Path, expected_hash: str | None = None) -> dict:
    """Reject an unapproved/ambiguous schema; never silently choose 107 or 119."""
    if expected_hash and sha256_file(path) != expected_hash:
        raise ValueError('Schema SHA-256 mismatch')
    s = json.loads(path.read_text(encoding='utf-8'))
    if s.get('approved') is not True:
        raise ValueError('Schema is not approved by the researcher')
    fields = s.get('categorical_fields', [])
    if len(fields) != 13 or len({f['name'] for f in fields}) != 13:
        raise ValueError('Expected thirteen distinct, ordered categorical fields')
    expected = ['action','event_category','department','role','resource_type',
                'source_application','device_type','network_zone','location_type',
                'purpose_of_use','relationship_type','outcome','ward_or_location']
    if [f['name'] for f in fields] != expected:
        raise ValueError('Categorical field order disagrees with the protocol')
    for f in fields:
        cats = f['categories']
        if not cats or len(cats) != len(set(cats)):
            raise ValueError('Empty or repeated category names')
        for ix in f.get('masked_target_indices', []):
            if not isinstance(ix, int) or not 0 <= ix < len(cats):
                raise ValueError('Invalid masked categorical target index')
    if s.get('binary_fields') != ['assigned_patient','break_glass','on_shift']:
        raise ValueError('Binary field order mismatch')
    expected_time = ['hour_sin','hour_cos','weekday_sin','weekday_cos',
                     'log1p_seconds_since_previous_event','log1p_seconds_since_session_start']
    if s.get('temporal_fields') != expected_time:
        raise ValueError('Temporal field order mismatch')
    dim = sum(len(f['categories']) for f in fields) + 9
    if s.get('input_dim') != dim:
        raise ValueError(f'Schema dimension must be {dim}, not {s.get("input_dim")}')
    return s


def calibrate(scores: Sequence[float]) -> dict:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError('Calibration requires finite, nonempty 1-D validation scores')
    centre = float(np.median(x))
    mad = float(np.median(np.abs(x-centre)))
    scale = 1.482602218505602 * mad
    method = 'scaled_MAD'
    if scale <= 1e-12:
        q25,q75 = np.quantile(x, [0.25,0.75], method='linear')
        scale = float((q75-q25)/1.3489795003921634)
        method = 'scaled_IQR'
    degenerate = scale <= 1e-12
    if degenerate:
        scale,method = 1.0,'degenerate_unit_scale'
    return {'centre':centre,'scale':scale,'method':method,'degenerate':bool(degenerate),
            'raw_thresholds':{str(q):float(np.quantile(x,q,method='linear'))
                              for q in [0.98,0.99,0.995]},
            'quantile_method':'linear','alert_operator':'>','n':int(x.size)}


def fedavg(arrays: Sequence[np.ndarray], weights: Sequence[int]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    a = [np.asarray(x,dtype=np.float64) for x in arrays]
    if not a or len(a) != len(w) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Positive finite client weights required')
    if any(x.shape != a[0].shape or not np.isfinite(x).all() for x in a):
        raise ValueError('Client array shape/nonfinite mismatch')
    return sum(x*wk for x,wk in zip(a,w)) / w.sum()


def representation_reference(arrays: Sequence[np.ndarray], weights: Sequence[int],
                             seed: int, c: float=8.0, q: int=2**22,
                             maximum_weight: int=4096, modulus: int=2**31) -> tuple[np.ndarray, dict]:
    """Plain numeric audit oracle, NOT Flower and NOT secure aggregation.

    Uses independently written stochastic grid rounding of weighted vectors.
    An installed Flower quantiser/protocol must separately pass parity/replay gates.
    With integer weights and Q divisible by max_weight the ratio quantisation is exact.
    """
    reference = fedavg(arrays, weights)
    w = np.asarray(weights,dtype=np.float64)
    if np.any(w > maximum_weight) or np.any(w != np.floor(w)):
        raise ValueError('Out-of-range/noninteger session weights')
    if q % maximum_weight != 0 or c <= 0:
        raise ValueError('This oracle assumes exact ratio quantisation and positive C')
    ratio = np.rint(w/maximum_weight*q)/q
    rng = np.random.default_rng(seed)
    ints = []
    maximum_abs = 0.0
    saturated = 0
    for x, r in zip(arrays, ratio):
        y = np.asarray(x,dtype=np.float64)*r
        maximum_abs = max(maximum_abs, float(np.max(np.abs(y))))
        saturated += int(np.count_nonzero(np.abs(y)>c))
        z = (np.clip(y,-c,c)+c)/(2*c)*q
        lo = np.floor(z)
        ints.append((lo+(rng.random(z.shape) < z-lo)).astype(np.int64))
    if saturated:
        raise ValueError('Weighted parameters saturate the configured clipping range')
    # No masks or secrets are generated; this is an ordinary integer sum.
    integer_sum = np.sum(ints,axis=0,dtype=np.int64)
    k=len(arrays)
    if modulus <= 0 or (modulus & (modulus-1)) != 0 or k*q >= modulus:
        raise ValueError('Insufficient/non-power-of-two modular range for this oracle')
    reconstructed=(integer_sum.astype(np.float64)*2*c/q-k*c)/ratio.sum()
    bound=k*2*c/(q-1)/(w.sum()/maximum_weight)+2e-6+1e-5*float(np.max(np.abs(reference)))
    error=float(np.max(np.abs(reconstructed-reference)))
    return reconstructed, {'max_abs_error':error,'analytic_acceptance_bound':float(bound),
            'max_abs_weighted_parameter':maximum_abs,'saturated_coordinates':saturated,
            'integer_sum_sha256':hashlib.sha256(integer_sum.tobytes()).hexdigest(),
            'modulus':int(modulus),
            'pass':error<=bound,'status':'PLAIN_REPRESENTATION_ORACLE_ONLY_NOT_SECAGG'}


def js_divergence(p: Sequence[float],q: Sequence[float]) -> float:
    p,q=np.asarray(p,dtype=float),np.asarray(q,dtype=float)
    if p.shape!=q.shape or (p<0).any() or (q<0).any() or p.sum()<=0 or q.sum()<=0:
        raise ValueError('Invalid distributions')
    p,q=p/p.sum(),q/q.sum()
    m=(p+q)/2
    def kl(x):
        ok=x>0
        return float(np.sum(x[ok]*np.log2(x[ok]/m[ok])))
    return .5*(kl(p)+kl(q))
