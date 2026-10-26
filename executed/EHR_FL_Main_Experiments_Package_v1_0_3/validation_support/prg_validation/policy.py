"""Pure validation only; no cryptographic primitive is supplied or replaced here."""
from __future__ import annotations
import math
import numpy as np
from .settings import scfg


def validate_public_fit_metrics(metrics, expected_input=None, toy=False):
    allowed={'client','input_digest','steps','draws'}
    if set(metrics)!=allowed: raise ValueError('Training reply contains missing or prohibited diagnostic fields')
    if metrics['client'] not in ('C1','C2','C3'): raise ValueError('Unknown client acknowledgement')
    if not isinstance(metrics['input_digest'],str) or len(metrics['input_digest'])!=64: raise ValueError('Invalid public global fingerprint')
    if expected_input is not None and metrics['input_digest']!=expected_input: raise ValueError('Stale public global input')
    if type(metrics['steps'])!=int or type(metrics['draws'])!=int: raise ValueError('Invalid step/count type')
    if not toy and (metrics['steps']!=4 or metrics['draws']!=256): raise ValueError('Training budget mismatch')
    if toy and (metrics['steps']<0 or metrics['draws']<1): raise ValueError('Invalid public toy budget')


def validate_cohort(sent_ids,reply_ids):
    if len(sent_ids)!=3 or len(set(sent_ids))!=3 or len(reply_ids)!=3 or len(set(reply_ids))!=3 or set(sent_ids)!=set(reply_ids):
        raise RuntimeError('Exactly three matching participants required; no fallback')


def local_range_check(arrays):
    conf=scfg();c=conf['unweighted_parameter_absolute_limit']
    if not arrays: raise ValueError('Empty client model')
    for a in arrays:
        if not isinstance(a,np.ndarray) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid local model array')
        if np.any(np.abs(a)>c):raise ValueError('Local parameter range exceeded; clipping not accepted')
    # No parameter maximum, norm, digest or individual coordinates returned.
    return True


def restored_float32(repeated,spec):
    """The 1.22 workflow repeats ONE unmasked aggregate in legacy FitRes objects."""
    if len(repeated)!=3 or not spec:raise ValueError('Expected three aggregate carriers')
    first=repeated[0]
    for arrs in repeated:
        if len(arrs)!=len(spec):raise ValueError('Invalid aggregate length')
        for a,s,b in zip(arrs,spec,first):
            if not isinstance(a,np.ndarray) or list(a.shape)!=s['shape'] or a.dtype not in (np.float32,np.float64) or not np.isfinite(a).all():raise ValueError('Invalid recovered aggregate')
            if not np.array_equal(a,b):raise ValueError('SecAgg callback carriers are not the same aggregate')
    converted=[np.asarray(a,dtype=np.float32).copy() for a in first]
    if any(not np.isfinite(a).all() for a in converted):raise ValueError('Aggregate float32 overflow')
    delta=max(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) for a,b in zip(first,converted))
    return converted,delta


def conservative_quantization_bound(weights,max_abs_parameter=8.0):
    """A priori coordinate error bound for integer-count weights, not a loss tolerance.

    Stock scaling uses dq_ratio = round(n/max_weight*Q)/Q. max_weight=4096
    and Q=2**22 make ALL integer session counts representable exactly.
    Include stochastic one-grid-cell error, conservative float32 arithmetic
    allowance before rounding and final float32-cast allowance.
    """
    p=scfg()['secagg'];q=p['quantization_range'];c=p['clipping_range'];mw=p['max_weight']
    if any(type(w)!=int or not 0<w<=mw for w in weights):raise ValueError('Invalid public aggregation weights')
    ratios=[round(w/mw*q)/q for w in weights]
    if any(w/mw!=r for w,r in zip(weights,ratios)):raise ValueError('Inexact weight ratio requires a separately derived bound')
    eps=np.finfo(np.float32).eps
    return len(weights)*(2*c/q+4*eps*c)/sum(ratios)+2*eps*max_abs_parameter
