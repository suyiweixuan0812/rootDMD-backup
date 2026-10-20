"""Deterministic query preconditioning; changes surrogate weighting, not posterior.

D_theta(y,sigma)=y-sigma*v_theta(y,sigma), so multiplying a clean-output
surrogate by k/sigma removes its explicit sigma Jacobian factor. k=.30 was
calibrated using preclip training gradients, never validation/test scores.
This is a practical time-weight ablation, not an exact DMD/KL equivalence.
"""
import math


def query_time_weight(sigma,mode='none',scale=.30):
    sigma,scale=float(sigma),float(scale)
    if not math.isfinite(sigma) or not 0<sigma<1:
        raise ValueError('sigma must be in (0,1)')
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('scale must be finite and positive')
    if mode=='none':return 1.
    if mode=='inverse_sigma':return scale/sigma
    raise ValueError('Unknown deterministic query time weight: '+mode)
