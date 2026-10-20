"""Equivalent last-node Gaussian coordinates avoiding near-zero inverse loss.

Sample the unrounded physical U in the BF16 cell directly. Standardizing a cell
near zero by subtracting a nonzero mean can collapse its endpoints in FP64.
An affine coordinate change U=mu_T+nu*E preserves the same local random-walk
conditional i-SIR kernel and Gaussian prior correction in exact arithmetic.
"""
import torch
from .path_ratio_smc import PathEvaluation, bf16_gaussian_log_ratio_at


def round_bf16_exact(value):
    """Round FP64 directly to nearest BF16, correcting FP32 double rounding.

    Torch's CUDA cast can first round to FP32; a value just inside an odd BF16
    cell then lands on the midpoint and rounds to its even neighbor. Compare
    exact FP64 distances to the three local BF16 values to restore the stated
    single-rounding Gaussian-cell model. The preliminary cast resolves exact
    midpoint ties to even; strict comparisons preserve that choice.
    """
    x=value.double()
    out=x.to(torch.bfloat16)
    if not torch.isfinite(x).all() or not torch.isfinite(out).all():
        raise ValueError('Nonfinite/overflow physical Gaussian state')
    lower=torch.nextafter(out,torch.full_like(out,-float('inf')))
    upper=torch.nextafter(out,torch.full_like(out,float('inf')))
    distance=(x-out.double()).abs()
    dl,du=(x-lower.double()).abs(),(x-upper.double()).abs()
    out=torch.where(dl<distance,lower,out)
    distance=torch.minimum(distance,dl)
    return torch.where(du<distance,upper,out)


@torch.no_grad()
def lift_physical_gaussian_cell(value, mean, std, generator=None, max_rounds=128):
    if value.dtype != torch.bfloat16 or std <= 0 or not torch.isfinite(value).all():
        raise ValueError('Need finite BF16 values and positive Gaussian scale')
    lower = .5*(value.double()+torch.nextafter(value, torch.full_like(value, -float('inf'))).double())
    upper = .5*(value.double()+torch.nextafter(value, torch.full_like(value, float('inf'))).double())
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or torch.any(lower >= upper):
        raise ValueError('Unsupported physical rounding cell')
    mean = torch.broadcast_to(mean.double(), value.shape)
    mode = torch.maximum(lower, torch.minimum(mean, upper))
    out = torch.empty_like(lower)
    pending = torch.ones_like(lower, dtype=torch.bool)
    for _ in range(max_rounds):
        n = int(pending.sum())
        if not n:
            assert torch.equal(round_bf16_exact(out), value)
            return out
        a, b, m, mu = lower[pending], upper[pending], mode[pending], mean[pending]
        proposal = a+(b-a)*torch.rand((n,), device=a.device, dtype=a.dtype, generator=generator)
        # Difference of squares in factored physical coordinates retains tiny
        # cell widths even when mu is O(1) and value is exactly zero.
        log_accept = -.5*((proposal-m)/std)*((proposal+m-2*mu)/std)
        take = (torch.rand((n,), device=a.device, dtype=a.dtype, generator=generator).log() < log_accept)
        take &= round_bf16_exact(proposal)==value[pending]
        values = out[pending]
        values[take] = proposal[take]
        out[pending] = values
        mask = pending.clone()
        mask[pending] = ~take
        pending = mask
    raise RuntimeError('Physical truncated Gaussian sampling budget exhausted')


@torch.no_grad()
def evaluate_last_physical_state(unrounded, observation, sigma, target_mean,
                                 teacher_mean, std, readout, prefix_log_ratio=0.):
    x = round_bf16_exact(unrounded[:,0])
    source = readout(x)
    ratio = bf16_gaussian_log_ratio_at(x, target_mean, teacher_mean, std)+prefix_log_ratio
    s = torch.as_tensor(sigma, device=x.device, dtype=torch.float64)
    likelihood = -.5*((observation.double()-(1-s)*source.double())/s).square().flatten(1).sum(1)
    zero = torch.zeros_like(ratio)
    return PathEvaluation(source, ratio+likelihood, ratio, zero, zero, likelihood)
