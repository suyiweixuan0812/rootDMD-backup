"""Corrected Scheme A: importance sampling on the generator's source space.

No reverse-kernel or deterministic-endpoint density is evaluated. Candidate
sources must be independent of the differentiable source, conditional on y,c,t.
"""
import torch


def gaussian_log_ratio(z, mean):
    """log N(z;0,I) / N(z;mean,I), summing ALL event coordinates."""
    return (0.5 * mean.double().square() - z.double() * mean.double()).flatten(1).sum(1)


def log_likelihood(y, source, sigma):
    sigma = sigma.double().reshape((-1,) + (1,) * (source.ndim - 1))
    if torch.any((sigma <= 0) | (sigma >= 1)):
        raise ValueError("The Gaussian corruption requires 0 < sigma < 1")
    residual = (y.double() - (1 - sigma) * source.double()) / sigma
    return -0.5 * residual.square().flatten(1).sum(1)


@torch.no_grad()
def posterior_mean(y, sigma, proposal_mean, draw_source, num_particles, generator=None):
    """draw_source(z) samples the SAME stop/CPS-noise law as the source generator.

    Only the initial Gaussian noise has a shifted proposal; all remaining
    generator randomness retains its prior law and cancels from p/q.
    """
    if num_particles < 2:
        raise ValueError("Use at least two particles to expose weight degeneracy")
    sources, logs = [], []
    for _ in range(num_particles):
        z = torch.randn(y.shape, device=y.device, dtype=torch.float32, generator=generator)
        z = z + proposal_mean.float()
        source = draw_source(z)
        sources.append(source.detach())
        logs.append(log_likelihood(y, source, sigma) + gaussian_log_ratio(z, proposal_mean))
    logs = torch.stack(logs)
    if not torch.isfinite(logs).all():
        raise FloatingPointError("Nonfinite particle log weights")
    weights = logs.softmax(dim=0)
    mean = sum(w.reshape((-1,) + (1,) * (y.ndim - 1)) * x.double()
               for w, x in zip(weights, sources)).float()
    ess = weights.square().sum(0).reciprocal()
    entropy = -(weights * weights.clamp_min(1e-300).log()).sum(0)
    return mean, {
        "posterior/ess": ess.mean().item(),
        "posterior/ess_fraction": (ess / num_particles).mean().item(),
        "posterior/max_weight": weights.max(0).values.mean().item(),
        "posterior/entropy": entropy.mean().item(),
        "posterior/log_weight_range": (logs.max(0).values - logs.min(0).values).mean().item(),
    }
