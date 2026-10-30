"""Endpoint-space Teacher proposal for the Version-C shared-noise bridge.

This is NOT a Teacher Markov rollout. It defines a Teacher-centred endpoint
proposal q_ref(x0|xt,c), samples endpoints from the density mixture
q_kappa = kappa*q_student + (1-kappa)*q_ref, and importance-corrects back to
the Student target q_student. The shared-noise bridge is then constructed
algebraically from each sampled endpoint and the fixed query.

Assumptions:
- batch size is one (matching the supplied paired training path);
- q_student is the fitted K-component diagonal Gaussian mixture;
- q_ref is a full-support diagonal Gaussian centred at Teacher clean readout;
- q_ref variance defaults to the Student mixture's marginal variance. This is
  a proposal heuristic, NOT a calibrated Teacher posterior. Replace it with
  held-out Teacher endpoint residual calibration before treating it as final.
"""
import math
import torch


def _normal_log_prob(x, mean, variance):
    """Diagonal Gaussian log density; x/mean/variance broadcast to [N,C,H,W]."""
    variance = variance.clamp_min(1e-8)
    return -0.5 * (
        math.log(2.0 * math.pi) + variance.log()
        + (x - mean).square() / variance
    ).flatten(1).sum(1,dtype=torch.float64)


def _student_mixture_log_prob(x, log_pi, mu, variance):
    """Full K-component Student target density, shape [N]."""
    n = x.shape[0]
    lp = log_pi.expand(n, -1)
    mm = mu.expand(n, -1, -1, -1, -1)
    vv = variance.expand_as(mm).clamp_min(1e-8)
    xx = x[:, None]
    component = -0.5 * (
        math.log(2.0 * math.pi) + vv.log()
        + (xx - mm).square() / vv
    ).flatten(2).sum(2,dtype=torch.float64)
    return torch.logsumexp(lp + component, dim=1)


@torch.no_grad()
def teacher_endpoint_proposal(
    query,
    sigma,
    log_pi,
    mu,
    variance,
    teacher_clean,
    kappa=0.5,
    particles=32,
    teacher_variance=None,
    generator=None,
):
    """Sample q_kappa endpoints and return Student-target IS estimate.

    q_student(x0|query) = sum_k pi_k N(mu_k, diag(var_k)).
    q_ref(x0|query) = N(teacher_clean, diag(ref_var)).
    q_kappa = kappa*q_student + (1-kappa)*q_ref.

    Importance weight: q_student(x0_i|query) / q_kappa(x0_i|query).
    This is endpoint-space IS. No path-space Radon-Nikodym ratio is used.
    """
    if query.shape[0] != 1:
        raise ValueError("teacher_endpoint_proposal currently requires batch size 1")
    if not (0.0 <= float(kappa) <= 1.0):
        raise ValueError("kappa must be in [0,1]")
    if particles < 1:
        raise ValueError("particles must be positive")

    device, dtype = query.device, query.dtype
    target_mean = (log_pi.exp()[:, :, None, None, None] * mu).sum(1)

    # Marginal variance of the Student mixture:
    # E[var_k + (mu_k - E[mu])^2]. Used only as a provisional proposal scale.
    pi = log_pi.exp()[:, :, None, None, None]
    ref_var = (pi * (variance + (mu - target_mean[:, None]).square())).sum(1)
    if teacher_variance is not None:
        ref_var = torch.as_tensor(teacher_variance, device=device, dtype=mu.dtype)
        if ref_var.ndim == 1:
            ref_var = ref_var.view(1, -1, 1, 1)
        elif ref_var.ndim == 3:
            ref_var = ref_var.unsqueeze(0)
        ref_var = ref_var.expand_as(teacher_clean)
    ref_var = ref_var.clamp_min(1e-6)

    # Select the source component of the proposal mixture for each particle.
    use_student = torch.rand((particles,), device=device, generator=generator) < float(kappa)
    endpoints = torch.empty(
        (particles, *teacher_clean.shape[1:]),
        device=device, dtype=mu.dtype
    )

    n_student = int(use_student.sum().item())
    n_teacher = particles - n_student

    if n_student:
        component = torch.multinomial(
            log_pi[0].exp(), n_student, replacement=True, generator=generator
        )
        noise = torch.randn(
            (n_student, *mu.shape[2:]), device=device, dtype=mu.dtype,
            generator=generator
        )
        endpoints[use_student] = (
            mu[0, component] + variance[0, component].clamp_min(1e-8).sqrt() * noise
        )

    if n_teacher:
        noise = torch.randn(
            (n_teacher, *teacher_clean.shape[1:]), device=device, dtype=mu.dtype,
            generator=generator
        )
        endpoints[~use_student] = (
            teacher_clean[0].to(mu.dtype)
            + ref_var[0].sqrt() * noise
        )

    log_target = _student_mixture_log_prob(endpoints, log_pi, mu, variance)
    log_ref = _normal_log_prob(
        endpoints,
        teacher_clean.expand_as(endpoints).to(mu.dtype),
        ref_var.expand_as(teacher_clean).expand_as(endpoints),
    )

    if kappa == 1.0:
        log_proposal = log_target
    elif kappa == 0.0:
        log_proposal = log_ref
    else:
        log_proposal = torch.logaddexp(
            math.log(float(kappa)) + log_target,
            math.log1p(-float(kappa)) + log_ref,
        )

    log_weight = log_target - log_proposal
    weights = torch.softmax(log_weight, dim=0)
    estimate = (weights[:, None, None, None] * endpoints).sum(0, keepdim=True)

    # Reconstruct the common noise and verify the shared-noise bridge identity.
    s = float(sigma)
    if not (0.0 < s <= 1.0):
        raise ValueError("sigma must be positive for endpoint-to-noise inversion")
    eps = (query.float() - (1.0 - s) * endpoints.float()) / s
    bridge_errors = []
    for ratio in (1.0, 0.9, 0.75, 0.5, 0.0):
        r = ratio * s
        path_a = (1.0 - r) * endpoints.float() + r * eps
        path_b = (r / s) * query.float() + (1.0 - r / s) * endpoints.float()
        bridge_errors.append((path_a - path_b).abs().max())
    bridge_error = torch.stack(bridge_errors).max()

    if not torch.isfinite(estimate).all() or not torch.isfinite(log_weight).all():
        raise FloatingPointError("non-finite Teacher-proposal endpoint estimate")
    if bridge_error > 2e-4:
        raise AssertionError(f"shared-noise bridge identity failed: {bridge_error.item()}")

    ess = 1.0 / weights.square().sum()
    diagnostics = {
        "teacher_proposal/kappa": float(kappa),
        "teacher_proposal/teacher_draw_fraction": float((~use_student).float().mean()),
        "teacher_proposal/teacher_weight_mass": float(weights[~use_student].sum()),
        "teacher_proposal/student_draw_count": n_student,
        "teacher_proposal/teacher_draw_count": n_teacher,
        "teacher_proposal/unnormalized_mass_estimate": float(log_weight.exp().mean()),
        "teacher_proposal/target_teacher_rms": float((target_mean-teacher_clean).square().mean().sqrt()),
        "teacher_proposal/normalized_mean_error": float((estimate-target_mean).square().mean().sqrt()/ref_var.mean().sqrt()),

        "teacher_proposal/ess": float(ess),
        "teacher_proposal/ess_fraction": float(ess / particles),
        "teacher_proposal/max_weight": float(weights.max()),
        "teacher_proposal/log_weight_std": float(log_weight.std(unbiased=False)),
        "teacher_proposal/bridge_identity_error": float(bridge_error),
        "teacher_proposal/weighted_endpoint_rms": float(estimate.square().mean().sqrt()),
        "teacher_proposal/student_target_mean_rms": float(target_mean.square().mean().sqrt()),
        "teacher_proposal/mc_vs_target_mean_rms": float((estimate - target_mean).square().mean().sqrt()),
        "teacher_proposal/ref_variance_mean": float(ref_var.mean()),
    }
    return estimate.to(query.dtype), diagnostics
