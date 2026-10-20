"""Research prototype: literal Student/Teacher CPS ratios on positive-noise paths.

Teacher paths share a Student terminal readout. The observation likelihood and
initial/stop proposal corrections target the actual forward-corruption posterior.
Annealing controls incremental weights; mutation is only an SMC rejuvenation
step, not a replacement for the path likelihood ratio. Not an SD3 trainer.
"""
from dataclasses import dataclass
import math
import torch


class TemperingBudgetExceeded(RuntimeError):
    """The final posterior was NOT reached; callers must not report a mean."""


@dataclass
class PathEvaluation:
    source: torch.Tensor
    log_weight: torch.Tensor
    log_path_ratio: torch.Tensor
    log_initial_ratio: torch.Tensor
    log_stop_ratio: torch.Tensor
    log_observation: torch.Tensor


@dataclass
class SMCResult:
    mean: torch.Tensor
    particles: torch.Tensor
    stops: torch.Tensor
    log_weights: torch.Tensor
    history: list
    evaluations: int
    distinct_initial_ancestors: int
    distinct_final_sources: int
    beta: float


def cps_mean(sample, velocity, sigma_from, sigma_to, eta):
    angle = eta * math.pi / 2
    return ((1 - sigma_to) * (sample - sigma_from * velocity)
            + sigma_to * math.cos(angle) * (sample + (1 - sigma_from) * velocity))


def gaussian_log_ratio_at(value, target_mean, proposal_mean, std):
    """Common positive scalar covariance; sum every event dimension."""
    if not math.isfinite(float(std)) or std <= 0:
        raise ValueError('A deterministic kernel has no ordinary Gaussian density ratio')
    a = ((value.double() - target_mean.double()) / std).square()
    b = ((value.double() - proposal_mean.double()) / std).square()
    return (-.5 * (a - b)).flatten(1).sum(1)


def bf16_gaussian_log_mass(value, mean, std):
    """Gaussian mass in the BF16 round-to-nearest cell, not density at value.

    Ideal Gaussian innovations are assumed; finite-RNG/multiply rounding remains
    a numerical approximation. Tiny cells use a midpoint evaluation with
    standardized width*max(1,|center|)<1e-4 (relative error of order 1e-8).
    """
    if value.dtype != torch.bfloat16 or std <= 0 or not torch.isfinite(value).all():
        raise ValueError('Need finite BF16 states and positive Gaussian standard deviation')
    previous=torch.nextafter(value,torch.full_like(value,-float('inf'))).double()
    following=torch.nextafter(value,torch.full_like(value,float('inf'))).double()
    midpoint=value.double()
    lower=.5*(previous+midpoint)
    upper=.5*(following+midpoint)
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError('Extreme overflow BF16 cells are outside this prototype')
    width=(upper-lower)/std
    center=(.5*(lower+upper)-mean.double())/std
    lo=(lower-mean.double())/std
    hi=(upper-mean.double())/std
    tiny=width*torch.maximum(torch.ones_like(center),center.abs())<1e-4
    # Right tails use survival probabilities to avoid subtracting values near 1.
    large=torch.where(lo>=0,torch.special.log_ndtr(-lo),torch.special.log_ndtr(hi))
    small=torch.where(lo>=0,torch.special.log_ndtr(-hi),torch.special.log_ndtr(lo))
    delta=(small-large).clamp(max=0)
    cdf_difference=large+torch.log(-torch.expm1(delta))
    midpoint_mass=width.log()-.5*center.square()-.5*math.log(2*math.pi)
    return torch.where(tiny,midpoint_mass,cdf_difference)


def bf16_gaussian_log_ratio_at(value,target_mean,proposal_mean,std):
    difference=(bf16_gaussian_log_mass(value,target_mean,std)
                -bf16_gaussian_log_mass(value,proposal_mean,std))
    return difference.flatten(1).sum(1)


@torch.no_grad()
def evaluate_teacher_path(innovations, stops, observation, sigma_observation,
                          positive_sigmas, eta, student_velocity, teacher_velocity,
                          initial_mean=None, target_stop_probs=None, proposal_stop_probs=None,
                          state_dtype=None):
    """Map iid Gaussian innovations through a *Teacher* prefix and weigh it.

    innovations: [N,K,...], with K strictly positive generator sigma nodes.
    stops: [N], zero-based readout nodes, NOT a fixed known observation label.
    Unused teacher suffixes are auxiliary variables with identical laws in the
    extended target and proposal; their ratios cancel. Thus random stopping is
    marginalized correctly and only prefix ratios enter each particle's weight.
    """
    grid = tuple(float(s) for s in positive_sigmas)
    n, count = innovations.shape[:2]
    if (count != len(grid) or count < 1 or not 0 < eta <= 1
            or any(not 0 < s <= 1 for s in grid)
            or any(a <= b for a, b in zip(grid, grid[1:]))):
        raise ValueError('Need strictly decreasing positive sigma nodes and 0<eta<=1')
    if stops.shape != (n,) or torch.any((stops < 0) | (stops >= count)):
        raise ValueError('Invalid zero-based stopping nodes')
    if not torch.isfinite(innovations).all():
        raise ValueError('Nonfinite innovations')
    sample = innovations[:, 0]
    shift = torch.zeros_like(sample) if initial_mean is None else torch.broadcast_to(initial_mean, sample.shape)
    sample = sample + shift
    quantized=state_dtype==torch.bfloat16
    if state_dtype is not None:
        sample=sample.to(state_dtype)
    initial_ratio = (bf16_gaussian_log_ratio_at(sample,torch.zeros_like(shift),shift,1.) if quantized
        else (.5 * shift.double().square() - sample.double() * shift.double()).flatten(1).sum(1))
    cumulative = torch.zeros(n, device=sample.device, dtype=torch.float64)
    reads, prefixes = [], []
    for k, s in enumerate(grid):
        student = student_velocity(sample, s)
        x=sample.float() if quantized else sample
        student=student.to(x.dtype)
        reads.append((x - s * student).to(sample.dtype))
        prefixes.append(cumulative)
        if k == int(stops.max().item()):
            break  # Shared Student clean readout: NO terminal Gaussian ratio.
        t = grid[k + 1]
        teacher = teacher_velocity(sample, s).to(x.dtype)
        target_mean = cps_mean(x, student, s, t, eta)
        proposal_mean = cps_mean(x, teacher, s, t, eta)
        std = t * math.sin(eta * math.pi / 2)
        proposed = (proposal_mean + std * innovations[:, k + 1]).to(sample.dtype)
        ratio_function=bf16_gaussian_log_ratio_at if quantized else gaussian_log_ratio_at
        cumulative = cumulative + ratio_function(proposed, target_mean, proposal_mean, std)
        sample = proposed
    rows = torch.arange(n, device=sample.device)
    source = torch.stack(reads, dim=1)[rows, stops]
    path_ratio = torch.stack(prefixes, dim=1)[rows, stops]
    def probabilities(values):
        values = (torch.ones(count, device=sample.device, dtype=torch.float64)
                  if values is None else torch.as_tensor(values, device=sample.device, dtype=torch.float64))
        if values.shape != (count,) or not torch.isfinite(values).all() or torch.any(values <= 0):
            raise ValueError('Every stop must have positive finite proposal/target mass')
        return values / values.sum()
    p, q = probabilities(target_stop_probs), probabilities(proposal_stop_probs)
    stop_ratio = (p.log() - q.log())[stops]
    sigma = torch.as_tensor(sigma_observation, device=sample.device, dtype=torch.float64)
    if sigma.numel() == 1:
        sigma = sigma.expand(n)
    if sigma.shape != (n,) or torch.any((sigma <= 0) | (sigma >= 1)):
        raise ValueError('Observation sigma must be in (0,1)')
    sigma = sigma.reshape((-1,) + (1,) * (source.ndim - 1))
    residual = (observation.double() - (1 - sigma) * source.double()) / sigma
    likelihood = -.5 * residual.square().flatten(1).sum(1)
    log_weight = initial_ratio + path_ratio + stop_ratio + likelihood
    if not torch.isfinite(log_weight).all():
        raise FloatingPointError('Nonfinite path-posterior weight')
    return PathEvaluation(source, log_weight, path_ratio, initial_ratio, stop_ratio, likelihood)


def normalized_log_weights(log_weights):
    return log_weights.double() - torch.logsumexp(log_weights.double(), dim=0)


def effective_sample_size(log_weights):
    log_weights = normalized_log_weights(log_weights)
    return float((-torch.logsumexp(2 * log_weights, dim=0)).exp())


def next_temperature(beta, log_weights, log_ratio, ess_fraction):
    """Largest temperature with predicted PRE-resampling ESS >= threshold."""
    n = log_ratio.numel()
    threshold = n * ess_fraction
    def ess(t):
        return effective_sample_size(log_weights + (t - beta) * log_ratio)
    if ess(1.) >= threshold:
        return 1.
    if ess(beta) < threshold - 1e-7:
        raise ValueError('Resample before requesting another ESS-constrained bridge')
    lo, hi = beta, 1.
    for _ in range(45):
        mid = .5 * (lo + hi)
        if ess(mid) >= threshold:
            lo = mid
        else:
            hi = mid
    if lo - beta < 1e-12:
        raise TemperingBudgetExceeded(f'Temperature stalled at beta={beta}; target not reached')
    return lo


def systematic_resample(log_weights, generator=None):
    w = normalized_log_weights(log_weights).exp()
    n = len(w)
    offset = torch.rand((), generator=generator, device=w.device, dtype=torch.float64)
    positions = (torch.arange(n, device=w.device, dtype=torch.float64) + offset) / n
    cumulative = w.cumsum(0)
    cumulative[-1] = 1.
    return torch.searchsorted(cumulative, positions).clamp_max(n - 1)


@torch.no_grad()
def annealed_path_smc(initial_innovations, initial_stops, evaluate, *,
                      proposal_stop_probs=None, ess_fraction=.8, max_stages=128,
                      mutation_steps=16, pcn_scale=.15, generator=None):
    """Targets q(E,J) exp(beta*log_w(E,J)); beta MUST reach one.

    q(E,J)=N(E;0,I)*q_J(J). q_J must match the initial stop draws AND evaluator.
    Mutation is q-reversible pCN / a separate prior stop refresh, so acceptance
    is beta*(log_w_new-log_w_old). No finite-N unbiased posterior-mean claim.
    """
    n, count = initial_innovations.shape[:2]
    if n < 2 or not 0 < ess_fraction < 1 or not 0 < pcn_scale < 1:
        raise ValueError('Invalid particle count, ESS fraction or pCN scale')
    if max_stages < 1 or mutation_steps < 0:
        raise ValueError('Invalid stage/mutation budget')
    particles, stops = initial_innovations.clone(), initial_stops.clone()
    qj = (torch.ones(count, device=particles.device, dtype=torch.float64)
          if proposal_stop_probs is None else torch.as_tensor(proposal_stop_probs, device=particles.device, dtype=torch.float64))
    if qj.shape != (count,) or not torch.isfinite(qj).all() or torch.any(qj <= 0):
        raise ValueError('Invalid stop proposal')
    qj = qj / qj.sum()
    current = evaluate(particles, stops)
    evaluations, beta = n, 0.
    weights = torch.full((n,), -math.log(n), device=particles.device, dtype=torch.float64)
    ancestors = torch.arange(n, device=particles.device)
    history = []
    for stage in range(max_stages):
        new_beta = next_temperature(beta, weights, current.log_weight, ess_fraction)
        weights = normalized_log_weights(weights + (new_beta - beta) * current.log_weight)
        before = effective_sample_size(weights)
        distinct_before = int(torch.unique(current.source.flatten(1), dim=0).shape[0])
        # No final resampling: final weights retain meaningful diagnostics.
        if new_beta < 1.:
            indices = systematic_resample(weights, generator)
            particles, stops, ancestors = particles[indices], stops[indices], ancestors[indices]
            current = PathEvaluation(*(getattr(current, field)[indices] for field in current.__dataclass_fields__))
            weights.fill_(-math.log(n))
        accepted, actual_moves = 0, 0
        for move in range(mutation_steps):
            if count > 1 and move % 4 == 3:
                candidate = particles
                proposed_stops = torch.multinomial(qj, n, replacement=True, generator=generator)
            else:
                noise = torch.randn(particles.shape, generator=generator, device=particles.device, dtype=particles.dtype)
                candidate = math.sqrt(1 - pcn_scale**2) * particles + pcn_scale * noise
                proposed_stops = stops
            proposed = evaluate(candidate, proposed_stops)
            evaluations += n
            log_acceptance = (new_beta * (proposed.log_weight - current.log_weight)).clamp(max=0)
            log_uniform = torch.rand(n, device=particles.device, generator=generator, dtype=torch.float64).log()
            take = log_uniform < log_acceptance
            changed = (proposed.source != current.source).flatten(1).any(1)
            accepted += int(take.sum())
            actual_moves += int((take & changed).sum())
            mask = take.reshape((-1,) + (1,) * (particles.ndim - 1))
            particles = torch.where(mask, candidate, particles)
            stops = torch.where(take, proposed_stops, stops)
            values = []
            for field in current.__dataclass_fields__:
                old, new = getattr(current, field), getattr(proposed, field)
                field_mask = take.reshape((-1,) + (1,) * (old.ndim - 1))
                values.append(torch.where(field_mask, new, old))
            current = PathEvaluation(*values)
        history.append(dict(stage=stage+1, beta=new_beta, ess_before_resampling=before,
            resampled=new_beta < 1., ess_after_resampling=(float(n) if new_beta < 1. else before),
            distinct_sources_before_resampling=distinct_before,
            distinct_initial_ancestors=int(torch.unique(ancestors).numel()),
            mutation_acceptance=(accepted / (n * mutation_steps) if mutation_steps else 0.),
            mutation_actual_move_rate=(actual_moves / (n * mutation_steps) if mutation_steps else 0.)))
        beta = new_beta
        if beta == 1.:
            mean = (weights.exp().reshape((-1,) + (1,) * (current.source.ndim - 1)) * current.source.double()).sum(0)
            return SMCResult(mean, particles, stops, weights, history, evaluations,
                int(torch.unique(ancestors).numel()), int(torch.unique(current.source.flatten(1), dim=0).shape[0]), beta)
    error=TemperingBudgetExceeded(
        f'Budget {max_stages} exhausted at beta={beta:.8f}; final posterior NOT reached')
    error.history=history
    error.evaluations=evaluations
    raise error
