"""Symmetric auxiliary i-SIR retaining the pure Teacher path mapping.

pi(E|y,J) is proportional to phi(E)*w_TS(E,J;y). On an independently chosen
coordinate block let C|E=N(E,delta^2 I), and propose E'|C=N(C,delta^2 I).
These two densities are identical at (C,E), so augmented importance weights
are phi(E)*w_TS(E), not w_TS(E). Inactive coordinates stay fixed.

Unlike the OU auxiliary, this proposal does not contract a distant posterior
reference towards the origin of the Teacher innovation space. No drift gradient,
MH acceptance, bridge field, or extra neural evaluation is introduced.
"""
import torch
from .conditional_path_is import ConditionalISResult


@torch.no_grad()
def conditional_random_walk_sweep(reference, stops, evaluate, *, num_particles=8,
                                  auxiliary_scale=.003, block_size=None, generator=None,
                                  bank_policy='uniform', gaussian_mean=None, gaussian_std=1.):
    if num_particles < 2 or auxiliary_scale <= 0:
        raise ValueError('Need at least two particles and a positive scale')
    if bank_policy not in {'uniform', 'last'}:
        raise ValueError('Unknown innovation-bank policy')
    if bank_policy == 'last' and block_size is None:
        raise ValueError('Last-bank sampling requires an explicit coordinate block')
    if gaussian_std <= 0:
        raise ValueError('Gaussian reference scale must be positive')
    batch = reference.shape[0]
    center = reference + auxiliary_scale * torch.randn(reference.shape, device=reference.device,
                                                       dtype=reference.dtype, generator=generator)
    candidates = center[:, None] + auxiliary_scale * torch.randn(
        (batch, num_particles-1)+reference.shape[1:], device=reference.device,
        dtype=reference.dtype, generator=generator)
    if block_size is not None:
        dimension = reference[0, 0].numel()
        if not 1 <= block_size <= dimension:
            raise ValueError('Invalid coordinate block')
        bank = (stops.clone() if bank_policy == 'last' else
                (torch.rand(batch, device=reference.device, generator=generator)*(stops+1)).long())
        offset = torch.randint(dimension, (batch,), device=reference.device, generator=generator)
        coords = (offset[:, None]+torch.arange(block_size, device=reference.device)) % dimension
        mask = torch.zeros_like(reference, dtype=torch.bool).flatten(2)
        mask[torch.arange(batch, device=reference.device)[:, None], bank[:, None], coords] = True
        candidates = torch.where(mask.reshape_as(reference)[:, None], candidates, reference[:, None])
    particles = torch.cat([reference[:, None], candidates], dim=1)
    particle_stops = stops[:, None].expand(batch, num_particles)
    value = evaluate(particles.flatten(0, 1), particle_stops.flatten())
    base_weight = value.log_weight.reshape(batch, num_particles).double()
    # Subtract the shared phi(reference) before summing: no large full-norm
    # subtraction, no artificial dimension normalization of the true weights.
    delta = particles.double()-reference[:, None].double()
    shifted = reference.double() if gaussian_mean is None else reference.double()-gaussian_mean.double()
    prior_correction = -.5*((delta/gaussian_std).square()+
        2*(shifted[:, None]/gaussian_std)*(delta/gaussian_std)).flatten(2).sum(2)
    log_weight = base_weight+prior_correction
    weights = log_weight.softmax(1)
    sources = value.source.reshape((batch, num_particles)+value.source.shape[1:])
    broadcast = weights.reshape((batch, num_particles)+(1,)*(sources.ndim-2))
    mean = (broadcast*sources.double()).sum(1)
    chosen = torch.multinomial(weights, 1, generator=generator).flatten()
    rows = torch.arange(batch, device=reference.device)
    difference = (sources[rows, chosen].float()-sources[:, 0].float()).flatten(1)
    displacement = (mean-sources[:, 0].double()).flatten(1)
    def span(v):
        v = v.reshape(batch, num_particles)
        return float((v.max(1).values-v.min(1).values).mean())
    variance = (broadcast*(sources.double()-mean[:, None]).square()).flatten(2).sum(1).mean()
    return ConditionalISResult(particles[rows, chosen], stops.clone(), mean.float(), dict(
        path_is_ess=float(weights.square().sum(1).reciprocal().mean()),
        path_is_max_weight=float(weights.max(1).values.mean()),
        path_is_reference_probability=float(weights[:, 0].mean()),
        path_is_actual_move_rate=float((difference.square().sum(1)>0).float().mean()),
        path_is_move_rms=float(difference.square().mean(1).sqrt().mean()),
        path_is_mean_displacement_rms=float(displacement.square().mean(1).sqrt().mean()),
        path_is_log_weight_range=span(log_weight),
        path_is_auxiliary_scale=auxiliary_scale,
        path_is_block_size=int(block_size) if block_size is not None else reference[0].numel(),
        path_is_path_ratio_span=span(value.log_path_ratio),
        path_is_observation_span=span(value.log_observation),
        path_is_teacher_base_weight_span=span(base_weight),
        path_is_gaussian_reference_correction_span=span(prior_correction),
        path_is_last_innovation_only=float(bank_policy == 'last'),
        path_is_rb_removed_variance_trace_per_coordinate=float(variance)))
