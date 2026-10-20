"""Teacher-guided Gaussian-reference MH on the actual Student source trace.

The kernel is posterior invariant. Paired joint initialization is stationary
conditional on y; this does NOT make the resulting chain samples independent.
"""
from dataclasses import dataclass
import math
import torch
from .source_posterior import log_likelihood


@dataclass(frozen=True)
class SourceState:
    z: torch.Tensor             # [batch, channels, height, width], Gaussian base noise
    noises: torch.Tensor        # [num_steps-1, batch, ...], all CPS innovations
    stop: torch.Tensor          # [batch], including unused innovations makes dimension fixed


def proposal_log_correction(z, proposed_z, teacher_mean):
    # q is reversible for N(teacher_mean,I), whereas the target prior is N(0,I).
    return -(teacher_mean.double() * (proposed_z.double() - z.double())).flatten(1).sum(1)


def propose(state, beta, teacher_mean, max_stop, random_stop=True, generator=None,
            block_size=16, refresh_stop=False):
    if refresh_stop:
        stop = (torch.randint(1, max_stop + 1, state.stop.shape,
                    device=state.z.device, generator=generator)
                if random_stop else state.stop)
        return SourceState(state.z, state.noises, stop)
    # A random-scan update conditional on the unchanged stop. Selecting only
    # active innovation banks is reversible: the reverse scan has the same J.
    batch, dimension = state.z.shape[0], state.z[0].numel()
    width = min(block_size, dimension)
    bank = (torch.rand((batch,), device=state.z.device, generator=generator)
            * state.stop.float()).long()
    offset = torch.randint(0, dimension, (batch,), device=state.z.device, generator=generator)
    coordinates = (offset[:, None] + torch.arange(width, device=state.z.device)) % dimension
    combined = torch.cat([state.z.unsqueeze(0), state.noises], dim=0)
    flat = combined.reshape(combined.shape[0], batch, dimension)
    rows = torch.arange(batch, device=state.z.device)[:, None]
    selected = flat[bank[:, None], rows, coordinates]
    center = torch.where(bank[:, None] == 0,
        teacher_mean.reshape(batch, dimension).gather(1, coordinates), torch.zeros_like(selected))
    rho = (1 - beta.square()).sqrt()[:, None]
    flat[bank[:, None], rows, coordinates] = (rho * selected + (1 - rho) * center
        + beta[:, None] * torch.randn(selected.shape, device=state.z.device,
                                      generator=generator, dtype=torch.float32))
    return SourceState(combined[0], combined[1:], state.stop)



@torch.no_grad()
def posterior_chain(y, sigma, initial_state, initial_source, teacher_mean,
                    evaluate_source, num_moves=8, random_stop=True,
                    proposal_scale=0.25, scales=(0.5, 1.0, 2.0), block_size=16, generator=None):
    """Use paired (U,y)~P(U)L(y|F(U)) initialization, with theta fixed.

    Average conditional expectations of MH next states (Rao-Blackwellization).
    For stationary initialization each term has mean E[F(U)|y]. For arbitrary
    initialization this finite chain need not be an accurate posterior estimate.
    """
    if num_moves < 1 or not scales or proposal_scale <= 0:
        raise ValueError('Need positive moves, scales and proposal scale')
    sigma = sigma.flatten().float()
    if torch.any((sigma <= 0) | (sigma >= 1)):
        raise ValueError('Require 0 < sigma < 1')
    if not torch.isfinite(teacher_mean).all():
        raise ValueError('Nonfinite teacher proposal center')
    state, source = initial_state, initial_source.detach().float()
    original = source.clone()
    current_ll = log_likelihood(y, source, sigma)
    dimension = source[0].numel()
    base_beta = proposal_scale * sigma / ((1 - sigma) * math.sqrt(min(block_size, dimension)))
    estimate = torch.zeros_like(source, dtype=torch.float64)
    accepted, probabilities, betas, ratios, moved = [], [], [], [], []
    stop_changes = []
    # Short-chain correlation diagnostics, explicitly not an ESS estimate.
    def projection(x):
        flat = x.flatten(1)
        return flat[:, ::2].mean(1) - flat[:, 1::2].mean(1) if flat.shape[1] > 1 else flat[:, 0]
    projections = [projection(source)]
    for index in range(num_moves):
        beta = (base_beta * scales[index % len(scales)]).clamp(1e-4, 0.5)
        candidate = propose(state, beta, teacher_mean, state.noises.shape[0] + 1,
                            random_stop, generator, block_size,
                            refresh_stop=(random_stop and index == num_moves - 1))
        candidate_source = evaluate_source(candidate).detach().float()
        candidate_ll = log_likelihood(y, candidate_source, sigma)
        log_alpha = candidate_ll - current_ll + proposal_log_correction(state.z, candidate.z, teacher_mean)
        if not torch.isfinite(log_alpha).all():
            raise FloatingPointError('Nonfinite MH acceptance ratio')
        prob = log_alpha.clamp(max=0).exp()
        expand = (-1,) + (1,) * (source.ndim - 1)
        p = prob.reshape(expand)
        estimate += p * candidate_source.double() + (1 - p) * source.double()
        take = torch.rand(prob.shape, device=y.device, generator=generator).double() < prob
        mask = take.reshape(expand)
        before_stop = state.stop
        state = SourceState(torch.where(mask, candidate.z, state.z),
                            torch.where(mask.unsqueeze(0), candidate.noises, state.noises),
                            torch.where(take, candidate.stop, state.stop))
        delta = torch.where(mask, candidate_source - source, torch.zeros_like(source))
        source = torch.where(mask, candidate_source, source)
        current_ll = torch.where(take, candidate_ll, current_ll)
        accepted.append(take.float())
        probabilities.append(prob)
        betas.append(beta)
        ratios.append(log_alpha)
        moved.append(delta.square().flatten(1).mean(1).sqrt())
        stop_changes.append((take & (before_stop != state.stop)).float())
        projections.append(projection(source))
    mean = (estimate / num_moves).float()
    projected = torch.stack(projections).double()
    left, right = projected[:-1], projected[1:]
    left, right = left - left.mean(0), right - right.mean(0)
    denominator = (left.square().sum(0) * right.square().sum(0)).sqrt()
    correlation = torch.where(denominator > 1e-16,
                              (left * right).sum(0) / denominator.clamp_min(1e-16),
                              torch.ones_like(denominator))
    actual_accepts = torch.stack(accepted)
    return mean, {
        'mh/accept_rate': actual_accepts.mean().item(),
        'mh/mean_accept_probability': torch.stack(probabilities).mean().item(),
        'mh/mean_beta': torch.stack(betas).mean().item(),
        'mh/mean_log_accept_ratio': torch.stack(ratios).mean().item(),
        'mh/min_log_accept_ratio': torch.stack(ratios).min().item(),
        'mh/stop_change_rate': torch.stack(stop_changes).mean().item(),
        'mh/move_rms': torch.stack(moved).mean().item(),
        'mh/source_move_rate': (torch.stack(moved) > 0).float().mean().item(),
        'mh/final_source_displacement_rms': (source - original).square().mean().sqrt().item(),
        'mh/mean_source_displacement_rms': (mean - original).square().mean().sqrt().item(),
        'mh/short_chain_projection_lag1': correlation.mean().item(),
        'mh/accepted_transitions': actual_accepts.sum(0).mean().item(),
        'mh/all_rejected': (actual_accepts.sum(0) == 0).float().mean().item(),
        'mh/sigma': sigma.mean().item(),
        'mh/source_dim': dimension,
        'mh/block_size': min(block_size, dimension),
    }
