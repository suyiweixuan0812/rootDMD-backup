"""Pure Teacher paths with symmetric auxiliary i-SIR and explicit phi weights."""
import math
import torch
from flow_grpo.path_ratio_smc import evaluate_teacher_path
from flow_grpo.conditional_path_is import lift_student_prefix
from flow_grpo.random_walk_path_is import conditional_random_walk_sweep
from .sd3_conditional_path_is import make_fields, sample_prefix


@torch.no_grad()
def conditional_path_mean(observation, sigma, states, source, student, teacher, positive_sigmas,
                          eta, num_particles=8, num_sweeps=2, base_scale=.05, block_size=16,
                          high_noise_block_size=4096, high_noise_threshold=.9,
                          max_innovation_radius=.5, bank_policy='uniform'):
    if source.shape[0] != 1:
        raise ValueError('Random-walk path comparison requires batch1')
    grid = [float(s) for s in positive_sigmas]
    reference = lift_student_prefix(states, grid, eta, teacher)
    stops = torch.tensor([len(states)-1], device=source.device)
    def evaluate(e, j):
        return evaluate_teacher_path(e, j, observation, sigma.flatten(), grid, eta,
                                     student, teacher, state_dtype=source.dtype)
    if not torch.equal(evaluate(reference, stops).source, source):
        raise RuntimeError('Pure-Teacher random-walk reference replay mismatch')
    s = float(sigma.item())
    active = high_noise_block_size if s >= high_noise_threshold else block_size
    if not 1 <= active <= source[0].numel():
        raise ValueError('Invalid coordinate block')
    radius = min(base_scale*s/(1-s), max_innovation_radius)
    scale = min(.5, max(1e-4, radius/math.sqrt(active)))
    estimates, diagnostics = [], []
    for _ in range(num_sweeps):
        result = conditional_random_walk_sweep(reference, stops, evaluate,
                 num_particles=num_particles, auxiliary_scale=scale, block_size=active,
                 bank_policy=bank_policy)
        reference, stops = result.innovations, result.stops
        estimates.append(result.mean)
        diagnostics.append(result.diagnostics)
    mean = torch.stack(estimates).mean(0)
    metrics = {k.replace('path_is_', 'path_is/', 1):sum(d[k] for d in diagnostics)/len(diagnostics)
               for k in diagnostics[0]}
    metrics.update({'path_is/sigma':s, 'path_is/stop':len(states),
                    'path_is/reference_replay_exact':1.,
                    'path_is/frozen_update':float(all(d['path_is_actual_move_rate']==0 for d in diagnostics)),
                    'path_is/total_mean_displacement_rms':float((mean-source.float()).square().mean().sqrt()),
                    'path_is/random_walk_auxiliary':1.})
    return mean, metrics
