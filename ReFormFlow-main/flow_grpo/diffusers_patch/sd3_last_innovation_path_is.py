"""Last positive innovation block, exact cached prefix fields, pure Teacher map."""
import math
import torch
from flow_grpo.path_ratio_smc import cps_mean, bf16_gaussian_log_ratio_at
from flow_grpo.physical_path_is import lift_physical_gaussian_cell, evaluate_last_physical_state
from flow_grpo.random_walk_path_is import conditional_random_walk_sweep
from .sd3_conditional_path_is import make_fields, sample_prefix


def cached_prefix_fields(student, teacher, states, positive_sigmas):
    """Only unchanged prefix parents can use cache; any mismatch fails loudly.

    Calls are batch1 before expanding the exact result, matching field_microbatch1.
    The current model parameters must stay fixed for the lifetime of this cache.
    """
    parents = {float(positive_sigmas[k]): states[k] for k in range(len(states)-1)}
    values = {}
    calls = {'student_network_calls': 0, 'teacher_network_calls': 0, 'cached_field_hits': 0}
    def wrap(field, role):
        @torch.no_grad()
        def predict(x, s):
            s = float(s)
            if s not in parents:
                calls[role+'_network_calls'] += x.shape[0]
                return field(x, s)
            parent = parents[s]
            if not torch.equal(x, parent.expand(x.shape[0], *parent.shape[1:])):
                raise RuntimeError('Last-bank cache saw a changed prefix parent')
            key = (role, s)
            if key not in values:
                values[key] = field(parent, s)
                calls[role+'_network_calls'] += 1
            else:
                calls['cached_field_hits'] += 1
            return values[key].expand(x.shape[0], *values[key].shape[1:])
        return predict
    return wrap(student, 'student'), wrap(teacher, 'teacher'), calls


@torch.no_grad()
def conditional_path_mean(observation, sigma, states, source, student, teacher,
                          positive_sigmas, eta, **kwargs):
    if kwargs.pop('bank_policy', 'last') != 'last':
        raise ValueError('Cached prefix only supports the last innovation bank')
    if source.shape[0] != 1:
        raise ValueError('Last-node conditional sampling requires batch1')
    grid = [float(s) for s in positive_sigmas]
    last = len(states)-1
    cached_student, cached_teacher, calls = cached_prefix_fields(student, teacher, states, grid)
    prefix_ratio = torch.zeros(1, device=source.device, dtype=torch.float64)
    target_mean = teacher_mean = torch.zeros_like(states[0], dtype=torch.float32)
    std = 1.
    # All earlier physical parents are conditioned on, so their ratios are
    # constant across candidates. No inversion of inactive innovations is needed.
    for k in range(last):
        s, t = grid[k], grid[k+1]
        x = states[k].float()
        target_mean = cps_mean(x, cached_student(states[k], s).float(), s, t, eta)
        teacher_mean = cps_mean(x, cached_teacher(states[k], s).float(), s, t, eta)
        std = t*math.sin(eta*math.pi/2)
        if k < last-1:
            prefix_ratio += bf16_gaussian_log_ratio_at(states[k+1], target_mean, teacher_mean, std)
    reference = lift_physical_gaussian_cell(states[-1], teacher_mean, std)[:, None]
    stops = torch.zeros(1, device=source.device, dtype=torch.long)
    def readout(x):
        return (x.float()-grid[last]*cached_student(x, grid[last]).float()).to(source.dtype)
    def evaluate(u, j):
        return evaluate_last_physical_state(u, observation, float(sigma.item()), target_mean,
                                            teacher_mean, std, readout, prefix_ratio)
    if not torch.equal(evaluate(reference, stops).source, source):
        raise RuntimeError('Physical last-node reference readout mismatch')
    s = float(sigma.item())
    block = kwargs.get('high_noise_block_size',4096) if s >= kwargs.get('high_noise_threshold',.9) else kwargs.get('block_size',16)
    radius = min(kwargs.get('base_scale',.05)*s/(1-s), kwargs.get('max_innovation_radius',.5))
    delta = min(.5, max(1e-4, radius/math.sqrt(block)))
    means, diagnostics = [], []
    for _ in range(kwargs.get('num_sweeps',2)):
        result = conditional_random_walk_sweep(reference, stops, evaluate,
            num_particles=kwargs.get('num_particles',8), auxiliary_scale=std*delta,
            block_size=block, bank_policy='last', gaussian_mean=teacher_mean[:,None], gaussian_std=std)
        reference = result.innovations
        result.diagnostics['path_is_physical_auxiliary_scale'] = std*delta
        result.diagnostics['path_is_auxiliary_scale'] = delta
        means.append(result.mean)
        diagnostics.append(result.diagnostics)
    mean = torch.stack(means).mean(0)
    metrics = {k.replace('path_is_','path_is/',1):sum(d[k] for d in diagnostics)/len(diagnostics)
               for k in diagnostics[0]}
    metrics.update({'path_is/sigma':s, 'path_is/stop':len(states), 'path_is/reference_replay_exact':1.,
        'path_is/frozen_update':float(all(d['path_is_actual_move_rate']==0 for d in diagnostics)),
        'path_is/total_mean_displacement_rms':float((mean-source.float()).square().mean().sqrt()),
        'path_is/physical_reference_coordinates':1.})
    metrics.update({'path_is/'+key: float(value) for key, value in calls.items()})
    metrics['path_is/prefix_cache_exact_checks'] = 1.
    return mean, metrics
