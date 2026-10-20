"""Replay the exact random-stop clean source map using explicit CPS innovations."""
import torch
from .solver import cps_step


@torch.no_grad()
def replay_source(transformer, state, embeds, pooled, sigmas, total_steps, eta,
                  mixed_precision_dtype=None):
    # This experiment is single-A100/batch1; keep the shared-stop law identical.
    if state.z.shape[0] != 1:
        raise ValueError('The SD3 trace replay currently requires batch_size=1')
    stop = int(state.stop.item())
    if not 1 <= stop < len(sigmas):
        raise ValueError('Invalid random-stop trace')
    sample = state.z.to(embeds.dtype)
    for index in range(stop):
        with torch.autocast(sample.device.type, dtype=mixed_precision_dtype,
                            enabled=mixed_precision_dtype is not None):
            velocity = transformer(hidden_states=sample,
                timestep=(sigmas[index] * total_steps).expand(sample.shape[0]).float(),
                encoder_hidden_states=embeds, pooled_projections=pooled,
                return_dict=False)[0]
        if index == stop - 1:
            return (sample.float() - sigmas[index] * velocity.float()).to(sample.dtype)
        sample = cps_step(sample, velocity, sigmas[index], sigmas[index + 1], eta,
                          noise=state.noises[index])
    raise AssertionError('Unreachable source replay')
