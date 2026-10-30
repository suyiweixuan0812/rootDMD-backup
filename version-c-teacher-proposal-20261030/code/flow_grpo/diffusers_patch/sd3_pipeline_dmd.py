"""SD3 sampling with gradients only through the selected final step."""

import torch

from .solver import cps_step


def generate_latents(
    transformer,
    initial_latents,
    prompt_embeds,
    pooled_prompt_embeds,
    sigmas,
    total_steps,
    stop_index,
    eta,
    mixed_precision_dtype=None,
    retain_grad=False,
):
    """Return the last clean prediction and the actual post-CPS boundary."""
    if not 1 <= stop_index < len(sigmas):
        raise ValueError("stop_index must select an executed denoising step")
    sample = initial_latents
    grad_enabled = torch.is_grad_enabled()
    for index in range(stop_index):
        last = index == stop_index - 1
        with torch.set_grad_enabled(grad_enabled and retain_grad and last):
            timestep = (sigmas[index] * total_steps).expand(sample.shape[0]).float()
            # Exit autocast before the next prediction so a no-grad cast is not reused by the final step.
            with torch.autocast(
                sample.device.type,
                dtype=mixed_precision_dtype,
                enabled=mixed_precision_dtype is not None,
            ):
                velocity = transformer(
                    hidden_states=sample,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
            next_sample = cps_step(
                sample, velocity, sigmas[index], sigmas[index + 1], eta,
            )
            if last:
                x0 = (sample.float() - sigmas[index] * velocity.float()).to(sample.dtype)
                return x0, next_sample
            sample = next_sample
