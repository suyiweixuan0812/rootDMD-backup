"""Independent observation views of one source, averaged before clipping.

Keep theta, source H, sigma and optimizer state fixed while accumulating views.
Conditionally iid view gradients have unchanged mean and 1/K conditional
covariance. Variance from H/sigma remains; clipping/Adam need separate testing.
"""
import torch


def sample_query_noise_bank(source, count=2, generator=None):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Need a positive integer number of query views')
    # Draw all views before particle RNG, so the paired-source control sees the
    # same observation noises under the same pre-update model and RNG state.
    return [torch.randn(source.shape, device=source.device, dtype=torch.float32,
                        generator=generator) for _ in range(count)]
