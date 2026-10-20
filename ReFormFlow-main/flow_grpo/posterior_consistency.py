"""Posterior-denoising consistency regularization of the same-observation surrogate.

For fixed observation and theta, the posterior-induced conditional gradient
covariance is multiplied by (1-lambda)^2. The expected update ALSO gains a
consistency term; this is a regularized method, not an unchanged KL gradient.
"""
import torch
import torch.nn.functional as F


def consistency_surrogate(prediction, posterior_mean, teacher_clean, weight=0.25, normalize=True):
    if not 0 <= weight <= 1:
        raise ValueError('Consistency weight must be in [0,1]')
    with torch.no_grad():
        gradient = ((1 - weight) * posterior_mean + weight * prediction - teacher_clean)
        if normalize:
            norm = (prediction - teacher_clean).abs().mean(
                dim=tuple(range(1, prediction.ndim)), keepdim=True)
            gradient = gradient / norm
        gradient = torch.nan_to_num(gradient)
    return 0.5 * F.mse_loss(prediction.double(), (prediction - gradient).detach().double())
