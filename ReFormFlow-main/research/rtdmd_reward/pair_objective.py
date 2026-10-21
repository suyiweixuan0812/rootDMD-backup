"""Pair reward objective used by the new paper's AC-DMD experiments.

The implementation deliberately operates on already computed per-image scores.
This keeps the training loop independent of a particular VLM scorer and makes
the mathematical part testable without loading SD3.
"""

from __future__ import annotations

import torch


def pair_reward(scores: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Return one scalar pair reward for ``scores[..., 2]``.

    ``R = (S0 + S1) / 2 + lam * S0 * S1``.  Leading dimensions are preserved,
    so a batch of prompt pairs can be reduced with ``mean()`` by the caller.
    """
    if scores.shape[-1] != 2:
        raise ValueError(f"expected final dimension 2, got {tuple(scores.shape)}")
    if not torch.is_floating_point(scores):
        scores = scores.float()
    s0, s1 = scores[..., 0], scores[..., 1]
    return 0.5 * (s0 + s1) + float(lam) * s0 * s1


def pair_reward_gradients(scores: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Analytic gradient wrt the two scores, useful for validation/logging."""
    if scores.shape[-1] != 2:
        raise ValueError("scores must have final dimension 2")
    s0, s1 = scores[..., 0], scores[..., 1]
    return torch.stack((0.5 + float(lam) * s1, 0.5 + float(lam) * s0), dim=-1)
