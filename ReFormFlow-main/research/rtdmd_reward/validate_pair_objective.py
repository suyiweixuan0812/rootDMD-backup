"""Fast CPU/CUDA validation of the pair-reward claims in the attached paper."""

import json
import sys
from pathlib import Path

import torch

from pair_objective import pair_reward, pair_reward_gradients


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    scores = torch.rand(4096, 2, device=device, dtype=torch.float64, requires_grad=True)
    lam = 1.0
    got = pair_reward(scores, lam).mean()
    got.backward()
    expected = pair_reward_gradients(scores.detach(), lam) / scores.shape[0]
    max_err = (scores.grad - expected).abs().max().item()
    # lambda=0 must reduce exactly to the pooled single-image objective.
    pooled = pair_reward(scores.detach(), 0.0)
    pooled_expected = scores.detach().mean(dim=-1)
    pooled_err = (pooled - pooled_expected).abs().max().item()
    result = {"device": str(device), "max_autograd_error": max_err, "lambda0_error": pooled_err}
    print(json.dumps(result, indent=2))
    if max_err > 1e-12 or pooled_err > 1e-15:
        raise SystemExit("pair objective validation failed")


if __name__ == "__main__":
    main()
