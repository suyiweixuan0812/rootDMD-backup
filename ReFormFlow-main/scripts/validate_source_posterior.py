"""Analytic Gaussian experiment: correctness at low dimension, particle collapse at high dimension."""
import json
from pathlib import Path
import torch
from flow_grpo.source_posterior import posterior_mean


def main():
    torch.set_num_threads(1)
    rows = []
    for dimension in (1, 16, 256, 4096):
        g = torch.Generator().manual_seed(20261020 + dimension)
        source = torch.randn((32, dimension), generator=g)
        y = .5 * source + .5 * torch.randn(source.shape, generator=g)
        true_mean = y  # a*y/(a^2+s^2), a=s=.5.
        for n in (8, 32, 128):
            estimate, diagnostics = posterior_mean(
                y, torch.full((32,), .5), torch.zeros_like(y), lambda z: z,
                n, generator=torch.Generator().manual_seed(81 + n),
            )
            rows.append(dict(dimension=dimension, particles=n,
                             posterior_mean_rmse=(estimate - true_mean).square().mean().sqrt().item(),
                             **diagnostics))
    out = Path(__file__).resolve().parents[1] / 'logs/rtdmd-source-is/comparison-1000/math-validation.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(model='X~N(0,I), Y=.5X+.5eps; exact E[X|Y]=Y',
                  interpretation='Low-dimensional consistency does not prevent high-dimensional finite-particle collapse.',
                  rows=rows)
    out.write_text(json.dumps(report, indent=2))
    for r in rows:
        print(f"d={r['dimension']:4d} N={r['particles']:3d} "
              f"RMSE={r['posterior_mean_rmse']:.4f} ESS={r['posterior/ess']:.3f}")


if __name__ == '__main__':
    main()
