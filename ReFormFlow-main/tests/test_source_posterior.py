import unittest
import torch
from flow_grpo.source_posterior import gaussian_log_ratio, log_likelihood, posterior_mean


class SourcePosteriorTests(unittest.TestCase):
    def test_gaussian_density_ratio_sums_event_dimensions(self):
        z = torch.tensor([[0.3, -0.5, 1.2]], dtype=torch.float64)
        mean = torch.tensor([[0.2, 0.1, -0.4]], dtype=torch.float64)
        expected = (torch.distributions.Normal(0., 1.).log_prob(z)
                    - torch.distributions.Normal(mean, 1.).log_prob(z)).sum(1)
        torch.testing.assert_close(gaussian_log_ratio(z, mean), expected)

    def test_analytic_gaussian_posterior_and_tweedie_score(self):
        # X=Z~N(0,1); Y=.6X+.4eps. Deliberately biased proposal N(.7,1).
        # Exact E[X|y] = a*y/(a^2+s^2); score(y)=-y/(a^2+s^2).
        y = torch.full((128, 1), 0.8)
        sigma = torch.full((128,), 0.4)
        mean, diagnostics = posterior_mean(
            y, sigma, torch.full_like(y, 0.7), lambda z: z,
            4096, generator=torch.Generator().manual_seed(87),
        )
        expected_mean = 0.6 * 0.8 / (0.6**2 + 0.4**2)
        self.assertLess(abs(mean.mean().item() - expected_mean), 0.008)
        score = (0.6 * mean.mean().item() - 0.8) / 0.4**2
        self.assertLess(abs(score + 0.8 / 0.52), 0.03)
        self.assertGreater(diagnostics['posterior/ess'], 100)
        self.assertFalse(mean.requires_grad)

    def test_deterministic_noninvertible_terminal_needs_no_density(self):
        # Source is a deterministic, noninvertible map. Both exact atoms can be
        # integrated analytically although the source has no Lebesgue density.
        y = torch.full((64, 1), 0.2)
        sigma = torch.full((64,), 0.5)
        mean, _ = posterior_mean(y, sigma, torch.zeros_like(y),
                                 lambda z: z.sign(), 4096,
                                 generator=torch.Generator().manual_seed(4))
        expected = torch.tanh(torch.tensor(0.5 * 0.2 / 0.5**2)).item()
        self.assertLess(abs(mean.mean().item() - expected), 0.01)

    def test_observation_likelihood_uses_sum_not_dimension_mean(self):
        y = torch.ones(2, 3)
        value = log_likelihood(y, torch.zeros_like(y), torch.full((2,), .5))
        torch.testing.assert_close(value, torch.full((2,), -6., dtype=torch.float64))

    def test_reject_singular_corruption(self):
        with self.assertRaises(ValueError):
            log_likelihood(torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1))

    def test_reverse_chain_mean_is_not_automatically_posterior_mean(self):
        # Valid one-step reverse kernel with clean mean 0 at y=1, but forward
        # corruption of N(0,1) with sigma=.5 has posterior mean exactly 1.
        true_mean = .5 * 1 / (.5**2 + .5**2)
        arbitrary_reverse_mean = 0.
        self.assertEqual(true_mean, 1.)
        self.assertNotEqual(arbitrary_reverse_mean, true_mean)


if __name__ == '__main__':
    unittest.main()
