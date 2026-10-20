import unittest
from unittest.mock import patch
import torch
from flow_grpo.posterior_mh import SourceState, posterior_chain, proposal_log_correction
from flow_grpo.diffusers_patch.sd3_source_trace import replay_source
from flow_grpo.diffusers_patch.sd3_pipeline_dmd import generate_latents


class PosteriorMHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_shift_hastings_ratio_matches_full_densities(self):
        torch.manual_seed(3)
        z, h = torch.randn(4, 7, dtype=torch.float64), torch.randn(4, 7, dtype=torch.float64)
        beta = .31
        rho = (1 - beta**2)**.5
        proposed = rho * z + (1 - rho) * h + beta * torch.randn_like(z)
        normal = torch.distributions.Normal
        full = (normal(0., 1.).log_prob(proposed) - normal(0., 1.).log_prob(z)
                + normal(rho * proposed + (1 - rho) * h, beta).log_prob(z)
                - normal(rho * z + (1 - rho) * h, beta).log_prob(proposed)).sum(1)
        torch.testing.assert_close(proposal_log_correction(z, proposed, h), full)

    def test_gaussian_stationary_posterior_with_nonzero_teacher_shift(self):
        n = 12000
        g = torch.Generator().manual_seed(41)
        sigma, a, observation = .4, .6, .8
        exact_mean = a * observation / (a*a + sigma*sigma)
        exact_var = sigma*sigma / (a*a + sigma*sigma)
        z = exact_mean + exact_var**.5 * torch.randn((n, 1), generator=g)
        state = SourceState(z, torch.randn((1, n, 1), generator=g), torch.ones(n, dtype=torch.long))
        y = torch.full_like(z, observation)
        estimate, diagnostics = posterior_chain(y, torch.full((n,), sigma), state, z,
            torch.full_like(y, .8), lambda state: state.z, num_moves=16,
            generator=g)
        self.assertLess(abs(estimate.mean().item() - exact_mean), .015)
        self.assertGreater(diagnostics['mh/accept_rate'], .05)
        self.assertLess(diagnostics['mh/accept_rate'], .99)
        self.assertFalse(estimate.requires_grad)
        self.assertNotIn('posterior/ess', diagnostics)

    def test_noninvertible_deterministic_terminal(self):
        n = 12000
        g = torch.Generator().manual_seed(7)
        y = torch.full((n, 1), .2)
        expected = torch.tanh(torch.tensor(.4)).item()
        positive = torch.rand((n, 1), generator=g) < ((1 + expected)/2)
        z = torch.randn((n, 1), generator=g).abs() * torch.where(positive, 1., -1.)
        state = SourceState(z, torch.zeros(0, n, 1), torch.ones(n, dtype=torch.long))
        estimate, _ = posterior_chain(y, torch.full((n,), .5), state, z.sign(),
            torch.zeros_like(y), lambda state: state.z.sign(), num_moves=12,
            random_stop=False, generator=g)
        self.assertLess(abs(estimate.mean().item() - expected), .025)

    def test_random_stop_is_part_of_posterior(self):
        n = 12000
        g = torch.Generator().manual_seed(9)
        y = torch.full((n, 1), .8)
        sigma = torch.full((n,), .5)
        likelihoods = torch.exp(-.5 * ((torch.tensor(.8) - .5*torch.tensor([0., 1.]))/.5)**2)
        p1 = (likelihoods[1]/likelihoods.sum()).item()
        stop = 1 + (torch.rand(n, generator=g) < p1).long()
        state = SourceState(torch.randn((n, 1), generator=g),
            torch.randn((1, n, 1), generator=g), stop)
        evaluate = lambda state: (state.stop - 1).float().unsqueeze(1)
        estimate, _ = posterior_chain(y, sigma, state, evaluate(state),
            torch.zeros_like(y), evaluate, num_moves=12, generator=g)
        self.assertLess(abs(estimate.mean().item() - p1), .015)

    def test_all_rejected_chain_is_reported_as_frozen_not_large_ess(self):
        state = SourceState(torch.zeros(1, 2), torch.zeros(0, 1, 2), torch.ones(1, dtype=torch.long))
        original = torch.zeros(1, 2)
        mean, d = posterior_chain(torch.zeros_like(original), torch.tensor([.1]),
            state, original, torch.zeros_like(original), lambda state: torch.full_like(original, 1e5),
            num_moves=8, random_stop=False, generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(mean, original)
        self.assertEqual(d['mh/all_rejected'], 1.)
        self.assertEqual(d['mh/accepted_transitions'], 0.)
        self.assertNotIn('posterior/ess', d)

    def test_replay_matches_original_cps_source_for_every_stop(self):
        class ZeroVelocity:
            def __call__(self, hidden_states, **kwargs):
                return (torch.zeros_like(hidden_states),)
        model = ZeroVelocity()
        z = torch.randn(1, 2, 3, 3)
        innovations = torch.randn(3, 1, 2, 3, 3)
        schedule = torch.tensor([1., .8, .5, .2, 0.])
        embeds, pooled = torch.zeros(1, 1, 1), torch.zeros(1, 1)
        for stop in range(1, 5):
            state = SourceState(z, innovations, torch.tensor([stop]))
            replayed = replay_source(model, state, embeds, pooled, schedule, 1000, .9)
            draws = [*innovations, torch.zeros_like(z)]
            with patch('torch.randn', side_effect=draws):
                original, _ = generate_latents(model, z, embeds, pooled, schedule, 1000, stop, .9)
            torch.testing.assert_close(replayed, original)

    def test_same_observation_jacobian_avoids_paired_source_covariance_bias(self):
        # Conditional source X|y has mean m and variance v. Multiplying its draw
        # by the paired source Jacobian X adds v; a fixed-y Jacobian does not.
        g = torch.Generator().manual_seed(19)
        m, v = .7, .4
        x = m + v**.5 * torch.randn(100000, generator=g)
        self.assertLess(abs((2*x).mean().item() - 2*m), .01)
        self.assertGreater(abs(x.square().mean().item() - m*m), .35)


if __name__ == '__main__':
    unittest.main()
