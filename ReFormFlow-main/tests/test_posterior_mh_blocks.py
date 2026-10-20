"""Independent Gaussian posterior check with multiple active innovation banks."""
import unittest
import torch
from flow_grpo.posterior_mh import SourceState, posterior_chain


class ActiveBlockPosteriorTests(unittest.TestCase):
    def test_correlated_z_innovation_posterior_preserved_under_local_updates(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(61020)
        batch, dimension = 2048, 32
        a, sigma, observed = .6, .4, .8
        mean = a * observed / (a*a + sigma*sigma)
        variance = sigma*sigma / (a*a + sigma*sigma)
        # Exact posterior for F=(Z+xi)/sqrt(2): F|y is Gaussian and the
        # orthogonal complement remains an independent standard Gaussian.
        source = mean + variance**.5 * torch.randn((batch, dimension), generator=g)
        orthogonal = torch.randn(source.shape, generator=g)
        z = (source + orthogonal) / 2**.5
        noise = ((source - orthogonal) / 2**.5).unsqueeze(0)
        state = SourceState(z, noise, torch.full((batch,), 2, dtype=torch.long))
        evaluate = lambda state: (state.z + state.noises[0]) / 2**.5
        estimate, diagnostics = posterior_chain(torch.full_like(source, observed),
            torch.full((batch,), sigma), state, source, torch.full_like(source, .8),
            evaluate, num_moves=16, random_stop=False, proposal_scale=1.5,
            scales=(1.,), block_size=4, generator=g)
        self.assertLess(abs(estimate.mean().item() - mean), .015)
        self.assertGreater(diagnostics['mh/source_move_rate'], .05)
        self.assertLess(diagnostics['mh/accept_rate'], .99)


if __name__ == '__main__':
    unittest.main()
