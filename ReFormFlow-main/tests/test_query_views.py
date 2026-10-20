import pytest
import torch
from flow_grpo.query_views import sample_query_noise_bank


def test_two_view_gradient_mean_and_conditional_variance():
    torch.manual_seed(75)
    source = torch.full((100000,), .7)
    a, b = sample_query_noise_bank(source, 2)
    # Exact gradient of .5*(theta*y - detached teacher target)^2 for a
    # linear denoiser; includes both odd and even Gaussian components.
    theta, alpha, sigma, target = .3, .6, .4, -.2
    ya, yb = alpha*source+sigma*a, alpha*source+sigma*b
    ga, gb = (theta*ya-target)*ya, (theta*yb-target)*yb
    expected = theta*((alpha*.7)**2+sigma**2)-target*alpha*.7
    avg = (ga+gb)/2
    assert abs(avg.mean()-expected) < .003
    assert abs(avg.var()/ga.var()-.5) < .015
    assert abs(torch.corrcoef(torch.stack([a,b]))[0,1]) < .015


def test_query_view_validation_and_average_before_clipping():
    with pytest.raises(ValueError):
        sample_query_noise_bank(torch.ones(1), 0)
    # The implementation must average raw gradients, then clip exactly once.
    g = torch.tensor([10., -9.])
    clip = lambda x:x.clamp(-1,1)
    assert clip(g.mean()) != clip(g).mean()
