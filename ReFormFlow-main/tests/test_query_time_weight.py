import pytest
import torch
from flow_grpo.query_time_weight import query_time_weight
from flow_grpo.posterior_consistency import consistency_surrogate


def test_inverse_sigma_cancels_clean_prediction_jacobian_factor():
    # This checks an actual autograd update, not only the scalar formula.
    y=torch.tensor([[1.,-2.]],dtype=torch.float64)
    posterior=torch.tensor([[.7,.1]],dtype=torch.float64)
    teacher=torch.tensor([[.2,-.4]],dtype=torch.float64)
    expected=-.30*(y*(posterior-teacher)).mean()
    for sigma in [.2,.8]:
        theta=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
        prediction=y-sigma*theta*y
        loss=query_time_weight(sigma,'inverse_sigma',.30)*consistency_surrogate(
            prediction,posterior,teacher,weight=0.,normalize=False)
        grad,=torch.autograd.grad(loss,theta)
        torch.testing.assert_close(grad,expected)
    assert query_time_weight(.2,'none')==1.


def test_query_weight_rejects_invalid_noise_or_mode():
    for sigma in [0.,1.,-.2,float('nan')]:
        with pytest.raises(ValueError):query_time_weight(sigma)
    with pytest.raises(ValueError):query_time_weight(.5,scale=0.)
    with pytest.raises(ValueError):query_time_weight(.5,mode='ESS-dependent')
