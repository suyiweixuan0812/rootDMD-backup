import unittest
import torch
from flow_grpo.posterior_consistency import consistency_surrogate


class PosteriorConsistencyTests(unittest.TestCase):
    def test_lambda_zero_is_original_gradient(self):
        x = torch.tensor([[1., 2.]], requires_grad=True)
        m, r = torch.tensor([[.4, .8]]), torch.tensor([[.2, .3]])
        gradient, = torch.autograd.grad(consistency_surrogate(x,m,r,0,False),x)
        torch.testing.assert_close(gradient, (m-r)/x.numel())

    def test_lambda_one_cancels_posterior_noise(self):
        for m in [torch.zeros(1,2), torch.full((1,2),100.)]:
            x = torch.tensor([[1.,2.]], requires_grad=True)
            r = torch.tensor([[.2,.3]])
            gradient, = torch.autograd.grad(consistency_surrogate(x,m,r,1,False),x)
            torch.testing.assert_close(gradient,(x.detach()-r)/x.numel())

    def test_conditional_gradient_variance_factor(self):
        torch.manual_seed(23)
        posterior = torch.randn(20000,1)
        fixed_prediction, fixed_teacher = .6, .2
        original = posterior-fixed_teacher
        for weight in [.25,.5]:
            regularized = (1-weight)*posterior+weight*fixed_prediction-fixed_teacher
            ratio = regularized.var()/original.var()
            self.assertLess(abs(ratio.item()-(1-weight)**2),1e-6)

    def test_expected_update_gains_consistency_term(self):
        x=torch.tensor([[.6]],requires_grad=True)
        m,r=torch.tensor([[.4]]),torch.tensor([[.2]])
        gradient,=torch.autograd.grad(consistency_surrogate(x,m,r,.25,False),x)
        torch.testing.assert_close(gradient,(m-r)+.25*(x.detach()-m))

    def test_posterior_and_teacher_targets_are_detached(self):
        x=torch.tensor([[1.,2.]],requires_grad=True)
        m=torch.tensor([[.4,.8]],requires_grad=True)
        r=torch.tensor([[.2,.3]],requires_grad=True)
        consistency_surrogate(x,m,r,.25,True).backward()
        self.assertIsNone(m.grad)
        self.assertIsNone(r.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_weight_rejected(self):
        with self.assertRaises(ValueError):
            consistency_surrogate(torch.ones(1,1),torch.ones(1,1),torch.zeros(1,1),1.1)

if __name__=='__main__': unittest.main()
