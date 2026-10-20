import unittest
import torch

from flow_grpo.conditional_path_is import conditional_importance_sweep,lift_student_prefix
from flow_grpo.path_ratio_smc import PathEvaluation,evaluate_teacher_path,cps_mean


class ConditionalPathISTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_gaussian_stationary_reference_preserves_mean_and_variance(self):
        g=torch.Generator().manual_seed(81022)
        n=60000
        m,v=.7,.3
        e=(m+v**.5*torch.randn(n,1,1,generator=g,dtype=torch.float64))
        def evaluate(x,j):
            source=x[:,0]
            w=(-.5*(source-m).square()/v+.5*source.square()).sum(1)
            z=torch.zeros_like(w)
            return PathEvaluation(source,w,w,z,z,z)
        result=conditional_importance_sweep(e,torch.zeros(n,dtype=torch.long),evaluate,
            num_particles=8,auxiliary_scale=.4,generator=g)
        self.assertLess(abs(float(result.mean.mean())-m),.006)
        self.assertLess(abs(float(result.innovations.mean())-m),.006)
        self.assertLess(abs(float(result.innovations.var())-v),.008)
        self.assertGreater(result.diagnostics['path_is_actual_move_rate'],.4)

    def test_auxiliary_gaussian_density_cancellation(self):
        e=torch.tensor([.3,-.5],dtype=torch.float64)
        c=torch.tensor([.7,.8],dtype=torch.float64)
        delta=.2
        rho=(1-delta**2)**.5
        phi=torch.distributions.Normal(0.,1.)
        forward=torch.distributions.Normal(rho*e,delta).log_prob(c)
        backward=torch.distributions.Normal(rho*c,delta).log_prob(e)
        torch.testing.assert_close(phi.log_prob(e)+forward,phi.log_prob(c)+backward)

    def test_bf16_prefix_lift_reproduces_actual_student_source(self):
        g=torch.Generator().manual_seed(81023)
        initial=torch.randn(1,10,generator=g).to(torch.bfloat16)
        std=.5*torch.sin(torch.tensor(.9*torch.pi/2))
        next_state=(cps_mean(initial.float(),.4*initial.float(),1.,.5,.9)
            +std*torch.randn(1,10,generator=g)).to(torch.bfloat16)
        teacher=lambda x,s:.1*x.float()
        e=lift_student_prefix([initial,next_state],[1.,.5],.9,teacher,generator=g)
        result=evaluate_teacher_path(e,torch.ones(1,dtype=torch.long),torch.zeros(1,10),.4,[1.,.5],.9,
            lambda x,s:.4*x.float(),teacher,state_dtype=torch.bfloat16)
        expected=(next_state.float()-.5*.4*next_state.float()).to(torch.bfloat16)
        torch.testing.assert_close(result.source,expected,atol=0,rtol=0)

    def test_no_claim_for_arbitrary_nonposterior_reference(self):
        # The constructor accepts a tensor; theory requires a posterior reference.
        # This test demonstrates that a bad initialization is not repaired magically.
        e=torch.full((100,1,1),100.)
        def evaluate(x,j):
            source=x[:,0]
            w=-source.square().sum(1)
            z=torch.zeros_like(w)
            return PathEvaluation(source,w,w,z,z,z)
        result=conditional_importance_sweep(e,torch.zeros(100,dtype=torch.long),evaluate,
            auxiliary_scale=.001,generator=torch.Generator().manual_seed(9))
        self.assertGreater(float(result.mean.mean()),90.)


if __name__=='__main__':unittest.main()
