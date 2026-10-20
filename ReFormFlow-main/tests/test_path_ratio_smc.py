import math
import unittest

import torch

from flow_grpo.path_ratio_smc import (
    PathEvaluation, TemperingBudgetExceeded, annealed_path_smc, cps_mean,
    effective_sample_size, evaluate_teacher_path, gaussian_log_ratio_at,
    normalized_log_weights,
    bf16_gaussian_log_mass, bf16_gaussian_log_ratio_at,
)


class PathRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_kernel_ratio_matches_normal_distribution_and_sums_coordinates(self):
        x = torch.tensor([[.3, -.4, .7]], dtype=torch.float64)
        a, b = x*.4, x+.2
        expected = (torch.distributions.Normal(a,.3).log_prob(x)
                    - torch.distributions.Normal(b,.3).log_prob(x)).sum(1)
        torch.testing.assert_close(gaussian_log_ratio_at(x,a,b,.3), expected)

    def test_cps_mean_and_noise_match_original_solver(self):
        from flow_grpo.diffusers_patch.solver import cps_step
        x, v, noise = torch.randn(4,2), torch.randn(4,2), torch.randn(4,2)
        expected = cps_mean(x,v,1.,.5,.9) + .5*math.sin(.9*math.pi/2)*noise
        torch.testing.assert_close(cps_step(x,v,1.,.5,.9,noise=noise),expected)

    def test_reject_zero_variance_instead_of_regularizing_silently(self):
        with self.assertRaises(ValueError):
            gaussian_log_ratio_at(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1,1),0.)
        with self.assertRaises(ValueError):
            evaluate_teacher_path(torch.zeros(2,2,1),torch.ones(2,dtype=torch.long),
                torch.zeros(1,1),.4,[1.,0.],.9,lambda x,s:x,lambda x,s:x)

    def test_bf16_probability_mass_matches_normal_cell_integral(self):
        x=torch.tensor([[.5,-.3]],dtype=torch.bfloat16)
        lower=.5*(torch.nextafter(x,torch.full_like(x,-float('inf'))).double()+x.double())
        upper=.5*(torch.nextafter(x,torch.full_like(x,float('inf'))).double()+x.double())
        normal=torch.distributions.Normal(torch.tensor(.1,dtype=torch.float64),torch.tensor(.3,dtype=torch.float64))
        expected=(normal.cdf(upper)-normal.cdf(lower)).log()
        torch.testing.assert_close(bf16_gaussian_log_mass(x,torch.full_like(x,.1).double(),.3),
            (torch.distributions.Normal(torch.full_like(x,.1).double(),.3).cdf(upper)
             -torch.distributions.Normal(torch.full_like(x,.1).double(),.3).cdf(lower)).log(),atol=1e-10,rtol=1e-10)

    def test_bf16_zero_cell_mass_stays_finite(self):
        x=torch.zeros(1,2,dtype=torch.bfloat16)
        value=bf16_gaussian_log_mass(x,torch.ones(1,2),.3)
        self.assertTrue(torch.isfinite(value).all())
        ratio=bf16_gaussian_log_ratio_at(x,torch.zeros(1,2),torch.ones(1,2),.3)
        self.assertAlmostEqual(float(ratio),1/.3**2,places=8)

    def test_bf16_identical_teacher_student_has_unit_path_ratio(self):
        g=torch.Generator().manual_seed(72014)
        e=torch.randn(32,3,2,generator=g)
        result=evaluate_teacher_path(e,torch.arange(32)%3,torch.zeros(1,2),.4,[1.,.7,.3],.9,
            lambda x,s:.4*x,lambda x,s:.4*x,state_dtype=torch.bfloat16)
        self.assertEqual(result.source.dtype,torch.bfloat16)
        torch.testing.assert_close(result.log_path_ratio,torch.zeros(32,dtype=torch.float64))

    def test_last_readout_is_student_not_teacher(self):
        e = torch.tensor([[[.4],[.2]],[[.8],[-.2]]],dtype=torch.float64)
        result=evaluate_teacher_path(e,torch.ones(2,dtype=torch.long),torch.zeros(1,1),
            .4,[1.,.5],.9,lambda x,s:.4*x,lambda x,s:.1*x)
        last=cps_mean(e[:,0],.1*e[:,0],1.,.5,.9)+.5*math.sin(.9*math.pi/2)*e[:,1]
        torch.testing.assert_close(result.source,.8*last)
        self.assertFalse(torch.allclose(result.source,.95*last))

    def test_unused_teacher_suffix_cancels_for_first_stop(self):
        e=torch.randn(20,3,2,dtype=torch.float64)
        kwargs=dict(stops=torch.zeros(20,dtype=torch.long),observation=torch.zeros(1,2),
            sigma_observation=.4,positive_sigmas=[1.,.7,.3],eta=.9,
            student_velocity=lambda x,s:.4*x,teacher_velocity=lambda x,s:.1*x)
        a=evaluate_teacher_path(e,**kwargs)
        e[:,1:]+=1000
        b=evaluate_teacher_path(e,**kwargs)
        torch.testing.assert_close(a.log_weight,b.log_weight)
        torch.testing.assert_close(a.log_path_ratio,torch.zeros(20,dtype=torch.float64))

    def test_initial_and_stop_proposal_corrections(self):
        e=torch.randn(30,2,1,dtype=torch.float64)
        stops=torch.arange(30)%2
        result=evaluate_teacher_path(e,stops,torch.zeros(1,1),.4,[1.,.5],.9,
            lambda x,s:.4*x,lambda x,s:.1*x,initial_mean=torch.tensor([.3]),
            target_stop_probs=[.7,.3],proposal_stop_probs=[.2,.8])
        x=e[:,0]+.3
        expected=(torch.distributions.Normal(0.,1.).log_prob(x)
                  -torch.distributions.Normal(.3,1.).log_prob(x)).sum(1)
        torch.testing.assert_close(result.log_initial_ratio,expected)
        torch.testing.assert_close(result.log_stop_ratio,
            (torch.tensor([.7,.3],dtype=torch.float64).log()
             -torch.tensor([.2,.8],dtype=torch.float64).log())[stops])

    def test_random_stop_posterior_matches_analytic_gaussian_mixture(self):
        g=torch.Generator().manual_seed(72011)
        n=150000
        e=torch.randn(n,2,1,generator=g,dtype=torch.float64)
        p=torch.tensor([.7,.3],dtype=torch.float64)
        q=torch.tensor([.2,.8],dtype=torch.float64)
        stops=torch.multinomial(q,n,replacement=True,generator=g)
        y,sigma=.7,.4
        result=evaluate_teacher_path(e,stops,torch.tensor([[y]]),sigma,[1.,.5],.9,
            lambda x,s:.4*x,lambda x,s:.1*x,initial_mean=torch.tensor([.2]),
            target_stop_probs=p,proposal_stop_probs=q)
        means=torch.zeros(1,1,dtype=torch.float64)
        a=float(cps_mean(torch.ones_like(means),.4*torch.ones_like(means),1.,.5,.9))
        variances=torch.tensor([.6**2,.8**2*(a*a+(.5*math.sin(.9*math.pi/2))**2)])
        observed_variances=.6**2*variances+sigma**2
        mass=(p.log()+torch.distributions.Normal(0.,observed_variances.sqrt()).log_prob(torch.tensor(y))).softmax(0)
        expected=(mass*(.6*variances/observed_variances*y)).sum()
        estimate=(result.log_weight.softmax(0)[:,None]*result.source).sum()
        self.assertLess(abs(float(estimate-expected)),.009)

    def test_noninvertible_student_terminal_sign_has_correct_posterior(self):
        g=torch.Generator().manual_seed(72012)
        n=100000
        e=torch.randn(n,2,1,generator=g,dtype=torch.float64)
        stops=torch.randint(0,2,(n,),generator=g)
        # Student source is sign(x), regardless of which symmetric prior prefix.
        student=lambda x,s:(x-x.sign())/s
        result=evaluate_teacher_path(e,stops,torch.tensor([[.2]]),.5,[1.,.5],.9,
            student,lambda x,s:.15*x+.2)
        estimate=(result.log_weight.softmax(0)[:,None]*result.source).sum()
        self.assertLess(abs(float(estimate)-math.tanh(.4)),.012)

    def test_pcn_acceptance_keeps_path_weight_in_tempered_target(self):
        g=torch.Generator().manual_seed(72013)
        e=torch.randn(7,4,generator=g,dtype=torch.float64)
        f=.95*e+math.sqrt(1-.95**2)*torch.randn(e.shape,generator=g,dtype=torch.float64)
        b=.6
        lw=lambda x:-(x-.8).square().sum(1)
        normal=torch.distributions.Normal(0.,1.)
        log_prior_delta=(normal.log_prob(f)-normal.log_prob(e)).sum(1)
        reverse=torch.distributions.Normal(.95*f,math.sqrt(1-.95**2)).log_prob(e).sum(1)
        forward=torch.distributions.Normal(.95*e,math.sqrt(1-.95**2)).log_prob(f).sum(1)
        full=log_prior_delta+b*(lw(f)-lw(e))+reverse-forward
        torch.testing.assert_close(full,b*(lw(f)-lw(e)))

    @staticmethod
    def toy_evaluator(e,stops):
        source=e[:,0]
        ratio=100*source[:,0]
        zero=torch.zeros_like(ratio)
        return PathEvaluation(source,ratio,ratio,zero,zero,zero)

    def test_resampling_ess_does_not_claim_particle_diversity(self):
        e=torch.arange(-8.,8.).reshape(16,1,1)
        result=annealed_path_smc(e,torch.zeros(16,dtype=torch.long),self.toy_evaluator,
            mutation_steps=0,max_stages=128,generator=torch.Generator().manual_seed(3))
        self.assertEqual(result.beta,1.)
        self.assertLess(result.distinct_initial_ancestors,16)
        self.assertLess(result.distinct_final_sources,16)
        self.assertTrue(all(r['mutation_actual_move_rate']==0 for r in result.history))
        self.assertTrue(all(r['ess_before_resampling']>=.8*16-1e-6 for r in result.history))

    def test_stage_budget_does_not_return_an_unfinished_target(self):
        e=torch.arange(-8.,8.).reshape(16,1,1)
        with self.assertRaises(TemperingBudgetExceeded):
            annealed_path_smc(e,torch.zeros(16,dtype=torch.long),self.toy_evaluator,
                mutation_steps=0,max_stages=1,generator=torch.Generator().manual_seed(3))

    def test_missing_likelihood_does_not_recover_forward_posterior(self):
        # Even if path ratio is exactly one (Student=Teacher), posterior != prior.
        y,alpha,sigma=1.,.6,.4
        true_mean=alpha*y/(alpha*alpha+sigma*sigma)
        self.assertNotEqual(true_mean,0.)
        self.assertAlmostEqual(true_mean,1.1538461538461537)


if __name__=='__main__':
    unittest.main()
