import torch
from flow_grpo.path_ratio_smc import PathEvaluation
from flow_grpo.conditional_path_is import conditional_importance_sweep
from flow_grpo.random_walk_path_is import conditional_random_walk_sweep


def problem(batch=4096, dimension=16):
    torch.manual_seed(21)
    mu, alpha, sigma = 40., .1, .9
    variance = 1/(1+alpha**2/sigma**2)
    reference = mu+variance**.5*torch.randn(batch, 1, dimension, dtype=torch.float64)
    stops = torch.zeros(batch, dtype=torch.long)
    def evaluate(e, j):
        source = e[:, 0]
        ratio = (mu*source-.5*mu**2).sum(1)
        likelihood = -.5*((alpha*mu-alpha*source)/sigma).square().sum(1)
        zero = torch.zeros_like(ratio)
        return PathEvaluation(source, ratio+likelihood, ratio, zero, zero, likelihood)
    return reference, stops, evaluate, mu, variance


def test_random_walk_recovers_shifted_gaussian_posterior_and_reduces_variance():
    ref, stops, evaluate, mu, variance = problem()
    result = conditional_random_walk_sweep(ref, stops, evaluate, num_particles=8, auxiliary_scale=.25)
    assert abs(result.mean.double().mean()-mu) < .02
    assert result.mean.double().var() < ref.var()
    assert abs(result.innovations.mean()-mu) < .02
    assert abs(result.innovations.var()-variance) < .03
    assert result.diagnostics['path_is_rb_removed_variance_trace_per_coordinate'] > 0


def test_random_walk_can_avoid_ou_contraction_collapse_without_changing_teacher_map():
    ref, stops, evaluate, _, _ = problem()
    rw = conditional_random_walk_sweep(ref, stops, evaluate, num_particles=8, auxiliary_scale=.25)
    ou = conditional_importance_sweep(ref, stops, evaluate, num_particles=8, auxiliary_scale=.25)
    assert rw.diagnostics['path_is_ess'] > 4*ou.diagnostics['path_is_ess']
    assert rw.diagnostics['path_is_actual_move_rate'] > ou.diagnostics['path_is_actual_move_rate']+.5


def test_random_walk_block_leaves_other_coordinates_exact_and_bad_weight_is_detectable():
    ref, stops, evaluate, mu, _ = problem()
    result = conditional_random_walk_sweep(ref, stops, evaluate, num_particles=8,
                                          auxiliary_scale=.5, block_size=2)
    assert (result.innovations != ref).sum(2).max() <= 2
    assert abs(result.mean.double().mean()-mu) < .02
    # Symmetric auxiliary densities cancel each other, NOT the Gaussian prior.
    # A fake evaluator cancelling the required prior factor has a measurable bias.
    def wrong(e, j):
        value = evaluate(e, j)
        value.log_weight = value.log_weight+.5*e.square().flatten(1).sum(1)
        return value
    biased = conditional_random_walk_sweep(ref, stops, wrong, num_particles=8,
                                          auxiliary_scale=.5, block_size=2)
    assert biased.mean.double().mean()-mu > .03


def test_last_bank_preserves_earlier_innovations_and_correlated_target():
    torch.manual_seed(85)
    ref = torch.randn(4096, 3, 2, dtype=torch.float64)
    ref[:, 1] = .6*ref[:, 0]+.8*ref[:, 1]
    ref[:, 2] = .7*ref[:, 1]+(.51**.5)*ref[:, 2]
    stops = torch.full((len(ref),), 2, dtype=torch.long)
    def evaluate(e, j):
        log_target = -.5*(e[:, 0].square().sum(1)+
                         ((e[:, 1]-.6*e[:, 0])/.8).square().sum(1)+
                         ((e[:, 2]-.7*e[:, 1])/(.51**.5)).square().sum(1))
        ratio = log_target+.5*e.square().flatten(1).sum(1)
        zero = torch.zeros_like(ratio)
        return PathEvaluation(e[:, 2], ratio, ratio, zero, zero, zero)
    result = conditional_random_walk_sweep(ref, stops, evaluate, auxiliary_scale=.3,
                                           block_size=2, bank_policy='last')
    assert torch.equal(result.innovations[:, :2], ref[:, :2])
    assert abs(result.innovations[:, 2].var()-1) < .05
    assert abs((result.innovations[:, 1]*result.innovations[:, 2]).mean()-.7) < .05
    assert result.mean.double().var() < ref[:, 2].var()
