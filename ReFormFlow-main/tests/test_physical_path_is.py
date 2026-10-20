import torch
from flow_grpo.physical_path_is import lift_physical_gaussian_cell, evaluate_last_physical_state
from flow_grpo.path_ratio_smc import bf16_gaussian_log_mass
from flow_grpo.random_walk_path_is import conditional_random_walk_sweep


def test_physical_lift_preserves_zero_and_tiny_cells_with_nonzero_mean():
    torch.manual_seed(83)
    x = torch.tensor([[0., 1e-30, -1e-30, .5]], dtype=torch.bfloat16)
    mean = torch.full_like(x, .7, dtype=torch.float32)
    # The old standardized bounds collapse at the zero cell in FP64.
    lo = .5*(x.double()+torch.nextafter(x, torch.full_like(x, -float('inf'))).double())
    hi = .5*(x.double()+torch.nextafter(x, torch.full_like(x, float('inf'))).double())
    assert ((lo-mean.double())/.4 == (hi-mean.double())/.4)[0, 0]
    u = lift_physical_gaussian_cell(x, mean, .4)
    assert torch.equal(u.to(torch.bfloat16), x)
    assert torch.isfinite(u).all()


def test_physical_sampler_matches_enumerated_bf16_posterior():
    torch.manual_seed(84)
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    grid = torch.unique(values[torch.isfinite(values) & (values.abs()<4)].float()).to(torch.bfloat16)[:,None]
    mu_s, mu_t, std, sigma, y = .7, -.2, .4, .7, .5
    s = torch.tensor([[mu_s]])
    t = torch.tensor([[mu_t]])
    log_target = bf16_gaussian_log_mass(grid, s, std).flatten()-.5*((y-(1-sigma)*grid.double().flatten())/sigma).square()
    prob = log_target.softmax(0)
    exact_mean = (prob*grid.double().flatten()).sum()
    x = grid[torch.multinomial(prob, 4096, replacement=True)]
    reference = lift_physical_gaussian_cell(x, t, std)[:,None]
    stops = torch.zeros(len(x), dtype=torch.long)
    def evaluate(u, j):
        return evaluate_last_physical_state(u, torch.tensor([[y]]), sigma, s, t, std, lambda z:z)
    result = conditional_random_walk_sweep(reference, stops, evaluate, auxiliary_scale=.15,
        block_size=1, bank_policy='last', gaussian_mean=t[:,None], gaussian_std=std)
    assert abs(result.mean.double().mean()-exact_mean) < .02
    assert result.mean.double().var() < x.double().var()
    assert result.diagnostics['path_is_rb_removed_variance_trace_per_coordinate'] > 0


def test_affine_coordinate_change_keeps_same_local_importance_kernel():
    torch.manual_seed(86)
    e = torch.randn(1, 1, 4, dtype=torch.float64)
    mu, std = torch.full_like(e, .3), .4
    u = mu+std*e
    target = torch.full((1,4), .7)
    teacher = mu[:,0]
    def physical(v, j):
        return evaluate_last_physical_state(v, torch.zeros(1,4), .7, target, teacher, std, lambda x:x)
    def standardized(v, j):return physical(mu+std*v, j)
    a = conditional_random_walk_sweep(e, torch.zeros(1,dtype=torch.long), standardized,
        auxiliary_scale=.2, block_size=4, bank_policy='last', generator=torch.Generator().manual_seed(17))
    b = conditional_random_walk_sweep(u, torch.zeros(1,dtype=torch.long), physical,
        auxiliary_scale=std*.2, block_size=4, bank_policy='last', gaussian_mean=mu, gaussian_std=std,
        generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(a.mean, b.mean, rtol=0, atol=1e-6)
    assert abs(a.diagnostics['path_is_ess']-b.diagnostics['path_is_ess']) < 1e-9
