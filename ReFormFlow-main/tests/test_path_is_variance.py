"""Check the RB variance identity and fixed-stop floor on known posteriors."""
import torch
from flow_grpo.conditional_path_is import conditional_importance_sweep
from flow_grpo.path_ratio_smc import PathEvaluation


def test_stationary_pool_rb_variance_identity():
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(72804)
    n=100000
    mean,variance=.7,.3
    e=mean+variance**.5*torch.randn(n,1,1,generator=g,dtype=torch.float64)
    recorded={}
    def evaluate(x,j):
        source=x[:,0]
        w=(-.5*(source-mean).square()/variance+.5*source.square()).sum(1)
        z=torch.zeros_like(w)
        recorded['source'],recorded['w']=source.reshape(n,8),w.reshape(n,8).softmax(1)
        return PathEvaluation(source,w,w,z,z,z)
    r=conditional_importance_sweep(e,torch.zeros(n,dtype=torch.long),evaluate,
        num_particles=8,auxiliary_scale=.4,generator=g)
    weighted_variance=(recorded['w']*(recorded['source']-r.mean.double()).square()).sum(1).mean()
    removed=r.innovations.var()-r.mean.var()
    assert weighted_variance>.03
    assert abs(float(removed-weighted_variance))<.006
    assert r.mean.var()<variance-.025


def test_fixed_stop_retains_between_stop_variance():
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(72805)
    n=100000
    j=torch.arange(n)%2
    means=(2*j-1).double()
    e=(means+.3**.5*torch.randn(n,generator=g,dtype=torch.float64)).reshape(n,1,1)
    def evaluate(x,stop):
        source=x[:,0]
        m=(2*stop-1).double()[:,None]
        w=(-.5*(source-m).square()/.3+.5*source.square()).sum(1)
        z=torch.zeros_like(w)
        return PathEvaluation(source,w,w,z,z,z)
    r=conditional_importance_sweep(e,j,evaluate,num_particles=8,auxiliary_scale=.4,generator=g)
    # This toy uses j only as a mixture label; block_size=None avoids any bank use.
    assert torch.equal(r.stops,j)
    assert float(r.mean.var())>1.0
    assert float(r.mean.var())<1.3
