import pytest
import torch
from flow_grpo.adaptive_bridge import PastKLBridge,source_teacher_kernel_kls


def test_past_bridge_maximizes_teacher_fraction_under_proxy_budget():
    policy=PastKLBridge(nodes=4,kl_budget=.25,decay=.95)
    assert policy.strength(4)==0.
    policy.observe([10.,20.,30.])
    for stop in [1,2,3,4]:
        kl=policy.estimated_kl(stop);g=policy.strength(stop)
        assert (1-g)**2*kl<=.25+1e-12
    assert policy.strength(4)>policy.strength(2)>0
    before=policy.strength(4)
    # Repeated reads do not adapt to current candidates or current reference.
    assert policy.strength(4)==before
    with pytest.raises(ValueError):policy.observe([float('nan')])


def test_source_kernel_kl_proxy_is_nonnegative_and_zero_for_equal_fields():
    x=torch.tensor([[.2,-.1]],dtype=torch.float32)
    states=[x,x]
    same=lambda x,s:.3*x
    assert source_teacher_kernel_kls(states,[1.,.5],.9,same,same)==[0.]
    value=source_teacher_kernel_kls(states,[1.,.5],.9,same,lambda x,s:-.2*x)
    assert len(value)==1 and value[0]>0
