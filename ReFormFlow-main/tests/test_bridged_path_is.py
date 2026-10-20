import torch
from flow_grpo.bridged_path_is import evaluate_bridged_path,bridge_velocity
from flow_grpo.path_ratio_smc import evaluate_teacher_path,cps_mean
from flow_grpo.conditional_path_is import lift_student_prefix


def test_zero_bridge_exactly_recovers_teacher_proposal():
    g=torch.Generator().manual_seed(62111)
    e=torch.randn(200,2,3,generator=g,dtype=torch.float64);j=torch.arange(200)%2
    kwargs=dict(observation=torch.full((1,3),.2),sigma_observation=.4,positive_sigmas=[1.,.5],eta=.9,
                student_velocity=lambda x,s:.4*x,teacher_velocity=lambda x,s:.1*x)
    a=evaluate_teacher_path(e,j,**kwargs);b=evaluate_bridged_path(e,j,bridge_strength=0.,**kwargs)
    torch.testing.assert_close(a.source,b.source,atol=0,rtol=0)
    torch.testing.assert_close(a.log_weight,b.log_weight,atol=0,rtol=0)
    torch.testing.assert_close(b.log_teacher_proposal,torch.zeros_like(b.log_teacher_proposal),atol=0,rtol=0)


def test_student_proposal_limit_has_unit_path_ratio_and_exact_correction():
    g=torch.Generator().manual_seed(62112)
    e=torch.randn(100,2,4,generator=g,dtype=torch.float64)
    r=evaluate_bridged_path(e,torch.ones(100,dtype=torch.long),torch.zeros(1,4),.4,[1.,.5],.9,
        lambda x,s:.4*x,lambda x,s:.1*x,bridge_strength=1.)
    torch.testing.assert_close(r.log_path_ratio,torch.zeros_like(r.log_path_ratio),atol=0,rtol=0)
    torch.testing.assert_close(r.log_student_teacher+r.log_teacher_proposal,r.log_path_ratio,atol=1e-12,rtol=0)


def test_bridge_factorization_and_analytic_forward_posterior():
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(62113)
    e=torch.randn(200000,2,1,generator=g,dtype=torch.float64)
    eta=.9;s,t=1.,.5
    r=evaluate_bridged_path(e,torch.ones(len(e),dtype=torch.long),torch.full((1,1),.25),.4,[s,t],eta,
        lambda x,s:.4*x,lambda x,s:.1*x,bridge_strength=.9)
    torch.testing.assert_close(r.log_student_teacher+r.log_teacher_proposal,r.log_path_ratio,atol=1e-12,rtol=1e-12)
    a=float(cps_mean(torch.ones(1),.4*torch.ones(1),s,t,eta))
    prior_var=.8**2*(a*a+(t*torch.sin(torch.tensor(eta*torch.pi/2)))**2)
    exact=.6*prior_var/(.36*prior_var+.16)*.25
    estimated=(r.log_weight.softmax(0)[:,None]*r.source).sum(0)
    assert abs(float(estimated)-float(exact))<.003


def test_bf16_bridged_reference_replays_and_factors_correctly():
    g=torch.Generator().manual_seed(62114)
    x=torch.randn(1,20,generator=g).bfloat16();student=lambda x,s:.4*x.float();teacher=lambda x,s:.1*x.float()
    mean=cps_mean(x.float(),student(x,1.),1.,.5,.9)
    std=.5*torch.sin(torch.tensor(.9*torch.pi/2))
    nxt=(mean+std*torch.randn(1,20,generator=g)).bfloat16()
    q=lambda x,s:bridge_velocity(student(x,s),teacher(x,s),.9)
    e=lift_student_prefix([x,nxt],[1.,.5],.9,q,generator=g)
    r=evaluate_bridged_path(e,torch.ones(1,dtype=torch.long),torch.zeros(1,20),.4,[1.,.5],.9,
        student,teacher,bridge_strength=.9,state_dtype=torch.bfloat16)
    expected=(nxt.float()-.5*student(nxt,.5)).bfloat16()
    assert torch.equal(r.source,expected)
    torch.testing.assert_close(r.log_student_teacher+r.log_teacher_proposal,r.log_path_ratio,atol=1e-12,rtol=1e-12)


def test_omitting_bridge_proposal_correction_targets_the_wrong_posterior():
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(62115)
    e=torch.randn(120000,2,1,generator=g,dtype=torch.float64)
    r=evaluate_bridged_path(e,torch.ones(len(e),dtype=torch.long),torch.full((1,1),.8),.4,[1.,.5],.9,
        lambda x,s:.4*x,lambda x,s:-.4*x,bridge_strength=.9)
    a=float(cps_mean(torch.ones(1),.4*torch.ones(1),1.,.5,.9))
    var=.8**2*(a*a+(.5*torch.sin(torch.tensor(.9*torch.pi/2)))**2)
    exact=.6*var/(.36*var+.16)*.8
    correct=(r.log_weight.softmax(0)[:,None]*r.source).sum()
    wrong_weight=r.log_student_teacher+r.log_initial_ratio+r.log_stop_ratio+r.log_observation
    wrong=(wrong_weight.softmax(0)[:,None]*r.source).sum()
    assert abs(float(correct-exact))<.004
    assert abs(float(wrong-exact))>.015


def test_continuous_cps_bridge_reduces_conditional_kernel_kl():
    x=torch.linspace(-2,2,100,dtype=torch.float64).reshape(10,10)
    student=.4*x.tanh();teacher=.1*x.sin();gamma=.9
    s,t,eta=.9,.5,.9
    a=cps_mean(x,student,s,t,eta);b=cps_mean(x,teacher,s,t,eta)
    q=cps_mean(x,bridge_velocity(student,teacher,gamma),s,t,eta)
    variance=(t*torch.sin(torch.tensor(eta*torch.pi/2,dtype=torch.float64)))**2
    original=(a-b).square().sum()/2/variance
    bridged=(a-q).square().sum()/2/variance
    torch.testing.assert_close(bridged,(1-gamma)**2*original,atol=1e-14,rtol=1e-12)
