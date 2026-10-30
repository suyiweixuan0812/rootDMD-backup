"""CPU numerical checks against independent probability-distribution formulas."""
import sys,json,ast,math,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flow_grpo.teacher_endpoint_proposal import teacher_endpoint_proposal,_student_mixture_log_prob,_normal_log_prob
torch.set_num_threads(2);torch.manual_seed(20261030)
B,K,C,H,W=1,4,2,2,2
log_pi=torch.log_softmax(torch.randn(B,K),1);mu=torch.randn(B,K,C,H,W)*.3;v=torch.rand_like(mu)+.2
x=torch.randn(16,C,H,W)
cat=torch.distributions.Categorical(logits=log_pi[0].double())
components=torch.distributions.Independent(torch.distributions.Normal(mu[0].double(),v[0].double().sqrt()),3)
oracle=torch.distributions.MixtureSameFamily(cat,components)
err=float((_student_mixture_log_prob(x,log_pi,mu,v)-oracle.log_prob(x.double())).abs().max())
assert err<2e-6,err
all_equal=[]
for kappa in (0.,.25,.5,.75,1.):
 # When q_ref == q_target all weights must be uniform for every kappa.
 q=torch.randn(1,C,H,W);m=torch.zeros(1,1,C,H,W);var=torch.ones_like(m)*.3
 est,d=teacher_endpoint_proposal(q,.25,torch.zeros(1,1),m,var,m[:,0],kappa,256,teacher_variance=var[:,0])
 assert abs(d['teacher_proposal/ess']-256)<1e-8
 assert abs(d['teacher_proposal/unnormalized_mass_estimate']-1)<1e-8
 all_equal.append(kappa)
# Full SD3-dimensional separated proposal: checks finite weights and records coverage rather than hiding it.
q=torch.zeros(1,16,64,64);m=torch.zeros(1,4,16,64,64);v=torch.ones_like(m)*.1
high=[]
for k in (.25,.5,.75,1.):
 est,d=teacher_endpoint_proposal(q,.6,torch.ones(1,4)*(-math.log(4)),m,v,torch.ones_like(q)*.5,k,32)
 assert torch.isfinite(est).all() and d['teacher_proposal/ess']>=1-1e-6
 assert d['teacher_proposal/max_weight']<=1+1e-10
 high.append(d)
# MC agreement with known Student mixture mean in low dimensions. This is not an SD3 quality test.
q=torch.zeros(1,1,1,1);lp=torch.log(torch.tensor([[.25,.75]]));m=torch.tensor([[[[[ -1.]]],[[[2.]]]]]);v=torch.ones_like(m)*.5
est,d=teacher_endpoint_proposal(q,.5,lp,m,v,torch.zeros_like(q),.5,65536)
assert abs(float(est)-1.25)<.025,float(est)
from flow_grpo.gaussian_bridge_c import set_covariance
from flow_grpo.single_gaussian_teacher_proposal import estimate as single
set_covariance([.025,1.],torch.ones(2,16)*.1,0)
query=torch.randn(1,16,4,4,requires_grad=True)
estimate,diagnostics=single(query,.5,lambda x,s:x*.3,query.detach()*.1)
assert not estimate.requires_grad
assert diagnostics['teacher_proposal/target_components']==1.
assert diagnostics['teacher_proposal/mixture_head_parameters']==0.
base=ast.parse((ROOT/'scripts/train_dmd_sd3.py').read_text())
new=ast.parse((ROOT/'scripts/train_teacher_comparison.py').read_text())
for name in ('dmd_loss','fake_loss','ac_loss','save_ckpt'):
 aa=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name==name)
 bb=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(aa,include_attributes=False)==ast.dump(bb,include_attributes=False),name
result=dict(status='passed',density_vs_independent_oracle_max_abs_error=err,uniform_equal_distributions_kappas=all_equal,
  high_dimensional_coverage=high,low_dimension_known_mean=1.25,low_dimension_snis=float(est),
  baseline_loss_and_save_functions_AST_identical=True,single_gaussian_target_no_head=True,
  scope='CPU formula checks only; not real SD3 or multi-GPU validation')
(ROOT/'checks/cpu-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

