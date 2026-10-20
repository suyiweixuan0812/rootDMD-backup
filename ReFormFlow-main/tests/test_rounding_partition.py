import math
import pytest
import torch
from flow_grpo.rounding_cells import unrepresentable_bf16_cells,UnrepresentableRoundingCell
from flow_grpo.conditional_path_is import lift_student_prefix,conditional_importance_sweep
from flow_grpo.path_ratio_smc import PathEvaluation,cps_mean,evaluate_teacher_path
from flow_grpo.diffusers_patch.sd3_conditional_path_is import conditional_path_mean


def test_inverse_partition_detects_only_numerically_unsafe_finite_cells():
    x=torch.zeros(1,1,dtype=torch.bfloat16)
    assert not unrepresentable_bf16_cells(x,torch.zeros(1,1),.4).any()
    assert unrepresentable_bf16_cells(x,torch.ones(1,1),.4).all()
    with pytest.raises(ValueError):
        unrepresentable_bf16_cells(x,torch.full((1,1),float('nan')),.4)
    states=[torch.tensor([[.2]],dtype=torch.bfloat16),x]
    field=lambda z,s:torch.zeros_like(z)
    with pytest.raises(UnrepresentableRoundingCell):
        lift_student_prefix(states,[1.,.5],.9,field)
    mean,diag=conditional_path_mean(torch.zeros(1,1),torch.tensor([.8]),states,x,field,field,[1.,.5],.9)
    assert torch.equal(mean,x.float())
    assert diag['path_is/identity_due_to_unrepresentable_cell']==1


def test_candidate_partition_is_opt_in_and_masks_the_same_inverse_domain():
    x=torch.tensor([[.2]],dtype=torch.bfloat16)
    mu=cps_mean(x.float(),torch.zeros_like(x).float(),1.,.5,.9)
    std=.5*math.sin(.9*math.pi/2)
    e=torch.stack([x.double(),-mu.double()/std],dim=1)
    stop=torch.ones(1,dtype=torch.long)
    field=lambda z,s:torch.zeros_like(z)
    kwargs=dict(observation=torch.zeros(1,1),sigma_observation=.8,positive_sigmas=[1.,.5],eta=.9,
                student_velocity=field,teacher_velocity=field,state_dtype=torch.bfloat16)
    ordinary=evaluate_teacher_path(e,stop,**kwargs)
    restricted=evaluate_teacher_path(e,stop,restrict_representable=True,**kwargs)
    assert torch.isfinite(ordinary.log_weight).all()
    assert torch.isneginf(restricted.log_weight).all()


def test_identity_stratum_plus_restricted_isir_preserves_global_posterior():
    torch.manual_seed(89)
    ref=.8+torch.randn(32768,1,1,dtype=torch.float64)
    safe=ref[:,0,0]>=.2
    def evaluator(restrict):
        def evaluate(e,j):
            x=e[:,0];ratio=(.8*x-.5*.8**2).sum(1)
            weight=ratio.masked_fill(x[:,0]<.2,-torch.inf) if restrict else ratio
            zero=torch.zeros_like(ratio)
            return PathEvaluation(x,weight,ratio,zero,zero,zero)
        return evaluate
    subset=ref[safe];stops=torch.zeros(len(subset),dtype=torch.long)
    result=conditional_importance_sweep(subset,stops,evaluator(True),auxiliary_scale=.5)
    correct=ref[:,0].clone();correct[safe]=result.mean.double()
    wrong=ref[:,0].clone()
    wrong[safe]=conditional_importance_sweep(subset,stops,evaluator(False),auxiliary_scale=.5).mean.double()
    assert abs(correct.mean()-.8)<.025
    assert abs(correct.mean()-.8)<abs(wrong.mean()-.8)
    assert correct.var()<ref.var()
