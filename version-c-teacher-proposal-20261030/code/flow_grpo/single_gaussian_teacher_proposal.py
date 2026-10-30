"""Original Version C single-Gaussian target, with Teacher endpoint proposal only.
No mixture head. No additional Teacher calibration. The target covariance is
the original Version C covariance; q_ref uses that same covariance as a heuristic.
"""
import torch
from .gaussian_bridge_c import covariance_at
from .teacher_endpoint_proposal import teacher_endpoint_proposal

@torch.no_grad()
def estimate(query,sigma,student,teacher_clean,kappa=.5,particles=32):
    mean=query.float()-sigma*student(query,sigma).float()
    covariance=covariance_at(sigma).view(1,-1,1,1).expand_as(mean)
    log_pi=torch.zeros((1,1),device=query.device,dtype=torch.float32)
    result,diagnostics=teacher_endpoint_proposal(
        query,sigma,log_pi,mean[:,None],covariance[:,None],teacher_clean.float(),
        kappa=kappa,particles=particles,teacher_variance=covariance)
    diagnostics['teacher_proposal/target_components']=1.
    diagnostics['teacher_proposal/additional_teacher_calibration']=0.
    diagnostics['teacher_proposal/mixture_head_parameters']=0.
    return result,diagnostics

