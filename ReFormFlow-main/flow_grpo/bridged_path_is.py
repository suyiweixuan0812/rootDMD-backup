"""Teacher-informed Gaussian bridges, with the exact proposal correction.

Q_gamma is a positive CPS Gaussian with v_Q=v_T+gamma*(v_S-v_T).
The target remains the actual Student prefix posterior. Its weight is
  (M_S/M_T)*(M_T/Q_gamma)*L = (M_S/Q_gamma)*L
at each positive transition. The original S/T ratio and the new correction are
both recorded; the effective log ratio is computed directly to avoid subtracting
large sums. gamma=0 recovers Teacher-only; gamma=1 is a Student-proposal CONTROL.
"""
from dataclasses import dataclass
import math
import torch
from .rounding_cells import unrepresentable_bf16_cells
from .path_ratio_smc import (PathEvaluation,cps_mean,bf16_gaussian_log_mass,
    bf16_gaussian_log_ratio_at,gaussian_log_ratio_at)


@dataclass
class BridgedPathEvaluation(PathEvaluation):
    log_student_teacher:torch.Tensor
    log_teacher_proposal:torch.Tensor


def bridge_velocity(student,teacher,strength):
    if not 0<=strength<=1:raise ValueError('Bridge strength must be in [0,1]')
    if strength==0:return teacher
    if strength==1:return student
    return teacher+strength*(student-teacher)


@torch.no_grad()
def evaluate_bridged_path(innovations,stops,observation,sigma_observation,
                          positive_sigmas,eta,student_velocity,teacher_velocity,
                          bridge_strength=.9,initial_mean=None,target_stop_probs=None,
                          proposal_stop_probs=None,state_dtype=None,restrict_representable=False):
    grid=tuple(float(x) for x in positive_sigmas)
    n,count=innovations.shape[:2]
    if (count!=len(grid) or not count or not 0<eta<=1 or not 0<=bridge_strength<=1
        or any(not 0<s<=1 for s in grid) or any(a<=b for a,b in zip(grid,grid[1:]))):
        raise ValueError('Need positive decreasing nodes, positive CPS variance and valid bridge')
    if stops.shape!=(n,) or torch.any((stops<0)|(stops>=count)):
        raise ValueError('Invalid stopping nodes')
    if not torch.isfinite(innovations).all():raise ValueError('Nonfinite innovations')
    sample=innovations[:,0]
    shift=torch.zeros_like(sample) if initial_mean is None else torch.broadcast_to(initial_mean,sample.shape)
    sample=sample+shift
    if state_dtype is not None:sample=sample.to(state_dtype)
    excluded=(unrepresentable_bf16_cells(sample,shift,1.).flatten(1).any(1)
              if restrict_representable else torch.zeros(n,device=sample.device,dtype=torch.bool))
    quantized=state_dtype==torch.bfloat16
    initial=(bf16_gaussian_log_ratio_at(sample,torch.zeros_like(shift),shift,1.) if quantized else
        (.5*shift.double().square()-sample.double()*shift.double()).flatten(1).sum(1))
    effective=torch.zeros(n,device=sample.device,dtype=torch.float64)
    base=torch.zeros_like(effective);correction=torch.zeros_like(effective)
    reads=[];prefix=[];bases=[];corrections=[]
    for k,s in enumerate(grid):
        x=sample.float() if quantized else sample
        student=student_velocity(sample,s).to(x.dtype)
        reads.append((x-s*student).to(sample.dtype))
        prefix.append(effective);bases.append(base);corrections.append(correction)
        if k==int(stops.max().item()):break
        t=grid[k+1]
        teacher=teacher_velocity(sample,s).to(x.dtype)
        target_mean=cps_mean(x,student,s,t,eta)
        teacher_mean=cps_mean(x,teacher,s,t,eta)
        proposal_mean=(teacher_mean if bridge_strength==0 else target_mean if bridge_strength==1 else
            cps_mean(x,bridge_velocity(student,teacher,bridge_strength),s,t,eta))
        std=t*math.sin(eta*math.pi/2)
        proposed=(proposal_mean+std*innovations[:,k+1]).to(sample.dtype)
        if restrict_representable:
            excluded |= unrepresentable_bf16_cells(proposed,proposal_mean,std).flatten(1).any(1) & (stops>k)
        if quantized:
            a=bf16_gaussian_log_mass(proposed,target_mean,std)
            b=bf16_gaussian_log_mass(proposed,teacher_mean,std)
            q=bf16_gaussian_log_mass(proposed,proposal_mean,std)
            effective=effective+(a-q).flatten(1).sum(1)
            base=base+(a-b).flatten(1).sum(1)
            correction=correction+(b-q).flatten(1).sum(1)
        else:
            effective=effective+gaussian_log_ratio_at(proposed,target_mean,proposal_mean,std)
            base=base+gaussian_log_ratio_at(proposed,target_mean,teacher_mean,std)
            correction=correction+gaussian_log_ratio_at(proposed,teacher_mean,proposal_mean,std)
        sample=proposed
    rows=torch.arange(n,device=sample.device)
    source=torch.stack(reads,1)[rows,stops]
    path=torch.stack(prefix,1)[rows,stops]
    base=torch.stack(bases,1)[rows,stops]
    correction=torch.stack(corrections,1)[rows,stops]
    def probabilities(values):
        v=torch.ones(count,device=sample.device,dtype=torch.float64) if values is None else torch.as_tensor(values,device=sample.device,dtype=torch.float64)
        if v.shape!=(count,) or not torch.isfinite(v).all() or torch.any(v<=0):raise ValueError('Need positive stop masses')
        return v/v.sum()
    p,q=probabilities(target_stop_probs),probabilities(proposal_stop_probs)
    stop_ratio=(p.log()-q.log())[stops]
    sigma=torch.as_tensor(sigma_observation,device=sample.device,dtype=torch.float64)
    if sigma.numel()==1:sigma=sigma.expand(n)
    if sigma.shape!=(n,) or torch.any((sigma<=0)|(sigma>=1)):raise ValueError('Observation sigma must be in (0,1)')
    sigma=sigma.reshape((-1,)+(1,)*(source.ndim-1))
    likelihood=-.5*((observation.double()-(1-sigma)*source.double())/sigma).square().flatten(1).sum(1)
    weight=initial+path+stop_ratio+likelihood
    if not torch.isfinite(weight).all():raise FloatingPointError('Nonfinite bridge weight')
    return BridgedPathEvaluation(source,weight.masked_fill(excluded,-torch.inf),path,initial,stop_ratio,likelihood,base,correction)
