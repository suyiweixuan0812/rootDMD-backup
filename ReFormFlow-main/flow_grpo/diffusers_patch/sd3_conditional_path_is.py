"""SD3 adapters for posterior-stationary conditional path importance sampling."""
import math
import torch

from flow_grpo.conditional_path_is import lift_student_prefix,conditional_importance_sweep
from flow_grpo.path_ratio_smc import evaluate_teacher_path
from flow_grpo.rounding_cells import UnrepresentableRoundingCell,identity_partition_result
from .solver import cps_step


def make_fields(transformer,embeds,pooled,negative,negative_pooled,total_steps,
                proposal_cfg=1.,mixed_precision_dtype=torch.bfloat16,micro_batch=1):
    def predict(sample,s,teacher=False):
        if sample.shape[0]>micro_batch:
            return torch.cat([predict(part,s,teacher) for part in sample.split(micro_batch)])
        n=sample.shape[0]
        cond,pool=embeds.expand(n,-1,-1),pooled.expand(n,-1)
        guided=teacher and proposal_cfg>1
        if guided:
            hidden=torch.cat([sample,sample])
            cond=torch.cat([negative.expand(n,-1,-1),cond])
            pool=torch.cat([negative_pooled.expand(n,-1),pool])
        else:hidden=sample
        context=transformer.disable_adapter() if teacher else torch.autocast(sample.device.type,enabled=False)
        with context,torch.autocast(sample.device.type,dtype=mixed_precision_dtype,
                                  enabled=mixed_precision_dtype is not None):
            result=transformer(hidden_states=hidden,
                timestep=torch.full((hidden.shape[0],),s*total_steps,device=sample.device),
                encoder_hidden_states=cond,pooled_projections=pool,return_dict=False)[0]
        if guided:
            uncond,cond=result.chunk(2)
            result=uncond+proposal_cfg*(cond-uncond)
        return result
    return lambda x,s:predict(x,s,False),lambda x,s:predict(x,s,True)


@torch.no_grad()
def sample_prefix(shape,student,sigmas,stop,eta,device,dtype):
    x=torch.randn(shape,device=device,dtype=dtype)
    states=[]
    for k in range(stop):
        states.append(x)
        s,t=float(sigmas[k]),float(sigmas[k+1])
        velocity=student(x,s)
        next_sample=cps_step(x,velocity,s,t,eta)
        if k==stop-1:
            return states,(x.float()-s*velocity.float()).to(x.dtype)
        x=next_sample
    raise ValueError('Invalid stop')


@torch.no_grad()
def conditional_path_mean(observation,sigma,states,source,student,teacher,positive_sigmas,
                          eta,num_particles=8,num_sweeps=2,base_scale=.15,block_size=16):
    if source.shape[0]!=1:raise ValueError('Comparison entry requires batch1')
    sigmas=[float(s) for s in positive_sigmas]
    try:
        reference=lift_student_prefix(states,sigmas,eta,teacher)
    except UnrepresentableRoundingCell:
        replay=(states[-1].float()-sigmas[len(states)-1]*student(states[-1],sigmas[len(states)-1]).float()).to(source.dtype)
        if not torch.equal(replay,source):raise RuntimeError('Identity-partition source readout mismatch')
        return identity_partition_result(source,float(sigma.item()),len(states))
    stops=torch.tensor([len(states)-1],device=source.device)
    def evaluate(e,j):
        return evaluate_teacher_path(e,j,observation,sigma.flatten(),sigmas,eta,
            student,teacher,state_dtype=source.dtype,restrict_representable=True)
    initial=evaluate(reference,stops)
    if not torch.equal(initial.source,source):
        raise RuntimeError(f'Posterior reference replay mismatch: {float((initial.source-source).abs().max())}')
    s=float(sigma.item())
    scale=min(.5,max(1e-4,base_scale*s/((1-s)*math.sqrt(block_size))))
    estimates,diagnostics=[],[]
    for _ in range(num_sweeps):
        result=conditional_importance_sweep(reference,stops,evaluate,num_particles=num_particles,
            auxiliary_scale=scale,block_size=block_size)
        reference,stops=result.innovations,result.stops
        estimates.append(result.mean)
        diagnostics.append(result.diagnostics)
    estimate=torch.stack(estimates).mean(0)
    metrics={k.replace('path_is_','path_is/',1):sum(r[k] for r in diagnostics)/len(diagnostics)
             for k in diagnostics[0]}
    metrics.update({'path_is/sigma':s,'path_is/stop':len(states),
        'path_is/frozen_update':float(all(r['path_is_actual_move_rate']==0 for r in diagnostics)),
        'path_is/reference_replay_exact':1.,
        'path_is/identity_due_to_unrepresentable_cell':0.,'path_is/candidate_pool_evaluated':1.,
        'path_is/total_mean_displacement_rms':float((estimate-source.float()).square().mean().sqrt())})
    return estimate,metrics
