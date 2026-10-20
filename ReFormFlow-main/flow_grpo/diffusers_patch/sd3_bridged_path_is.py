"""Same conditional i-SIR target, with an explicitly corrected bridge proposal."""
import math
import torch
from flow_grpo.bridged_path_is import evaluate_bridged_path,bridge_velocity
from flow_grpo.conditional_path_is import lift_student_prefix,conditional_importance_sweep
from flow_grpo.rounding_cells import UnrepresentableRoundingCell,identity_partition_result
from .sd3_conditional_path_is import make_fields,sample_prefix


@torch.no_grad()
def conditional_path_mean(observation,sigma,states,source,student,teacher,positive_sigmas,
                          eta,num_particles=8,num_sweeps=2,base_scale=.05,block_size=16,
                          bridge_strength=.9,high_noise_block_size=4096,high_noise_threshold=.9,
                          max_innovation_radius=.5,long_bridge_strength=.99,long_bridge_min_stop=3):
    if source.shape[0]!=1:raise ValueError('Bridge comparison requires batch1')
    grid=[float(s) for s in positive_sigmas]
    bridge_strength=long_bridge_strength if len(states)>=long_bridge_min_stop else bridge_strength
    def proposal(x,s):
        return bridge_velocity(student(x,s).float(),teacher(x,s).float(),bridge_strength)
    try:
        reference=lift_student_prefix(states,grid,eta,proposal)
    except UnrepresentableRoundingCell:
        replay=(states[-1].float()-grid[len(states)-1]*student(states[-1],grid[len(states)-1]).float()).to(source.dtype)
        if not torch.equal(replay,source):raise RuntimeError('Bridge identity-partition readout mismatch')
        mean,metrics=identity_partition_result(source,float(sigma.item()),len(states))
        metrics.update({'path_is/bridge_strength':bridge_strength,
            'path_is/base_student_teacher_span':0.,'path_is/teacher_proposal_correction_span':0.,
            'path_is/innovation_radius_limit':float(max_innovation_radius or 0)})
        return mean,metrics
    stops=torch.tensor([len(states)-1],device=source.device)
    last=[]
    def evaluate(e,j):
        value=evaluate_bridged_path(e,j,observation,sigma.flatten(),grid,eta,student,teacher,
            bridge_strength=bridge_strength,state_dtype=source.dtype,restrict_representable=True)
        last[:]=[value]
        return value
    initial=evaluate(reference,stops)
    if not torch.equal(initial.source,source):raise RuntimeError('Bridge reference replay mismatch')
    s=float(sigma.item())
    active_block=high_noise_block_size if s>=high_noise_threshold else block_size
    if not 1<=active_block<=source[0].numel():raise ValueError('Invalid bridge coordinate block')
    radius=base_scale*s/(1-s)
    if max_innovation_radius is not None:
        if max_innovation_radius<=0:raise ValueError("Innovation radius must be positive")
        radius=min(radius,max_innovation_radius)
    scale=min(.5,max(1e-4,radius/math.sqrt(active_block)))
    estimates=[];diagnostics=[]
    for _ in range(num_sweeps):
        result=conditional_importance_sweep(reference,stops,evaluate,num_particles=num_particles,
            auxiliary_scale=scale,block_size=active_block)
        value=last[0]
        weights=value.log_weight.softmax(0)
        sources=value.source.flatten(1).double()
        variance=(weights[:,None]*(sources-result.mean.flatten(1).double()).square()).sum(0).mean()
        extra={'bridge_strength':bridge_strength,
            'innovation_radius_limit':float(max_innovation_radius or 0),
            'base_student_teacher_span':float(value.log_student_teacher.max()-value.log_student_teacher.min()),
            'teacher_proposal_correction_span':float(value.log_teacher_proposal.max()-value.log_teacher_proposal.min()),
            'rb_removed_variance_trace_per_coordinate':float(variance)}
        result.diagnostics.update({'path_is_'+k:v for k,v in extra.items()})
        reference,stops=result.innovations,result.stops
        estimates.append(result.mean);diagnostics.append(result.diagnostics)
    mean=torch.stack(estimates).mean(0)
    metrics={k.replace('path_is_','path_is/',1):sum(d[k] for d in diagnostics)/len(diagnostics)
             for k in diagnostics[0]}
    metrics.update({'path_is/sigma':s,'path_is/stop':len(states),'path_is/reference_replay_exact':1.,
        'path_is/identity_due_to_unrepresentable_cell':0.,'path_is/candidate_pool_evaluated':1.,
        'path_is/frozen_update':float(all(d['path_is_actual_move_rate']==0 for d in diagnostics)),
        'path_is/total_mean_displacement_rms':float((mean-source.float()).square().mean().sqrt())})
    return mean,metrics
