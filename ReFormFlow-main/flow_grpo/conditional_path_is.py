"""Auxiliary conditional importance resampling with literal path-ratio weights.

The reference is a posterior draw; it is NOT an arbitrary initial guess. A
Gaussian auxiliary center gives local Teacher-innovation proposals while the
original Student/Teacher path ratio and observation likelihood remain in the
weights. Invariance follows from conditional i-SIR on the augmented target.
"""
from dataclasses import dataclass
import math
import torch

from .path_ratio_smc import cps_mean


def _truncated_normal(lower,upper,generator,max_rounds=128):
    if torch.any(lower>=upper) or not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError('Need finite nonempty rounding cells')
    mode=torch.maximum(lower,torch.minimum(torch.zeros_like(lower),upper))
    sample=torch.empty_like(lower)
    pending=torch.ones_like(lower,dtype=torch.bool)
    for _ in range(max_rounds):
        shape=(int(pending.sum()),)
        if shape[0]==0:return sample
        a,b,m=lower[pending],upper[pending],mode[pending]
        proposal=a+(b-a)*torch.rand(shape,device=a.device,dtype=a.dtype,generator=generator)
        log_uniform=torch.rand(shape,device=a.device,dtype=a.dtype,generator=generator).log()
        take=log_uniform<-.5*(proposal.square()-m.square())
        values=sample[pending]
        values[take]=proposal[take]
        sample[pending]=values
        mask=pending.clone()
        mask[pending]=~take
        pending=mask
    raise RuntimeError('Truncated-Gaussian lift budget exhausted')


@torch.no_grad()
def lift_student_prefix(states,positive_sigmas,eta,teacher_velocity,initial_mean=None,generator=None):
    """Lift an actual Student prefix to its conditional Teacher innovations.

    BF16 rounding makes the map many-to-one. Sample within Teacher Gaussian
    rounding cells, rather than setting every innovation to the cell midpoint.
    Unused suffix innovations keep their independent reference Gaussian law.
    This prototype implements batch1, matching the comparison trainer.
    """
    if not states or states[0].shape[0]!=1 or len(states)>len(positive_sigmas):
        raise ValueError('Need a batch1 actual prefix')
    count=len(positive_sigmas)
    innovations=torch.randn((1,count)+states[0].shape[1:],device=states[0].device,
        dtype=torch.float32,generator=generator)
    def inverse(value,mean,std):
        if value.dtype!=torch.bfloat16:
            return ((value-mean)/std).float()
        lo=.5*(value.double()+torch.nextafter(value,torch.full_like(value,-float('inf'))).double())
        hi=.5*(value.double()+torch.nextafter(value,torch.full_like(value,float('inf'))).double())
        lower=(lo-mean.double())/std
        upper=(hi-mean.double())/std
        out=_truncated_normal(lower,upper,generator).float()
        # Exclude cells shifted by the actual FP32 multiply/add roundoff.
        for _ in range(16):
            mismatch=(mean.float()+std*out).to(value.dtype)!=value
            if not mismatch.any():return out
            out[mismatch]=_truncated_normal(lower[mismatch],upper[mismatch],generator).float()
        raise RuntimeError('Lift did not reproduce the BF16 prefix exactly')
    shift=torch.zeros_like(states[0],dtype=torch.float32) if initial_mean is None else initial_mean.float()
    innovations[:,0]=inverse(states[0],shift,1.)
    for k in range(len(states)-1):
        s,t=float(positive_sigmas[k]),float(positive_sigmas[k+1])
        velocity=teacher_velocity(states[k],s).float()
        mean=cps_mean(states[k].float(),velocity,s,t,eta)
        innovations[:,k+1]=inverse(states[k+1],mean,t*math.sin(eta*math.pi/2))
    return innovations


@dataclass
class ConditionalISResult:
    innovations:torch.Tensor
    stops:torch.Tensor
    mean:torch.Tensor
    diagnostics:dict


@torch.no_grad()
def conditional_importance_sweep(reference,stops,evaluate,*,num_particles=8,
                                 auxiliary_scale=.003,generator=None):
    """Local conditional i-SIR; keep J fixed within each posterior-invariant sweep.

    Let phi be the standard Gaussian Teacher-innovation reference. With
    rho=sqrt(1-delta^2), C|E~N(rho E,delta^2 I) and E'|C~N(rho C,delta^2 I).
    phi(E) q(C|E)=phi(C) q(E|C), so augmented-target/proposal weights are the
    SAME original w(E,J). Include the current posterior reference among N paths.
    Keeping J fixed is invariant for the joint posterior, but does not mix J.
    """
    if num_particles<2 or not 0<auxiliary_scale<1:
        raise ValueError('Need at least two particles and a scale in (0,1)')
    batch=reference.shape[0]
    delta=auxiliary_scale
    rho=math.sqrt(1-delta**2)
    center=rho*reference+delta*torch.randn(reference.shape,device=reference.device,
        dtype=reference.dtype,generator=generator)
    candidates=rho*center[:,None]+delta*torch.randn((batch,num_particles-1)+reference.shape[1:],
        device=reference.device,dtype=reference.dtype,generator=generator)
    particles=torch.cat([reference[:,None],candidates],dim=1)
    particle_stops=stops[:,None].expand(batch,num_particles)
    evaluated=evaluate(particles.flatten(0,1),particle_stops.flatten())
    weights=evaluated.log_weight.reshape(batch,num_particles).softmax(1)
    sources=evaluated.source.reshape((batch,num_particles)+evaluated.source.shape[1:])
    broadcast=weights.reshape((batch,num_particles)+(1,)*(sources.ndim-2))
    mean=(broadcast*sources.double()).sum(1)
    chosen=torch.multinomial(weights,1,generator=generator).flatten()
    rows=torch.arange(batch,device=reference.device)
    selected=sources[rows,chosen]
    difference=(selected.float()-sources[:,0].float()).flatten(1)
    displacement=(mean-sources[:,0].double()).flatten(1)
    return ConditionalISResult(particles[rows,chosen],stops.clone(),mean.float(),dict(
        path_is_ess=float(weights.square().sum(1).reciprocal().mean()),
        path_is_max_weight=float(weights.max(1).values.mean()),
        path_is_reference_probability=float(weights[:,0].mean()),
        path_is_actual_move_rate=float((difference.square().sum(1)>0).float().mean()),
        path_is_move_rms=float(difference.square().mean(1).sqrt().mean()),
        path_is_mean_displacement_rms=float(displacement.square().mean(1).sqrt().mean()),
        path_is_log_weight_range=float((evaluated.log_weight.reshape(batch,num_particles).max(1).values
            -evaluated.log_weight.reshape(batch,num_particles).min(1).values).mean()),
        path_is_auxiliary_scale=delta,
        path_is_path_ratio_span=float((evaluated.log_path_ratio.reshape(batch,num_particles).max(1).values
            -evaluated.log_path_ratio.reshape(batch,num_particles).min(1).values).mean()),
        path_is_observation_span=float((evaluated.log_observation.reshape(batch,num_particles).max(1).values
            -evaluated.log_observation.reshape(batch,num_particles).min(1).values).mean()),
    ))
