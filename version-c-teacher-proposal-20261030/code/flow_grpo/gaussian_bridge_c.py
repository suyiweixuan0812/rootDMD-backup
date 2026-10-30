"""PDF Version C: calibrated diagonal endpoint Gaussian and shared-noise bridge.

The target is a declared Gaussian approximation, not the true SD3 posterior.
All endpoint calculations are FP32. kappa=1 implies exact uniform IS weights.
"""
import json
import time
from pathlib import Path
import torch
import torch.distributed as dist
from .train_utils import capture_rng, restore_rng

KNOTS=(.025,.1,.25,.45,.65,.8,.95,1.)
_table=None
_knots=None
_refresh_step=None


def set_covariance(knots,table,step):
    global _table,_knots,_refresh_step
    if table.ndim!=2 or table.shape[0]!=len(knots):raise ValueError('sigma/channel covariance shape')
    if not torch.isfinite(table).all() or torch.any(table<=0):raise ValueError('invalid calibrated variance')
    _table=table.detach().float().clone()
    _knots=torch.as_tensor(knots,device=table.device,dtype=torch.float32)
    if torch.any(_knots[1:]<=_knots[:-1]):raise ValueError('unordered calibration knots')
    _refresh_step=int(step)


def covariance_at(sigma):
    if _table is None:raise RuntimeError('Version C calibration is missing')
    if not 0<sigma<=1:raise ValueError('query sigma outside (0,1]')
    if sigma<=float(_knots[0]):
        # Boundary C(0)=0, quadratically scaled from the smallest measured knot.
        return (_table[0]*(sigma/float(_knots[0]))**2).clamp_min(1e-8)
    i=min(int(torch.searchsorted(_knots,torch.tensor(sigma,device=_knots.device))),len(_knots)-1)
    a=(sigma-float(_knots[i-1]))/float(_knots[i]-_knots[i-1])
    return ((1-a)*_table[i-1]+a*_table[i]).clamp_min(1e-8)


@torch.no_grad()
def bridge_estimate(query,sigma,student,num_particles=32,generator=None):
    if query.shape[0]!=1 or num_particles!=32:raise ValueError('experiment requires batch1 and N32')
    mean=query.float()-sigma*student(query,sigma).float()
    cov=covariance_at(sigma).view(1,-1,1,1)
    if cov.shape[1]!=query.shape[1]:raise ValueError('latent channel mismatch')
    z=torch.randn((num_particles,*query.shape[1:]),device=query.device,dtype=torch.float32,generator=generator)
    endpoint=mean+cov.sqrt()*z
    eps=(query.float()-(1-sigma)*endpoint)/sigma
    errors=[]
    for ratio in (1.,.9,.75,.5,0.):
        r=ratio*sigma
        actual=(1-r)*endpoint+r*eps
        expected=ratio*query.float()+(1-ratio)*endpoint
        errors.append((actual-expected).abs().max())
    algebra=float(torch.stack(errors).max())
    if not torch.isfinite(endpoint).all() or algebra>2e-4:raise FloatingPointError('bridge algebra/numerics')
    result=endpoint.mean(0,keepdim=True)
    return result,{'bridge/ess':float(num_particles),'bridge/max_weight':1/num_particles,
                   'bridge/kappa':1.,'bridge/calibration_step':float(_refresh_step),
                   'bridge/variance_mean':float(cov.mean()),'bridge/algebra_error':algebra,
                   'bridge/MC_mean_noise_rms':float((result-mean).square().mean().sqrt()),
                   'bridge/expected_MC_noise_rms':float((cov.mean()/num_particles).sqrt())}


@torch.no_grad()
def refresh_covariance(model,text_encoders,tokenizers,config,sigmas,total_steps,shape,device,dtype,output,step):
    """Fresh independent source/noise pools, isolated from the actual train query.

    Use actual random-stop source generation. Separate calibration and heldout
    prompts/noise; no optimization, no heldout fitting. Pixel pooling is a channel
    uncertainty proxy; pixels are not treated as independent statistical samples.
    """
    from .diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
    from .diffusers_patch.sd3_pipeline_dmd import generate_latents
    started=time.perf_counter()
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    rng=capture_rng();modes=[(m,m.training) for m in model.modules()]
    grads=[(p,p.requires_grad) for p in model.parameters()];adapter=model.active_adapter
    model.set_adapter('generator');model.eval()
    prompts=[p.strip() for p in (Path(config.dataset)/'train.txt').read_text().splitlines() if p.strip()]
    nfit=int(config.bridge.calibration_per_rank);nheld=int(config.bridge.heldout_per_rank)
    sums=torch.zeros(len(KNOTS),shape[1],device=device,dtype=torch.float64)
    held_mse=torch.zeros_like(sums);held_mean=torch.zeros_like(sums);coverage=torch.zeros_like(sums)
    residuals=[]
    try:
        for pool,n in enumerate((nfit,nheld)):
            for j in range(n):
                seed=202610280000+step*10000+pool*1000+rank*20+j
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                # Keep calibration prompts in a fixed separate window from
                # heldout prompts; source/noise draws are new at every refresh.
                index=(len(prompts)-4096+pool*2048+rank*20+j)%len(prompts)
                embeds,pooled=encode_prompt(text_encoders,tokenizers,[prompts[index]],
                    config.train.max_sequence_length,device=device)
                stop=(rank*n+j)%len(sigmas[:-1])+1
                source,_=generate_latents(model,torch.randn(shape,device=device,dtype=embeds.dtype),
                    embeds,pooled,sigmas,total_steps,stop,config.sample.noise_level,
                    mixed_precision_dtype=dtype,retain_grad=False)
                for k,s in enumerate(KNOTS):
                    noise=torch.randn(shape,device=device,dtype=torch.float32)
                    query=((1-s)*source.float()+s*noise).to(embeds.dtype)
                    with torch.autocast(device.type,dtype=dtype,enabled=dtype is not None):
                        v=model(hidden_states=query,timestep=torch.full((shape[0],),s*total_steps,device=device),
                            encoder_hidden_states=embeds,pooled_projections=pooled,return_dict=False)[0]
                    error=source.float()-(query.float()-s*v.float())
                    mse=error.double().square().mean((0,2,3))
                    if pool==0:sums[k]+=mse
                    else:
                        held_mse[k]+=mse;held_mean[k]+=error.double().mean((0,2,3))
                        residuals.append((k,error.detach()))
        if dist.is_initialized():dist.all_reduce(sums)
        fitted=sums/(nfit*world)
        if not torch.isfinite(fitted).all():raise FloatingPointError('nonfinite covariance calibration')
        final=fitted.clamp(1e-8,1e4).float()
        clipped=int(torch.count_nonzero((fitted<1e-8)|(fitted>1e4)))
        for k,error in residuals:
            coverage[k]+=(error.abs()<=1.959964*final[k].sqrt().view(1,-1,1,1)).double().mean((0,2,3))
        if dist.is_initialized():
            for x in (held_mse,held_mean,coverage):dist.all_reduce(x)
        held_mse/=nheld*world;held_mean/=nheld*world;coverage/=nheld*world
        set_covariance(KNOTS,final,step)
        torch.cuda.synchronize()
        seconds=torch.tensor(time.perf_counter()-started,device=device,dtype=torch.float64)
        if dist.is_initialized():dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
        receipt={'outer_step':step,'knots':list(KNOTS),'covariance':final.cpu().tolist(),
            'calibration_source_samples':nfit*world,'heldout_source_samples':nheld*world,
            'heldout_MSE':held_mse.cpu().tolist(),'heldout_mean_residual':held_mean.cpu().tolist(),
            'heldout_95_coverage':coverage.cpu().tolist(),'clipped_channels':clipped,
            'seconds':float(seconds),'target':'diagonal calibrated Gaussian approximation, not exact posterior',
            'heldout_used_for_fitting':False,'training_query_used_for_fitting':False,
            'pixel_pooling_note':'channel spatial averages; no independence assumption for pixels'}
        if rank==0:
            folder=Path(output)/'bridge-calibration';folder.mkdir(parents=True,exist_ok=True)
            (folder/f'step-{step}.json').write_text(json.dumps(receipt,indent=2))
            print('BRIDGE_CALIBRATED '+json.dumps({'step':step,'seconds':float(seconds),
                'mean_coverage':float(coverage.mean()),'min_coverage':float(coverage.min()),
                'max_covariance':float(final.max()),'clipped_channels':clipped}),flush=True)
        return receipt
    finally:
        model.set_adapter(adapter)
        for m,mode in modes:m.training=mode
        for p,req in grads:p.requires_grad_(req)
        restore_rng(rng)

def checkpoint_state():
    return None if _table is None else dict(knots=_knots.detach().cpu(),table=_table.detach().cpu(),step=_refresh_step)
