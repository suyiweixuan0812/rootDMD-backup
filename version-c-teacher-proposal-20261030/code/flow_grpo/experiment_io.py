"""Measurement, reproducible evaluation and complete checkpoint evidence."""
from pathlib import Path
import json,hashlib,time,random,gc
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from .train_utils import capture_rng,restore_rng
from .paired_audit import state_digest
from .drawbench_online import score_saved_images,write_step_report
from .diffusers_patch.sd3_pipeline_dmd import generate_latents
from .diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

def rank():return dist.get_rank() if dist.is_initialized() else 0
def world():return dist.get_world_size() if dist.is_initialized() else 1
def sync():
    torch.cuda.synchronize()
    if dist.is_initialized():dist.barrier()
def write_json(path,value):
    if rank()==0:
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
        tmp.replace(path)
def gather(value):
    result=[None]*world()
    if dist.is_initialized():dist.all_gather_object(result,value)
    else:result=[value]
    return result

def save_full(path,step,gen,seconds,parameters,optimizers,schedulers,scalers,ema,config):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    state=dict(outer_step=step,generator_step=gen,training_seconds=seconds,
      parameters={r:{k:v.detach().cpu() for k,v in ps.items()} for r,ps in parameters.items()},
      optimizers={k:v.state_dict() for k,v in optimizers.items()},
      schedulers={k:v.state_dict() for k,v in schedulers.items()},
      scalers={k:v.state_dict() for k,v in scalers.items()},
      ema=None if ema is None else {k:v.detach().cpu() for k,v in ema.items()})
    hashes={k:state_digest(state[k]) for k in ('parameters','optimizers','schedulers','scalers','ema')}
    all_hashes=gather(hashes);assert all(x==hashes for x in all_hashes),'Model or optimizer rank divergence'
    if rank()==0:torch.save(state,path/'training-state.pt')
    torch.save(dict(rank=rank(),world=world(),rng=capture_rng()),path/f'rng-rank-{rank()}.pt')
    if config.experiment.arm=='teacher_c':
        from .gaussian_bridge_c import checkpoint_state
        if rank()==0:torch.save(checkpoint_state(),path/'original-c-covariance.pt')
    sync()
    if rank()==0:
        files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in path.glob('*.pt')}
        write_json(path/'complete.json',dict(status='complete',outer_step=step,generator_step=gen,
          training_seconds=seconds,rank_state_hashes=all_hashes,files=files))
    sync()
    if step==0 and config.experiment.arm=='teacher_c' and config.experiment.matched_time:
        other=Path(config.experiment.reference_dir)/'checkpoints/checkpoint-0/complete.json'
        assert json.loads(other.read_text())['rank_state_hashes']==all_hashes,'Initial states are unpaired'

@torch.no_grad()
def evaluate(pipeline,model,parameters,ema,text_encoders,tokenizers,sigmas,total_steps,config,output,step,gen,seconds,dtype):
    rng=capture_rng();adapter=model.active_adapter
    modes=[(m,m.training) for m in model.modules()]
    flags=[(p,p.requires_grad) for p in model.parameters()]
    stored={n:p.detach().clone() for n,p in parameters['generator'].items()}
    model.set_adapter('generator');model.eval()
    if ema is not None and config.eval.use_ema:
        for n,p in parameters['generator'].items():p.data.copy_(ema[n])
    folder=Path(output)/'eval'/f'step-{step}'/f'rank-{rank()}'
    folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config.eval.seed+rank());np.random.seed(config.eval.seed+rank());random.seed(config.eval.seed+rank())
    prompts=[p.strip() for p in (Path(config.eval.dataset)/'test.txt').read_text().splitlines() if p.strip()][:config.eval.num_prompts]
    records=[(i,dict(prompt=p,metadata={})) for i,p in enumerate(prompts) if i%world()==rank()]
    write=(folder/'prompts.json').write_text(json.dumps([dict(index=i,**r) for i,r in records],ensure_ascii=False,indent=2))
    try:
        for start in range(0,len(records),config.sample.test_batch_size):
            batch=records[start:start+config.sample.test_batch_size]
            emb,pool=encode_prompt(text_encoders,tokenizers,[r['prompt'] for _,r in batch],config.train.max_sequence_length,device=pipeline.device)
            size=config.resolution//pipeline.vae_scale_factor
            initial=torch.randn((len(batch),model.config.in_channels,size,size),device=emb.device,dtype=emb.dtype)
            x,_=generate_latents(model,initial,emb,pool,sigmas,total_steps,len(sigmas)-1,config.sample.noise_level,mixed_precision_dtype=dtype)
            latent=x.float()/pipeline.vae.config.scaling_factor+pipeline.vae.config.shift_factor
            imgs=(pipeline.vae.decode(latent,return_dict=False)[0]/2+.5).clamp(0,1)
            for (i,_),im in zip(batch,imgs):
                Image.fromarray(im.mul(255).round().byte().permute(1,2,0).cpu().numpy()).save(folder/f'{i:06d}.png')
        if config.experiment.score_images:
            scores=score_saved_images(folder,records,emb.device)
            local={k:(sum(v),len(v)) for k,v in scores.items()}
            all_scores=gather(local)
            if rank()==0:
                sums=defaultdict(float);counts=defaultdict(int)
                for local in all_scores:
                    for k,(v,n) in local.items():sums[k]+=v;counts[k]+=n
                values={'eval/'+k:sums[k]/counts[k] for k in sums}
                write_step_report(Path(output),step,config,counts,values,world())
                p=Path(output)/'eval'/f'step-{step}'/'final-metrics.json'
                report=json.loads(p.read_text());report.update(training_seconds=seconds,training_minutes=seconds/60,generator_step=gen)
                p.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        sync()
        if rank()==0:
            paths=sorted((Path(output)/'eval'/f'step-{step}').glob('rank-*/*.png'))
            assert len(paths)==len(prompts)
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            write_json(Path(output)/'eval'/f'step-{step}'/'image-hashes.json',hashes)
            if step==0 and config.experiment.arm=='teacher_c' and config.experiment.matched_time:
                ref=Path(config.experiment.reference_dir)/'eval/step-0/image-hashes.json'
                assert json.loads(ref.read_text())==hashes,'Initial images are unpaired'
    finally:
        for n,p in parameters['generator'].items():p.data.copy_(stored[n])
        model.set_adapter(adapter)
        for m,v in modes:m.training=v
        for p,v in flags:p.requires_grad_(v)
        restore_rng(rng)
        gc.collect();torch.cuda.empty_cache()
    sync()

def reference_targets(config):
    if config.experiment.arm=='dmd' or not config.experiment.matched_time:return []
    p=Path(config.experiment.reference_dir)
    done=json.loads((p/'training-complete.json').read_text())
    assert done['outer_step']==10000 and done['generator_step']==1000
    targets=json.loads((p/'evaluation-times.json').read_text())
    assert len(targets)==11 and targets[0]['training_seconds']==0
    return targets[1:]

