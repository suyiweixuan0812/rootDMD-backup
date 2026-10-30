"""Actual state and initial image pairing gates; never generates audit results."""
import hashlib,json,math,struct
from pathlib import Path
import torch
import numpy as np
import torch.distributed as dist

def state_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous();h.update(str((str(t.dtype),list(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((str(v.dtype),list(v.shape))).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda x:str(x)):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

def live_audit(output,step,parameters,optimizers,schedulers,scalers,ema,policy):
    world=dist.get_world_size() if dist.is_initialized() else 1;rank=dist.get_rank() if dist.is_initialized() else 0
    state=dict(parameters=parameters,optimizers={k:v.state_dict() for k,v in optimizers.items()},
        schedulers={k:v.state_dict() for k,v in schedulers.items()},
        scalers={k:v.state_dict() for k,v in scalers.items()},ema=ema,policy=policy)
    hashes={k:state_digest(v) for k,v in state.items()}
    gathered=[dict(rank=rank,hashes=hashes)]
    if dist.is_initialized():
        gathered=[None]*world;dist.all_gather_object(gathered,dict(rank=rank,hashes=hashes))
    assert world in (1,8) and all(r['hashes']==hashes for r in gathered), 'Live state replicas differ'
    assert step in (0,10)
    if rank==0:
        folder=Path(output)/'startup-audit';folder.mkdir(exist_ok=True)
        (folder/f'state-step-{step}.json').write_text(json.dumps(dict(status='passed',step=step,
            world_size=world,actual_gpu=True,ranks=gathered),indent=2))
    if dist.is_initialized():dist.barrier()

def verify_initial_pair(output):
    output=Path(output)
    assert output.name=='version_c'
    other=output.parent/'dmd'
    a=torch.load(other/'checkpoints/checkpoint-0/training-state.pt',map_location='cpu',weights_only=False)
    b=torch.load(output/'checkpoints/checkpoint-0/training-state.pt',map_location='cpu',weights_only=False)
    assert a['outer_step']==b['outer_step']==a['generator_step']==b['generator_step']==0
    keys=('parameters','ema','optimizers','schedulers','scalers')
    hashes={k:state_digest(a[k]) for k in keys}
    assert hashes=={k:state_digest(b[k]) for k in keys}
    rng_hashes={}
    for rank in range(8):
        ra=torch.load(other/f'checkpoints/checkpoint-0/rng-rank-{rank}.pt',map_location='cpu',weights_only=False)
        rb=torch.load(output/f'checkpoints/checkpoint-0/rng-rank-{rank}.pt',map_location='cpu',weights_only=False)
        assert ra['rank']==rb['rank']==rank and ra['world_size']==rb['world_size']==8
        rng_hashes[rank]=state_digest(ra['rng'])
        assert rng_hashes[rank]==state_digest(rb['rng']), ('Initial RNG differs',rank)
    eval_a=other/'eval/step-0';eval_b=output/'eval/step-0'
    rows_a=sorted(json.loads((eval_a/'per-image-scores.json').read_text()),key=lambda r:r['index'])
    rows_b=sorted(json.loads((eval_b/'per-image-scores.json').read_text()),key=lambda r:r['index'])
    assert len(rows_a)==len(rows_b)==1000
    maxdiff=0.
    for i,(ra,rb) in enumerate(zip(rows_a,rows_b)):
        assert ra['index']==rb['index']==i and ra['prompt']==rb['prompt']
        assert ra['scores'].keys()==rb['scores'].keys()
        for k in ra['scores']:
            delta=abs(ra['scores'][k]-rb['scores'][k]);maxdiff=max(maxdiff,delta)
            assert math.isfinite(delta) and delta<=1e-6, (i,k,delta)
    png={}
    for i in range(20):
        pa=list(eval_a.glob(f'rank-*/{i:06d}.png'));pb=list(eval_b.glob(f'rank-*/{i:06d}.png'))
        assert len(pa)==len(pb)==1
        png[i]=hashlib.sha256(pa[0].read_bytes()).hexdigest()
        assert png[i]==hashlib.sha256(pb[0].read_bytes()).hexdigest()
    return dict(status='passed',initial_state_hashes=hashes,rank_rng_hashes=rng_hashes,images=1000,max_raw_score_difference=maxdiff,
        score_absolute_tolerance=1e-6,twenty_png_sha256=png,scientific_scope='Fresh initial pairing, not metric improvement')
