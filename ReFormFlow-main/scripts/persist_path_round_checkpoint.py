"""Persist an original FP32 inference EMA adapter, verify hashes and finiteness."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-run',required=True)
p.add_argument('--step',type=int,default=1000)
p.add_argument('--name',required=True)
a=p.parse_args()
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]/'logs/path-ratio-round1'
checkpoint=root/'runs'/a.training_run/'checkpoints'/f'checkpoint-{a.step}'
source=checkpoint/'lora/generator'
assert source.is_dir()
parent=root/'archived-checkpoints'
parent.mkdir(exist_ok=True)
dest=parent/a.name
assert not dest.exists(), 'Never overwrite an archived checkpoint'
assert not (parent/(a.name+'.copying')).exists()
files=[x for x in source.iterdir() if x.is_file()]
size=sum(f.stat().st_size for f in files)+ (checkpoint/'config.json').stat().st_size
assert shutil.disk_usage(parent).free>size+20*1024**2, 'Keep at least 20 MiB disk margin'
tensors=load_file(str(source/'adapter_model.safetensors'))
assert len(tensors)==382 and all(torch.isfinite(t).all() for t in tensors.values())
assert sum(float(t.double().square().sum()) for k,t in tensors.items() if 'lora_B' in k)>0
pending=parent/(a.name+'.copying');(pending/'lora/generator').mkdir(parents=True)
for f in files:shutil.copy2(f,pending/'lora/generator'/f.name)
shutil.copy2(checkpoint/'config.json',pending/'config.json')
runtime=root/'runs'/a.training_run/'runtime-numerics.json'
if runtime.exists():shutil.copy2(runtime,pending/'runtime-numerics.json')
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for f in files:assert digest(f)==digest(pending/'lora/generator'/f.name)
pending.rename(dest)
manifest={'training_run':a.training_run,'step':a.step,'source':str(source),'persistent_copy':str(dest/'lora/generator'),
          'all_original_fp32_tensors':True,'tensor_count':len(tensors),'all_finite':True,
          'scope':'Inference EMA generator only; no exact optimizer/RNG resume.',
          'files_sha256':{str(f.relative_to(dest)):digest(f) for f in dest.rglob('*') if f.is_file()}}
(dest/'storage-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
