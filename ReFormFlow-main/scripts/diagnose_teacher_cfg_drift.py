"""Frozen-field diagnostic only; never tune candidate by new test scores."""
import argparse,json,math,hashlib
from pathlib import Path
import torch
from diffusers import StableDiffusion3Pipeline
from flow_grpo.checkpoint_loading import load_exact_fp32_lora
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.diffusers_patch.sd3_conditional_path_is import make_fields,sample_prefix
from flow_grpo.path_ratio_smc import cps_mean,bf16_gaussian_log_ratio_at
p=argparse.ArgumentParser();p.add_argument('--adapter',required=True);p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-sequence-length',type=int,default=128);a=p.parse_args()
root=Path(__file__).resolve().parents[1];torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
pipe=StableDiffusion3Pipeline.from_pretrained(str(root/'models/sd3-medium'),torch_dtype=torch.bfloat16,local_files_only=True)
encoders=[pipe.text_encoder,pipe.text_encoder_2,pipe.text_encoder_3];tokenizers=[pipe.tokenizer,pipe.tokenizer_2,pipe.tokenizer_3]
for encoder in encoders:encoder.requires_grad_(False).eval().to('cuda')
prompts=(root/'dataset/drawbench/test.txt').read_text().splitlines()[:2]
with torch.no_grad():
 embeds,pooled=encode_prompt(encoders,tokenizers,prompts,a.max_sequence_length,device='cuda')
 negative,npool=encode_prompt(encoders,tokenizers,[''],a.max_sequence_length,device='cuda')
for encoder in encoders:encoder.to('cpu')
torch.cuda.empty_cache()
transformer,audit=load_exact_fp32_lora(pipe.transformer,a.adapter,adapter_name='student');transformer.to('cuda').eval()
grid=torch.tensor([1.,.9,.75,.5,0.],device='cuda');shape=(1,transformer.config.in_channels,512//pipe.vae_scale_factor,512//pipe.vae_scale_factor)
rows=[]
with torch.no_grad():
 for index,prompt in enumerate(prompts):
  fields={cfg:make_fields(transformer,embeds[index:index+1],pooled[index:index+1],negative,npool,1000,proposal_cfg=cfg,micro_batch=1) for cfg in [1.,4.5]}
  student=fields[1.][0]
  for seed in [81021,81022,81023,81024]:
   torch.manual_seed(seed)
   states,_=sample_prefix(shape,student,grid,4,.9,'cuda',torch.bfloat16)
   for k in range(3):
    s,t=float(grid[k]),float(grid[k+1]);std=t*math.sin(.9*math.pi/2)
    target=cps_mean(states[k].float(),student(states[k],s).float(),s,t,.9)
    for cfg in [1.,4.5]:
     mean=cps_mean(states[k].float(),fields[cfg][1](states[k],s).float(),s,t,.9)
     rows.append(dict(prompt_index=index,seed=seed,step=k+1,proposal_cfg=cfg,
       gaussian_conditional_KL=float(((target.double()-mean.double())/std).square().sum()/2),
       actual_source_sample_log_mass_ratio=float(bf16_gaussian_log_ratio_at(states[k+1],target,mean,std))))
summary={}
for cfg in [1.,4.5]:
 selected=[r for r in rows if r['proposal_cfg']==cfg]
 summary[str(cfg)]={'mean_sum_conditional_Gaussian_KL_per_path':sum(r['gaussian_conditional_KL'] for r in selected)/8,
  'mean_source_path_log_mass_ratio':sum(r['actual_source_sample_log_mass_ratio'] for r in selected)/8}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(label=a.label,adapter_loading_audit=audit,prompts=prompts,seeds=[81021,81022,81023,81024],source_stop=4,eta=.9,resolution=512,max_sequence_length=a.max_sequence_length,summary=summary,rows=rows,scope='Exploratory proposal-field mismatch on8fixed Student source paths. Conditional Gaussian KL is a per-parent geometric proxy; not posterior ESS, image quality or a new trained candidate. Teacher CFG changes only in this diagnostic.'),indent=2))
print(a.label,summary,flush=True)
