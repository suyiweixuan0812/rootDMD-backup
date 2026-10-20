"""GPU diagnostics of literal CPS path ratios; no training or test-score claims."""
import argparse
from functools import partial
import json
import math
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from flow_grpo.checkpoint_loading import load_exact_fp32_lora

from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.diffusers_patch.sd3_source_trace import replay_source
from flow_grpo.posterior_mh import SourceState
from flow_grpo.path_ratio_smc import evaluate_teacher_path,effective_sample_size,annealed_path_smc,TemperingBudgetExceeded
from flow_grpo.conditional_path_is import lift_student_prefix,conditional_importance_sweep
from flow_grpo.bridged_path_is import evaluate_bridged_path,bridge_velocity
from flow_grpo.diffusers_patch.solver import cps_step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--model',default=str(root/'models/sd3-medium'))
    parser.add_argument('--adapter',default=str(root/'logs/dmd/sd3-medium-pickscore-no-reward-train/checkpoints/checkpoint-1000/lora/generator'))
    parser.add_argument('--particles',type=int,default=8)
    parser.add_argument('--prompts',type=int,default=2)
    parser.add_argument('--smc-stages',type=int,default=16)
    parser.add_argument('--local',action='store_true')
    parser.add_argument('--micro-batch',type=int,default=8)
    parser.add_argument('--block-size',type=int,default=0)
    parser.add_argument('--block-sizes',type=int,nargs='+',default=[16,4096,65536])
    parser.add_argument('--max-radius',type=float,default=0.)
    parser.add_argument('--record-sigma-min',type=float,default=0.)
    parser.add_argument('--bridge-strength',type=float,default=.9)
    parser.add_argument('--query-sigmas',type=float,nargs='+',default=[.75,.9,.95,.98])
    parser.add_argument('--proposal-cfgs',type=float,nargs='+',default=[1.])
    parser.add_argument('--source-stops',type=int,nargs='+')
    parser.add_argument('--local-scales',type=float,nargs='+',default=[.0003,.001,.003,.01])
    parser.add_argument('--output',type=Path,default=root/'logs/path-ratio-prototype/sd3-diagnostics')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(81021)
    args.output.mkdir(parents=True,exist_ok=True)
    print('Loading SD3-Medium for real path-density diagnostics',flush=True)
    pipe=StableDiffusion3Pipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True)
    for encoder in [pipe.text_encoder,pipe.text_encoder_2,pipe.text_encoder_3]:
        encoder.requires_grad_(False).eval().to('cuda')
    prompts=(root/'dataset/drawbench/test.txt').read_text().splitlines()[:args.prompts]
    with torch.no_grad():
        embeds,pooled=encode_prompt([pipe.text_encoder,pipe.text_encoder_2,pipe.text_encoder_3],
            [pipe.tokenizer,pipe.tokenizer_2,pipe.tokenizer_3],prompts,128,device='cuda')
        negative,negative_pooled=encode_prompt([pipe.text_encoder,pipe.text_encoder_2,pipe.text_encoder_3],
            [pipe.tokenizer,pipe.tokenizer_2,pipe.tokenizer_3],[''],128,device='cuda')
    for encoder in [pipe.text_encoder,pipe.text_encoder_2,pipe.text_encoder_3]:encoder.to('cpu')
    torch.cuda.empty_cache()
    transformer,adapter_audit=load_exact_fp32_lora(pipe.transformer,args.adapter,adapter_name='student')
    (args.output/'adapter-loading-audit.json').write_text(json.dumps(adapter_audit,indent=2))
    transformer.requires_grad_(False).eval().to('cuda')
    total=pipe.scheduler.config.num_train_timesteps
    shift=pipe.scheduler.config.shift
    sigmas=[round(shift*(step/total)/(1+(shift-1)*(step/total)),6) for step in [1000,750,500,250]]
    tensor_sigmas=torch.tensor(sigmas+[0.],device='cuda')
    shape=(1,transformer.config.in_channels,512//pipe.vae_scale_factor,512//pipe.vae_scale_factor)
    records=[]
    @torch.no_grad()
    def predict(sample,s,teacher=False,cfg=1.):
        n=sample.shape[0]
        if n>args.micro_batch:
            return torch.cat([predict(part,s,teacher,cfg) for part in sample.split(args.micro_batch)])
        cond=selected_embeds.expand(n,-1,-1)
        pool=selected_pooled.expand(n,-1)
        if teacher and cfg>1:
            hidden=torch.cat([sample,sample])
            cond=torch.cat([negative.expand(n,-1,-1),cond])
            pool=torch.cat([negative_pooled.expand(n,-1),pool])
        else:hidden=sample
        context=transformer.disable_adapter() if teacher else torch.autocast('cuda',enabled=False)
        with context,torch.autocast('cuda',dtype=torch.bfloat16):
            velocity=transformer(hidden_states=hidden,timestep=torch.full((hidden.shape[0],),s*total,device='cuda'),
                encoder_hidden_states=cond,pooled_projections=pool,return_dict=False)[0]
        if teacher and cfg>1:
            uncond,cond=velocity.chunk(2)
            velocity=uncond+cfg*(cond-uncond)
        return velocity
    for index,prompt in enumerate(prompts):
        selected_embeds,selected_pooled=embeds[index:index+1],pooled[index:index+1]
        if index==0:
            audit_g=torch.Generator(device='cuda').manual_seed(88731)
            audit_x=torch.randn(shape,device='cuda',dtype=torch.bfloat16,generator=audit_g)
            parameters={n:p for n,p in transformer.named_parameters() if 'lora_' in n}
            stored={n:p.detach().clone() for n,p in parameters.items()}
            field_rows=[]
            for audit_sigma in sigmas:
                exact_field=predict(audit_x,audit_sigma)
                for n,p in parameters.items():p.data=stored[n].to(torch.bfloat16)
                rounded_field=predict(audit_x,audit_sigma)
                for n,p in parameters.items():p.data=stored[n]
                field_rows.append({'sigma':audit_sigma,'fields_bit_exact':torch.equal(exact_field,rounded_field),
                    'max_field_difference':float((exact_field.float()-rounded_field.float()).abs().max())})
            adapter_audit['BF16_autocast_field_rounding_comparison']=field_rows
            (args.output/'adapter-loading-audit.json').write_text(json.dumps(adapter_audit,indent=2))
            print('Adapter exact-load and effective-field audit: '+json.dumps(adapter_audit),flush=True)

        state=SourceState(torch.randn(shape,device='cuda'),torch.randn((3,)+shape,device='cuda'),
            torch.randint(1,5,(1,),device='cuda'))
        if args.source_stops:
            state=SourceState(state.z,state.noises,torch.tensor([args.source_stops[index%len(args.source_stops)]],device='cuda'))
        source=replay_source(transformer,state,selected_embeds,selected_pooled,tensor_sigmas,total,.9,torch.bfloat16)
        for sigma in args.query_sigmas:
            observation=(1-sigma)*source.float()+sigma*torch.randn_like(source,dtype=torch.float32)
            if sigma<args.record_sigma_min:continue
            for cfg in args.proposal_cfgs:
                generator=torch.Generator(device='cuda').manual_seed(91200+index)
                innovations=torch.randn((args.particles,4)+shape[1:],device='cuda',generator=generator)
                stops=torch.randint(0,4,(args.particles,),device='cuda',generator=generator)
                last_evaluation=[]
                def evaluate(e,j):
                    value=evaluate_bridged_path(e,j,observation,sigma,sigmas,.9,
                        partial(predict,teacher=False),partial(predict,teacher=True,cfg=cfg),
                        bridge_strength=args.bridge_strength,state_dtype=torch.bfloat16)
                    last_evaluation[:]=[value]
                    return value
                if args.local:
                    prefix=[]
                    x=state.z.to(selected_embeds.dtype)
                    for k in range(int(state.stop.item())):
                        prefix.append(x)
                        if k+1<int(state.stop.item()):
                            x=cps_step(x,predict(x,sigmas[k]),sigmas[k],sigmas[k+1],.9,noise=state.noises[k])
                    proposal=lambda x,s:bridge_velocity(predict(x,s,False).float(),predict(x,s,True,cfg).float(),args.bridge_strength)
                    lifted=lift_student_prefix(prefix,sigmas,.9,proposal,generator=generator)
                    j=torch.tensor([len(prefix)-1],device='cuda')
                    replay=evaluate(lifted,j)
                    error=float((replay.source.float()-source.float()).abs().max())
                    if error!=0:raise RuntimeError(f'Reference source replay mismatch: {error}')
                    for active_block in args.block_sizes:
                        for base_scale in args.local_scales:
                            dimension=source[0].numel()
                            denominator=math.sqrt(active_block or len(prefix)*dimension)
                            radius=base_scale*sigma/(1-sigma)
                            if args.max_radius>0:radius=min(radius,args.max_radius)
                            scale=min(.5,max(.00001,radius/denominator))
                            local_g=torch.Generator(device='cuda').manual_seed(93400+index*100+round(sigma*100))
                            started=time.monotonic()
                            result=conditional_importance_sweep(lifted,j,evaluate,num_particles=args.particles,
                                auxiliary_scale=scale,block_size=active_block or None,generator=local_g)
                            ev=last_evaluation[0]
                            weights=ev.log_weight.softmax(0)
                            values=ev.source.flatten(1).double()
                            weighted_variance=(weights[:,None]*(values-result.mean.flatten(1).double()).square()).sum(0).mean()
                            unique=[]
                            for value in ev.source:
                                if not any(torch.equal(value,old) for old in unique):unique.append(value)
                            torch.cuda.synchronize()
                            row=dict(prompt_index=index,prompt=prompt,sigma=sigma,proposal_cfg=cfg,
                                particles=args.particles,base_scale=base_scale,source_stop=len(prefix),
                                active_block_size=active_block,unique_candidate_sources=len(unique),
                                bridge_strength=args.bridge_strength,max_radius=args.max_radius,
                                base_student_teacher_span=float(ev.log_student_teacher.max()-ev.log_student_teacher.min()),
                                teacher_proposal_correction_span=float(ev.log_teacher_proposal.max()-ev.log_teacher_proposal.min()),
                                log_ratio_factorization_max_error=float((ev.log_path_ratio-ev.log_student_teacher-ev.log_teacher_proposal).abs().max()),
                                rb_removed_variance_trace_per_coordinate=float(weighted_variance),
                                exact_reference_replay=True,elapsed_seconds=time.monotonic()-started,
                                sigma_grid=sigmas,**result.diagnostics)
                            records.append(row)
                            (args.output/'diagnostics.json').write_text(json.dumps(dict(
                                scope='Sampling-only development diagnostic: fixed stopping nodes, weight ESS versus actual diversity and RB variance; no image-quality claim',
                                rows=records,configuration=vars(args)),indent=2,default=str,ensure_ascii=False))
                            print(json.dumps(row,ensure_ascii=False),flush=True)
                    continue
                started=time.monotonic()
                result=evaluate(innovations,stops)
                torch.cuda.synchronize()
                row=dict(prompt_index=index,prompt=prompt,sigma=sigma,proposal_cfg=cfg,
                    particles=args.particles,ess=effective_sample_size(result.log_weight),
                    max_weight=float(result.log_weight.softmax(0).max()),
                    path_ratio_span=float(result.log_path_ratio.max()-result.log_path_ratio.min()),
                    observation_log_span=float(result.log_observation.max()-result.log_observation.min()),
                    total_log_span=float(result.log_weight.max()-result.log_weight.min()),
                    elapsed_seconds=time.monotonic()-started,sigma_grid=sigmas,
                    precision='BF16 states; ideal-Gaussian rounding-cell probabilities in FP64; finite RNG/arithmetic approximation remains')
                if index==0 and sigma==.75 and cfg==1. and args.smc_stages>0:
                    try:
                        smc=annealed_path_smc(innovations,stops,evaluate,max_stages=args.smc_stages,
                            mutation_steps=2,pcn_scale=.05,ess_fraction=.8,generator=generator)
                        row['smc']=dict(reached_target=True,beta=smc.beta,stages=len(smc.history),
                            evaluations=smc.evaluations,history=smc.history,
                            distinct_initial_ancestors=smc.distinct_initial_ancestors,
                            distinct_final_sources=smc.distinct_final_sources)
                    except TemperingBudgetExceeded as error:
                        row['smc']=dict(reached_target=False,error=str(error),
                            history=getattr(error,'history',[]),evaluations=getattr(error,'evaluations',None))
                records.append(row)
                (args.output/'diagnostics.json').write_text(json.dumps(dict(
                    scope='Real SD3/DMD-EMA path-sampling diagnostic on separate DrawBench development prompts; no image-quality evaluation',
                    configuration=vars(args)|{'output':str(args.output)},rows=records),indent=2,default=str,ensure_ascii=False))
                print(json.dumps(row,ensure_ascii=False),flush=True)
    print('SD3 path diagnostics completed',flush=True)


if __name__=='__main__':main()
