"""Analytic Gaussian-mixture benchmark, NOT PickScore/SD3 performance evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from flow_grpo.path_ratio_smc import (
    annealed_path_smc, cps_mean, effective_sample_size, evaluate_teacher_path,
)


def reference_mean(dimension):
    coefficient=float(cps_mean(torch.ones(1),.4*torch.ones(1),1.,.5,.9))
    v=torch.tensor([.6**2,.8**2*(coefficient**2+(.5*math.sin(.9*math.pi/2))**2)],dtype=torch.float64)
    obs_variance=.6**2*v+.4**2
    log_mass=-.5*dimension*(obs_variance.log()+math.log(2*math.pi))-.5*dimension*.25**2/obs_variance
    mass=log_mass.softmax(0)
    conditional=.6*v/obs_variance*.25
    return float((mass*conditional).sum()),mass.tolist()


def streaming_is(count, dimension, evaluator, generator, batch_size=2048):
    log_sum=None
    weighted_sum=None
    log_sum_square=None
    for start in range(0,count,batch_size):
        n=min(batch_size,count-start)
        e=torch.randn(n,2,dimension,generator=generator,dtype=torch.float64)
        stops=torch.randint(0,2,(n,),generator=generator)
        result=evaluator(e,stops)
        local=torch.logsumexp(result.log_weight,0)
        local_mean=(result.log_weight.softmax(0)[:,None]*result.source).sum(0)
        local_sq=torch.logsumexp(2*result.log_weight,0)
        if log_sum is None:
            log_sum,weighted_sum,log_sum_square=local,local_mean,local_sq
        else:
            combined=torch.logaddexp(log_sum,local)
            weighted_sum=(log_sum-combined).exp()*weighted_sum+(local-combined).exp()*local_mean
            log_sum,log_sum_square=combined,torch.logaddexp(log_sum_square,local_sq)
    return weighted_sum,float((2*log_sum-log_sum_square).exp())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensions',type=int,nargs='+',default=[8,64])
    parser.add_argument('--particles',type=int,default=64)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,59])
    parser.add_argument('--mutation-steps',type=int,default=16)
    parser.add_argument('--output',type=Path,default=Path('logs/path-ratio-prototype/analytic-validation'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for dimension in args.dimensions:
        expected,mass=reference_mean(dimension)
        def evaluate(e,stops):
            return evaluate_teacher_path(e,stops,torch.full((1,dimension),.25),.4,[1.,.5],.9,
                lambda x,s:.4*x,lambda x,s:.1*x,initial_mean=torch.full((1,dimension),.1))
        for seed in args.seeds:
            g=torch.Generator().manual_seed(seed)
            e=torch.randn(args.particles,2,dimension,generator=g,dtype=torch.float64)
            stops=torch.randint(0,2,(args.particles,),generator=g)
            initial=evaluate(e,stops)
            direct_mean=(initial.log_weight.softmax(0)[:,None]*initial.source).sum(0)
            started=time.monotonic()
            smc=annealed_path_smc(e,stops,evaluate,mutation_steps=args.mutation_steps,
                ess_fraction=.8,max_stages=128,pcn_scale=.15,
                generator=torch.Generator().manual_seed(seed+10000))
            elapsed=time.monotonic()-started
            equal_mean,equal_ess=streaming_is(smc.evaluations,dimension,evaluate,
                torch.Generator().manual_seed(seed+20000))
            rmse=lambda x:float((x-expected).square().mean().sqrt())
            row=dict(dimension=dimension,seed=seed,particles=args.particles,
                exact_mean_per_coordinate=expected,exact_stop_posterior=mass,
                one_shot=dict(ess=effective_sample_size(initial.log_weight),rmse=rmse(direct_mean),evaluations=args.particles),
                smc=dict(beta=smc.beta,stages=len(smc.history),rmse=rmse(smc.mean),
                    final_weight_ess=effective_sample_size(smc.log_weights),
                    minimum_pre_resampling_ess=min(r['ess_before_resampling'] for r in smc.history),
                    mean_actual_move_rate=float(np.mean([r['mutation_actual_move_rate'] for r in smc.history])),
                    distinct_initial_ancestors=smc.distinct_initial_ancestors,
                    distinct_final_sources=smc.distinct_final_sources,evaluations=smc.evaluations,
                    elapsed_seconds=elapsed,history=smc.history),
                equal_evaluation_budget_is=dict(ess=equal_ess,rmse=rmse(equal_mean),evaluations=smc.evaluations))
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='smc'} | {'smc':{k:v for k,v in row['smc'].items() if k!='history'}},ensure_ascii=False),flush=True)
    summary={}
    for dimension in args.dimensions:
        selected=[r for r in records if r['dimension']==dimension]
        summary[str(dimension)]={
            'one_shot_mean_ess':float(np.mean([r['one_shot']['ess'] for r in selected])),
            'one_shot_mean_rmse':float(np.mean([r['one_shot']['rmse'] for r in selected])),
            'smc_mean_rmse':float(np.mean([r['smc']['rmse'] for r in selected])),
            'smc_mean_final_weight_ess':float(np.mean([r['smc']['final_weight_ess'] for r in selected])),
            'smc_mean_stages':float(np.mean([r['smc']['stages'] for r in selected])),
            'smc_mean_evaluations':float(np.mean([r['smc']['evaluations'] for r in selected])),
            'smc_mean_actual_move_rate':float(np.mean([r['smc']['mean_actual_move_rate'] for r in selected])),
            'equal_budget_is_mean_rmse':float(np.mean([r['equal_evaluation_budget_is']['rmse'] for r in selected])),
            'equal_budget_is_mean_ess':float(np.mean([r['equal_evaluation_budget_is']['ess'] for r in selected])),
        }
    output=dict(scope='CPU linear-Gaussian path benchmark with exact random-stop-mixture posterior; NOT SD3/PickScore',
        summary=summary,runs=records,
        caveats=['Adaptive bridges and invariant mutation define the intended final target, not exact finite-N samples.',
            'Final weight ESS is not an independent sample count; ancestor and movement diagnostics are reported.',
            'SMC uses many more evaluations than same-particle one-shot IS; equal-evaluation-budget IS is also reported.',
            'No neural-network training, PickScore quality, scalability, or novelty claim is established.'])
    root=Path(__file__).resolve().parents[1]
    output['sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root/'flow_grpo/path_ratio_smc.py',Path(__file__),root/'tests/test_path_ratio_smc.py']}
    (args.output/'results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
