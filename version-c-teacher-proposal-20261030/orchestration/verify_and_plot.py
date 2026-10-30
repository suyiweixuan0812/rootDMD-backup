"""CPU verification from raw scores; usable remotely and from a local export."""
from pathlib import Path
import argparse, json, hashlib, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS={'CLIP Score':('clipscore',100.),'Aesthetic Score':('aesthetic',1.),'Pick Score':('pickscore',26.),'HP Score':('hpsv2',100.)}
def main(root):
    root=Path(root);out=root/'comparison';out.mkdir(exist_ok=True)
    curves={};ends={};all_rows=[];checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name)
    for arm in ('dmd','teacher_c'):
        folder=root/'runs'/arm
        times=json.loads((folder/'evaluation-times.json').read_text())
        end=json.loads((folder/'training-complete.json').read_text());ends[arm]=end
        check(arm+' initial time zero',times[0]['training_seconds']==0)
        check(arm+' monotonic times',all(a['training_seconds']<b['training_seconds'] for a,b in zip(times,times[1:])))
        check(arm+' final time matches',times[-1]['training_seconds']==end['training_seconds'])
        data=[]
        for event in times:
            step=event['outer_step'];ev=folder/'eval'/f'step-{step}'
            rows=json.loads((ev/'per-image-scores.json').read_text())
            report=json.loads((ev/'final-metrics.json').read_text())
            check(f'{arm}/{step} 1000 ordered samples',[r['index'] for r in rows]==list(range(1000)))
            prompts=[r['prompt'] for r in rows]
            check(f'{arm}/{step} 200 prompts x5',len(set(prompts))==200 and all(prompts.count(p)==5 for p in set(prompts)))
            vals={k:float(np.mean([r['scores'][raw] for r in rows],dtype=np.float64)*scale) for k,(raw,scale) in METRICS.items()}
            check(f'{arm}/{step} recomputed scores',all(abs(vals[k]-report['table'][k])<1e-9 for k in vals))
            check(f'{arm}/{step} report time',report['training_seconds']==event['training_seconds'])
            entry=dict(method=arm,**event,training_minutes=event['training_seconds']/60,**vals)
            data.append(entry);all_rows.append(entry)
        curves[arm]=data
        final_cp=folder/'checkpoints'/f"checkpoint-{end['outer_step']}"/'complete.json'
        receipt=json.loads(final_cp.read_text())
        check(arm+' checkpoint ranks synchronized',len(receipt['rank_state_hashes'])==8 and all(h==receipt['rank_state_hashes'][0] for h in receipt['rank_state_hashes']))
        for name,meta in receipt['files'].items():
            file=final_cp.parent/name
            if file.exists():check(arm+' checkpoint hash '+name,hashlib.sha256(file.read_bytes()).hexdigest()==meta['sha256'])
    check('DMD10000',ends['dmd']['outer_step']==10000 and ends['dmd']['generator_step']==1000 and len(curves['dmd'])==11)
    target=ends['dmd']['training_seconds'];c=ends['teacher_c']
    check('C first complete update crossing final budget',c['previous_update_boundary_seconds']<target<=c['training_seconds'])
    events=json.loads((root/'runs/teacher_c/time-budget-events.json').read_text())
    check('ten measured DMD target times',len(events)==10)
    for event,ref in zip(events,curves['dmd'][1:]):
        check('first crossing '+str(ref['outer_step']),event['target']['training_seconds']==ref['training_seconds'] and event['previous_update_seconds']<ref['training_seconds']<=event['training_seconds'])
    for name in ('image-hashes.json',):
        a=json.loads((root/'runs/dmd/eval/step-0'/name).read_text());b=json.loads((root/'runs/teacher_c/eval/step-0'/name).read_text())
        check('initial 1000 PNG paired',a==b and len(a)==1000)
    a=json.loads((root/'runs/dmd/checkpoints/checkpoint-0/complete.json').read_text())
    b=json.loads((root/'runs/teacher_c/checkpoints/checkpoint-0/complete.json').read_text())
    check('initial complete states paired',a['rank_state_hashes']==b['rank_state_hashes'])
    bootstrap={};rng=np.random.default_rng(20261030)
    finals={a:json.loads((root/'runs'/a/'eval'/f"step-{ends[a]['outer_step']}"/'per-image-scores.json').read_text()) for a in ends}
    prompts=sorted({r['prompt'] for r in finals['dmd']});draw=rng.integers(0,len(prompts),(5000,len(prompts)))
    for title,(key,scale) in METRICS.items():
        values={a:np.array([np.mean([r['scores'][key]*scale for r in finals[a] if r['prompt']==p]) for p in prompts]) for a in ends}
        delta=values['teacher_c']-values['dmd'];ci=np.quantile(delta[draw].mean(1),[.025,.975])
        bootstrap[title]=dict(delta_C_minus_DMD=float(delta.mean()),cluster_bootstrap_95ci=ci.tolist())
    summary=dict(status='verified',dmd=ends['dmd'],teacher_c=c,final_scores={a:curves[a][-1] for a in curves},
        delta=bootstrap,time_difference_seconds=c['training_seconds']-target,checks_passed=len(checks),
        limitations='Single seed; DrawBench200 prompts x5. Bootstrap quantifies prompt uncertainty, not seed uncertainty. Actual times are not interpolated.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'curves.json').write_text(json.dumps(curves,indent=2))
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,metric in zip(axes.flat,METRICS):
        for arm,label in [('dmd','DMD (new supplied code)'),('teacher_c','Version C + Teacher endpoint proposal')]:
            ax.plot([r['training_minutes'] for r in curves[arm]],[r[metric] for r in curves[arm]],marker='o',markersize=3,label=label)
        ax.set(xlabel='Actual training time (minutes)',ylabel=metric,title=metric);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('SD3 | Equal training time | DrawBench 200 prompts x 5 | seed 42')
    for ext in ('png','svg'):fig.savefig(out/f'four_metrics_training_time.{ext}',dpi=180)
    plt.close(fig)
    (root/'formal-verification.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
