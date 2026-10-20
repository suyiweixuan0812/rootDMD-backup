"""Static research plot from the actual saved generator-update logs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/path-ratio-round1/figures'
OUT.mkdir(exist_ok=True)
items=[('lambda=0','train000'),('lambda=.25','train025'),('lambda=.10','train010'),('lambda=.5*(1-sigma)','train_sigma')]
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
colors=['#2463a7','#e38627','#279272','#9c4d9d']
for (label,preset),color in zip(items,colors):
    p=ROOT/'logs/path-ratio-round1/runs'/('sd3-medium-path-is-'+preset)/'metrics.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines() if 'path_is/ess' in x]
    assert len(rows)==100
    sigma=np.array([x['path_is/sigma'] for x in rows])
    ess=np.array([x['path_is/ess'] for x in rows])
    displacement=np.array([x['path_is/total_mean_displacement_rms'] for x in rows])
    axes[0,0].scatter(sigma,ess,s=12,alpha=.55,label=label,color=color)
    axes[0,1].scatter(sigma,np.maximum(displacement,1e-12),s=12,alpha=.55,color=color)
    centers=[];move=[];frozen=[]
    for lo,hi in [(0,.25),(.25,.5),(.5,.75),(.75,1)]:
        part=[r for r in rows if lo<=r['path_is/sigma']<hi]
        if part:
            centers.append((lo+hi)/2)
            move.append(np.mean([r['path_is/actual_move_rate'] for r in part]))
            frozen.append(np.mean([r['path_is/frozen_update'] for r in part]))
    axes[1,0].plot(centers,move,'o-',label=label,color=color)
    axes[1,1].plot(centers,frozen,'o-',color=color)
axes[0,0].set(title='Candidate-pool weight ESS (not independent ESS)',ylabel='Mean of two N=8 pools',ylim=(.7,8.3))
axes[0,0].legend(fontsize=8)
axes[0,1].set(title='Weighted mean displacement from paired source',ylabel='Latent RMS (zeros placed at 1e-12)',yscale='log')
axes[1,0].set(title='Actual selected-source movement',ylabel='Fraction of conditional sweeps',ylim=(-.03,1.03))
axes[1,1].set(title='Frozen whole updates',ylabel='Both sweeps have no source movement',ylim=(-.03,1.03))
for ax in axes.flat:
    ax.set(xlabel='Observation noise sigma',xlim=(0,1));ax.grid(alpha=.2)
fig.savefig(OUT/'path-is-diagnostics.png',dpi=180)
fig.savefig(OUT/'path-is-diagnostics.pdf')
print(OUT)
