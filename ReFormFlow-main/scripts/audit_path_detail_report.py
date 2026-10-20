"""Label actual active sampler/time weighting; leave every numeric score intact."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir',required=True,type=Path)
p.add_argument('--step',default=1000,type=int)
a=p.parse_args()
config=json.loads((a.run_dir/'config.json').read_text())
folder=a.run_dir/'eval'/f'step-{a.step}'
f=folder/'final-metrics.json';r=json.loads(f.read_text());before=r['raw_repository_means'].copy()
particle=config['path_is']['enabled'];mode=config['posterior_time_weight']['mode']
label=('Path-IS' if particle else 'Paired-source control')+f'; lambda={config["posterior_consistency"]["weight"]:.2f}'
if mode=='inverse_sigma':label+='; time weight=0.30/sigma'
else:label+='; time weight=1'
r['table']['Post-Train']=label
r['protocol']['posterior_time_weight']=config['posterior_time_weight']
r['protocol']['interpretation']=('Literal Student/Teacher path weights and same posterior target; deterministic query-time preconditioning.' if particle else 'Control: single paired Student source, no path ratio or particles; same query Jacobian and regularizer.')
assert before==r['raw_repository_means']
f.write_text(json.dumps(r,indent=2,ensure_ascii=False))
with (folder/'final-metrics.csv').open('w',newline='') as stream:
    w=csv.DictWriter(stream,fieldnames=list(r['table']));w.writeheader();w.writerow(r['table'])
(folder/'final-metrics.md').write_text(json.dumps(r['table'],indent=2,ensure_ascii=False)+'\n')
print(label)
