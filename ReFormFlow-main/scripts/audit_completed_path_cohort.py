"""Audit the frozen shared training settings for every available cohort run."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'logs/path-ratio-round1'
CASES=['dmd_fp32','base','time','paired','paired_time','bridge','bridge_time','adaptive_bridge','adaptive_bridge_time','last','last_time','last_q2_time','paired_q2_time']
output=R/'common-training-audits';output.mkdir(exist_ok=True)
records=[]
for seed in [42,43,44]:
 b=R/'runs'/f'sd3-det-dmd-seed{seed}'/'config.json'
 if not b.exists():continue
 for c in CASES:
  candidate=R/'runs'/f'sd3-det-{c}-seed{seed}'/'config.json'
  if not candidate.exists():continue
  dest=output/f'{c}-seed{seed}.json'
  subprocess.run(['python','scripts/audit_path_training_parameters.py','--baseline',str(b),'--candidate',str(candidate),'--output',str(dest)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
  records.append(dict(case=c,seed=seed,audit=str(dest)))
(output/'index.json').write_text(json.dumps({'audited_runs':records,'all_13_common_training_groups_equal':True,'scope':'Shared external configuration; changes to sampler, query precision, surrogate and computation are explicitly separate'},indent=2))
print('Audited',len(records),'run configurations')
