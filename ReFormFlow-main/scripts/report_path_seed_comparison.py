"""Report every completed matched training-seed result, including negative seeds."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'logs/path-ratio-round1'
PAIRS=[(42,'dmd-ema-pick2048-exact','path-is-selected-pick2048-exact'),
       (43,'dmd-seed43-pick2048-exact','path-is-seed43-pick2048-exact')]
KEYS=['CLIP Score','Aesthetic Score','Pick Score','HP Score']
rows=[]
for seed,dmd,path in PAIRS:
    a,b=[json.loads((ROOT/'evidence'/name/'final-metrics.json').read_text()) for name in [dmd,path]]
    assert a['num_images']==b['num_images']==2048
    for key in ['test_file_sha256','resolution','seed','ema','clip','aesthetic','pickscore','hp']:
        assert a['protocol'][key]==b['protocol'][key]
    rows.append({'training_seed':seed,'DMD':a['table'],'PathIS':b['table'],
                 'deltas':{k:b['table'][k]-a['table'][k] for k in KEYS}})
summary={'rows':rows,'number_of_training_seeds':len(rows),
         'interpretation':'Seed42 improves; seed43 does not. Small positive two-run preference means do not establish stable superiority. CUDA backward is nondeterministic in this cohort; deterministic comparisons are separately planned.',
         'means':{m:{k:float(np.mean([r[m][k] for r in rows])) for k in KEYS} for m in ['DMD','PathIS']},
         'mean_deltas':{k:float(np.mean([r['deltas'][k] for r in rows])) for k in KEYS},
         'std_of_deltas_across_runs':{k:float(np.std([r['deltas'][k] for r in rows],ddof=1)) for k in KEYS},
         'warning':'Prompt-bootstrap intervals conditional on trained weights cannot establish training-seed stability. Do not combine these runs with a changed deterministic runtime cohort without separating them.'}
out=ROOT/'comparison-two-seeds';out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
lines=['两组匹配训练种子，原完整PickScore2048；包含训练CUDA非确定性。','本轮初始路径比版本：lambda=.10,N8,S2，推理NFE4。','']
for r in rows:
    lines.append(f'seed{r["training_seed"]}: '+json.dumps(r['deltas'],ensure_ascii=False))
lines+=['','DMD均值：'+json.dumps(summary['means']['DMD'],ensure_ascii=False),'PathIS均值：'+json.dumps(summary['means']['PathIS'],ensure_ascii=False),'均值差：'+json.dumps(summary['mean_deltas'],ensure_ascii=False),summary['interpretation']]
(out/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
