"""Finish metadata only when all2048 four-model scores already exist.

Never generates images, invokes a scorer, or fills missing metric values.
This handles a budget cutoff after scoring but before CPU checkpoint audit.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-run',required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    r=root/'logs/path-ratio-round1'
    run=r/'runs'/a.evaluation_run
    evidence=r/'evidence'/a.evaluation_run
    folder=run/'eval/step-1'
    final=folder/'final-metrics.json';scores=folder/'per-image-scores.json'
    part=folder/'scores-rank-0.json'
    if (not final.exists() or not scores.exists()) and part.exists():
        existing=json.loads(part.read_text())
        keys=['clipscore','aesthetic','pickscore','hpsv2']
        assert len(existing)==2048 and [x['index'] for x in existing]==list(range(2048))
        means={key:float(np.mean([x['scores'][key] for x in existing],dtype=np.float64)) for key in keys}
        assert all(np.isfinite(value) for value in means.values())
        # Execute the unmodified original CPU reporting function only.
        source=root/'scripts/score_rtdmd_path_is_images.py'
        tree=ast.parse(source.read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='write_report')
        namespace={'Path':Path,'json':json,'hashlib':hashlib,'csv':csv}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        scores.write_bytes(part.read_bytes())
        namespace['write_report'](run,folder,1,existing,means)
    if not final.exists() or not scores.exists():
        print(a.evaluation_run,'No complete scored report; nothing to finalize')
        return
    status=evidence/'status.txt'
    if status.exists() and status.read_text().strip()=='COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION':
        print(a.evaluation_run,'Already completed and audited')
        return
    report=json.loads(final.read_text());records=json.loads(scores.read_text())
    assert report['num_images']==len(records)==2048
    assert [x['index'] for x in records]==list(range(2048))
    scales={'clipscore':('CLIP Score',100),'aesthetic':('Aesthetic Score',1),
            'pickscore':('Pick Score',26),'hpsv2':('HP Score',100)}
    for key,(column,scale) in scales.items():
        values=np.asarray([x['scores'][key] for x in records],dtype=float)
        assert np.isfinite(values).all()
        assert np.isclose(values.mean()*scale,report['table'][column],rtol=0,atol=1e-10)
    score_sha=hashlib.sha256(scores.read_bytes()).hexdigest()
    table={column:report['table'][column] for column,_ in scales.values()}
    cfg=json.loads((run/'config.json').read_text())
    checkpoint=(root/cfg['lora']['generator_path']).resolve()
    label=a.evaluation_run.removesuffix('-pick2048-exact')
    wrapper=(root/'jobs/evaluate_loaded_path_checkpoint_no_online.sh').read_text()
    marker='python - "$run" "$label" "$checkpoint" <<\'PY\'\n'
    body=wrapper.split(marker,1)[1].split('\nPY\n',1)[0]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-',str(run),label,str(checkpoint)],input=body,text=True,env=env,check=True,cwd=root)
    subprocess.run([sys.executable,'scripts/audit_loaded_checkpoint_evaluation.py','--run-dir',str(run)],env=env,check=True,cwd=root)
    out=json.loads(final.read_text())
    out['protocol']['unused_online_auxiliary_score_skipped']=True
    out['protocol']['cpu_only_metadata_finalization']=True
    assert table=={column:out['table'][column] for column,_ in scales.values()}
    assert hashlib.sha256(scores.read_bytes()).hexdigest()==score_sha
    final.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    evidence.mkdir(exist_ok=True)
    for source in [run/'config.json',run/'metrics.jsonl',run/'checkpoint-audit.json',
                   final,folder/'final-metrics.csv',scores]:
        shutil.copy2(source,evidence/source.name)
    (evidence/'cpu-metadata-recovery-audit.json').write_text(json.dumps(dict(
        scorer_or_generation_calls=0,all2048_four_scores_preexisted=True,
        per_image_score_bytes_unchanged=True,per_image_score_sha256=score_sha,
        source_CPU_postprocessing_sha256=hashlib.sha256(body.encode()).hexdigest()),indent=2))
    status.write_text('COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION\n')
    print(a.evaluation_run,'Existing complete scores audited and finalized on CPU')


if __name__=='__main__':
    main()
