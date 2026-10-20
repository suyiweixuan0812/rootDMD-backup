"""Run an unchanged training entry with explicit deterministic CUDA settings."""
import json
import hashlib
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime,timezone

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
if len(sys.argv)<2:raise SystemExit('usage: run_deterministic_training.py TRAINER [normal trainer flags]')
trainer=sys.argv[1]
forwarded=sys.argv[2:]
name=next((x.split('=',1)[1] for x in forwarded if x.startswith('--config.run_name=')),None)
if not name:raise SystemExit('Explicit unique --config.run_name is required')
root=Path(__file__).resolve().parents[1]
run=root/'logs/path-ratio-round1/runs'/name
run.mkdir(parents=True,exist_ok=True)
record={'started_utc':datetime.now(timezone.utc).isoformat(),'deterministic_algorithms':True,
        'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'cudnn_benchmark':False,'cudnn_deterministic':True,'torch_version':torch.__version__,
        'trainer':trainer,'trainer_arguments':forwarded,'status':'RUNNING'}
record['cpu_runtime']={'torch_num_threads':torch.get_num_threads(),
    'torch_num_interop_threads':torch.get_num_interop_threads(),
    'thread_environment':{k:os.environ.get(k) for k in
        ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
sources={root/trainer,Path(__file__).resolve()}
sources.update((root/'flow_grpo').glob('*path_is*.py'))
sources.update((root/'flow_grpo').glob('*path_ratio*.py'))
sources.update((root/'flow_grpo').glob('*rounding*.py'))
sources.update((root/'flow_grpo').glob('*query*.py'))
sources.update((root/'flow_grpo/diffusers_patch').glob('sd3*path_is*.py'))
sources.update((root/'config').glob('*.py'))
record['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(sources) if p.is_file()}
path=run/'runtime-numerics.json'
assert not path.exists(),'Never overwrite a deterministic training record'
path.write_text(json.dumps(record,indent=2))
sys.argv=[trainer,*forwarded]
try:
    runpy.run_path(trainer,run_name='__main__')
except SystemExit as e:
    record['status']='COMPLETED' if e.code in (None,0) else 'FAILED'
    raise
except Exception:
    record['status']='FAILED'
    raise
else:record['status']='COMPLETED'
finally:
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(record,indent=2))
