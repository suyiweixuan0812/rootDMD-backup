"""One-shot dependency for this round, not a recurring automation."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-run',required=True)
p.add_argument('--evaluation-run',required=True)
p.add_argument('--label',required=True)
p.add_argument('--checkpoint-step',default=1000,type=int)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
round_root=root/'logs/path-ratio-round1'
status=round_root/'evidence'/a.training_run/'status.txt'
out=round_root/'evidence'/a.evaluation_run
out.mkdir(exist_ok=True)
(out/'status.txt').write_text('WAITING_FOR_MATCHED_TRAINING\n')
deadline=datetime(2026,10,21,1,20,tzinfo=timezone.utc).timestamp()
while True:
    if time.time()>deadline-1800:
        (out/'status.txt').write_text('NOT_LAUNCHED_INSUFFICIENT_ROUND_TIME\n')
        raise RuntimeError('Insufficient time before the fixed round deadline')
    value=status.read_text().strip() if status.exists() else ''
    if value=='FAILED':raise RuntimeError('Training failed; no checkpoint evaluation launched')
    if value in {'MATCHED_SEED_TRAINING_COMPLETED','MATCHED_10000_STEP_TRAINING_COMPLETED'}:break
    time.sleep(10)
checkpoint=round_root/'runs'/a.training_run/'checkpoints'/f'checkpoint-{a.checkpoint_step}'/'lora/generator'
subprocess.run(['bash','jobs/evaluate_loaded_path_checkpoint_v2.sh',str(checkpoint),
    a.evaluation_run,'dataset/pickscore','2048',a.label],cwd=root,check=True)
