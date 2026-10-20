"""Persist the matched long-run checkpoints and report their completed tests."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

root = Path(__file__).resolve().parents[1]
r = root/'logs/path-ratio-round1'
deadline = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()
training = ['dmd-seed42-10000-confirm', 'path-is-train010-seed42-10000-confirm']
evaluations = ['dmd-seed42-10000-pick2048', 'path-is-seed42-10000-pick2048']


def wait(names, value):
    while True:
        status = [(r/'evidence'/n/'status.txt').read_text().strip()
                  if (r/'evidence'/n/'status.txt').exists() else '' for n in names]
        if 'FAILED' in status:
            raise RuntimeError(dict(zip(names, status)))
        if all(s == value for s in status):
            return
        if time.time() > deadline-120:
            raise RuntimeError('Round deadline reached before completion')
        time.sleep(10)


def execute(args):
    subprocess.run(['python', *args], cwd=root, check=True)


wait(training, 'MATCHED_10000_STEP_TRAINING_COMPLETED')
for name in training:
    archive = name+'-ema10000'
    if not (r/'archived-checkpoints'/archive/'storage-manifest.json').exists():
        execute(['scripts/persist_path_round_checkpoint.py', '--training-run', name,
                 '--step', '10000', '--name', archive])
execute(['scripts/audit_path_training_parameters.py', '--baseline', str(r/'runs'/training[0]/'config.json'),
         '--candidate', str(r/'runs'/training[1]/'config.json'), '--output', str(r/'parameter-audit-long.json')])
execute(['scripts/report_path_training_drift.py', '--baseline', str(r/'runs'/training[0]),
         '--candidate', str(r/'runs'/training[1]), '--step', '10000', '--output', str(r/'training-drift-long')])
wait(evaluations, 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION')
execute(['scripts/report_path_ratio_round.py', '--baseline', str(r/'evidence'/evaluations[0]),
         '--candidate', str(r/'evidence'/evaluations[1]), '--output', str(r/'comparison-long')])
(r/'long-confirmation-completion.json').write_text(json.dumps(dict(
    status='TRAINING_CHECKPOINT_ARCHIVE_PARAMETER_AUDIT_AND_FULL_TEST_COMPARISON_COMPLETE',
    completed_utc=datetime.now(timezone.utc).isoformat(), training=training,
    evaluations=evaluations), indent=2))
