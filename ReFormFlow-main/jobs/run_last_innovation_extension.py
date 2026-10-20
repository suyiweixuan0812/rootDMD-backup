"""Two additional pure Teacher-path detail trials after the bridge controls."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

root = Path(__file__).resolve().parents[1]
r = root/'logs/path-ratio-round1'
deadline = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()
progress = r/'last-innovation-extension-progress.json'


def record(data):
    pending = progress.with_suffix('.pending')
    pending.write_text(json.dumps(data, indent=2))
    pending.replace(progress)


record({'status':'WAITING_FOR_ADAPTIVE_BRIDGE_CONTROLS'})
while True:
    try:
        d = json.loads((r/'adaptive-extension-progress.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        d = {}
    if d.get('status') == 'BOTH_ADAPTIVE_DEV_TRIALS_COMPLETED':
        break
    if d.get('status') == 'CASE_FAILED':
        raise RuntimeError('Earlier cohort requires repair')
    if time.time() > deadline-3600:
        record({'status':'BUDGET_STOP_WHILE_WAITING'})
        raise RuntimeError('Insufficient remaining round time')
    time.sleep(10)
for preset in ['last', 'last_time']:
    remaining = int(deadline-time.time()-120)
    if remaining < 1800:
        record({'status':'BUDGET_STOP_BEFORE_NEW_TRIAL', 'next':preset})
        break
    record({'status':'RUNNING', 'current_case':preset})
    with (Path('/rootDMD')/f'path-is-det-{preset}-seed42.log').open('w') as stream:
        result = subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', str(remaining),
            'bash', 'jobs/run_deterministic_comparison_trial_v3.sh', preset, '42'],
            cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        record({'status':'CASE_FAILED', 'case':preset, 'returncode':result.returncode})
        raise SystemExit(result.returncode)
else:
    record({'status':'BOTH_LAST_INNOVATION_DEV_TRIALS_COMPLETED'})
