"""Two predeclared proposal-history controls after the initial factorial cohort."""
from pathlib import Path
from datetime import datetime,timezone
import json
import subprocess
import time

root=Path(__file__).resolve().parents[1];r=root/'logs/path-ratio-round1'
deadline=datetime(2026,10,21,1,20,tzinfo=timezone.utc).timestamp()
progress=r/'adaptive-extension-progress.json'
while True:
    p=r/'deterministic-cohort-progress.json'
    d=json.loads(p.read_text()) if p.exists() else {}
    if d.get('status')=='ALL_FACTORIAL_AND_CONTROL_DEV_TRIALS_COMPLETED':break
    if d.get('status')=='CASE_FAILED':raise RuntimeError('First cohort needs repair before extension')
    if time.time()>deadline-3600:raise RuntimeError('Insufficient remaining round time')
    time.sleep(10)
for preset in ['adaptive_bridge','adaptive_bridge_time']:
    remaining=int(deadline-time.time()-120)
    if remaining<1800:
        progress.write_text(json.dumps({'status':'BUDGET_STOP_BEFORE_NEW_TRIAL','next':preset},indent=2));break
    progress.write_text(json.dumps({'status':'RUNNING','current_case':preset},indent=2))
    with (Path('/rootDMD')/f'path-is-det-{preset}-seed42.log').open('w') as f:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(remaining),'bash',
            'jobs/run_deterministic_comparison_trial_v2.sh',preset,'42'],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        progress.write_text(json.dumps({'status':'CASE_FAILED','case':preset,'returncode':result.returncode},indent=2));raise SystemExit(result.returncode)
else:progress.write_text(json.dumps({'status':'BOTH_ADAPTIVE_DEV_TRIALS_COMPLETED'},indent=2))
