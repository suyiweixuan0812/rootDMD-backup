"""Bounded sequential factorial/control experiments after the S8 trial."""
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

root=Path(__file__).resolve().parents[1]
r=root/'logs/path-ratio-round1'
status=r/'evidence/sd3-medium-path-is-train010-sweeps8/status.txt'
deadline=datetime(2026,10,21,1,20,tzinfo=timezone.utc).timestamp()
progress=r/'deterministic-cohort-progress.json'
cases=['dmd','dmd_fp32','base','time','paired','paired_time','bridge','bridge_time']
while True:
    value=status.read_text().strip() if status.exists() else ''
    if value in {'COMPLETED_DEV64','FAILED'}:break
    if time.time()>deadline-3600:raise RuntimeError('Insufficient round time')
    time.sleep(10)
for preset in cases:
    name=f'sd3-det-{preset}-seed42'
    completed=r/'evidence'/name/'status.txt'
    if completed.exists() and completed.read_text().strip()=='COMPLETED_DETERMINISTIC_DEV64':continue
    remaining=int(deadline-time.time()-120)
    if remaining<1800:
        progress.write_text(json.dumps({'status':'BUDGET_STOP_BEFORE_NEW_TRIAL','next':preset},indent=2));break
    row={'status':'RUNNING','current_case':preset,'all_cases':cases,'started_utc':datetime.now(timezone.utc).isoformat()}
    progress.write_text(json.dumps(row,indent=2))
    log=Path('/rootDMD')/f'path-is-det-{preset}-seed42.log'
    with log.open('w') as stream:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(remaining),
            'bash','jobs/run_deterministic_comparison_trial.sh',preset,'42'],cwd=root,
            stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        row.update(status='CASE_FAILED',returncode=result.returncode,log=str(log));progress.write_text(json.dumps(row,indent=2));raise SystemExit(result.returncode)
else:progress.write_text(json.dumps({'status':'ALL_FACTORIAL_AND_CONTROL_DEV_TRIALS_COMPLETED','cases':cases},indent=2))
