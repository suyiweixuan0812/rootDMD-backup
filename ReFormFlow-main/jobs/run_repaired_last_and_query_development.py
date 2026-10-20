"""Bounded repair retry and predeclared two-query comparison; no test-set tuning."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,os
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'logs/path-ratio-round1'
D=datetime(2026,10,21,1,20,tzinfo=timezone.utc).timestamp()
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def write(file,**x):
 p=R/file;t=p.with_suffix('.pending');t.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**x),indent=2));t.replace(p)
def state(c):
 p=R/'evidence'/f'sd3-det-{c}-seed42/status.txt';return p.read_text().strip() if p.exists() else ''
def groups():
 for cases,file,done in [(['last','last_time'],'last-innovation-extension-progress.json','BOTH_LAST_INNOVATION_DEV_TRIALS_COMPLETED'),(['adaptive_bridge','adaptive_bridge_time'],'adaptive-extension-progress.json','BOTH_ADAPTIVE_DEV_TRIALS_COMPLETED'),(['last_q2_time','paired_q2_time'],'query-view-extension-progress.json','BOTH_QUERY_VIEW_DEV_TRIALS_COMPLETED')]:
  ss=[state(c) for c in cases];status=done if all(s=='COMPLETED_DETERMINISTIC_DEV64' for s in ss) else 'CASE_FAILED' if 'FAILED' in ss else 'RUNNING_REPAIRED_OR_QUEUED'
  write(file,status=status,cases=cases,states=ss)
write('query-view-extension-progress.json',status='WAITING_FOR_PHYSICAL_SINGLE_ROUND_SMOKE')
smoke=R/'runs/sd3-query-views-single-round-smoke/runtime-numerics.json'
while True:
 x=json.loads(smoke.read_text()) if smoke.exists() else {}
 if x.get('status')=='COMPLETED':break
 if x.get('status')=='FAILED':raise RuntimeError('Single-round smoke failed')
 time.sleep(5)
# The previous controller has exited after its valid adaptive job. Do not stop a valid trainer.
old=68790
try:os.kill(old,15)
except ProcessLookupError:pass
for phase in [['last','last_time'],['last_q2_time','paired_q2_time']]:
 if phase[0].startswith('last_q2'):
  while state('adaptive_bridge_time')!='COMPLETED_DETERMINISTIC_DEV64':
   if state('adaptive_bridge_time')=='FAILED':raise RuntimeError('Adaptive trial requires repair')
   groups();time.sleep(5)
 children={}
 for c in phase:
  if state(c)=='COMPLETED_DETERMINISTIC_DEV64':continue
  if D-time.time()<3600:raise RuntimeError('Budget prevents new dev launch')
  log=open(Path('/rootDMD')/f'path-is-det-{c}-seed42.log','w')
  children[c]=subprocess.Popen(['bash','jobs/run_deterministic_comparison_trial_v4.sh',c,'42'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while children:
  for c,p in list(children.items()):
   if p.poll() is None:continue
   if p.returncode:groups();raise RuntimeError(f'{c} failed')
   del children[c]
  groups();time.sleep(5)
groups()
print('ALL_REPAIRED_AND_QUERY_DEVELOPMENT_COMPLETE',flush=True)
