"""Read-only compact monitor; exits at the fixed round deadline."""
from pathlib import Path
from datetime import datetime,timezone
import json,time
R=Path(__file__).resolve().parents[1]/'logs/path-ratio-round1';D=datetime(2026,10,21,1,20,tzinfo=timezone.utc).timestamp()
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def last(p):
 try:
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));lines=f.read().splitlines()
  for line in reversed(lines):
   try:return json.loads(line)
   except json.JSONDecodeError:pass
 except FileNotFoundError:pass
 return {}
seen={p.parent.name for p in (R/'evidence').glob('*/status.txt') if p.read_text().strip().startswith('COMPLETED')}
while time.time()<D:
 states={x:read(R/(x+'.json')) for x in ['last-innovation-extension-progress','query-view-extension-progress','selected-deterministic-validation-progress','optional-faithful-long-progress','optional-seed44-progress']}
 active=[]
 for p in (R/'evidence').glob('*/status.txt'):
  s=p.read_text().strip();name=p.parent.name
  if s=='DETERMINISTIC_TRAINING_AND_DEV_SCORING':
   row=last(R/'runs'/name/'metrics.jsonl');active.append(f'{name}:{row.get("step","loading")}/1000')
  elif s=='CHECKPOINT_EVALUATION':active.append(f'{name}:{len(list((R/"runs"/name/"eval/step-1/rank-0").glob("*.png")))}PNGs')
  if name not in seen and s.startswith('COMPLETED'):
   report=read(p.parent/'final-metrics.json')
   if report:print('COMPLETED',name,json.dumps(report['table']),flush=True)
   seen.add(name)
 resources=last(R.parents[2]/'development-load-guard/resource-monitor.jsonl')
 print(datetime.now(timezone.utc).strftime('%H:%M:%S'),json.dumps({'active':active,'statuses':{k:v.get('status') for k,v in states.items()},'resources':[resources.get('mean_last12_cpu'),resources.get('mean_last12_gpu')]}),flush=True)
 time.sleep(min(55,max(0,D-time.time())))
