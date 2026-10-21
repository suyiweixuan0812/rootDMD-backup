"""One bounded8-PNG parity check when the third model slot becomes free."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,time,hashlib
import run_selected_deterministic_validation_v3 as v
R,ROOT,D=v.R,v.ROOT,v.D
out=R/'unused-online-score-parity-audit.json'
while time.time()<D-1800:
 if all(v.state(f'sd3-det-{c}-seed43-pick2048-exact')=='COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION' for c in ['dmd','bridge','dmd_fp32','paired']):break
 if v.read(v.PROGRESS).get('status','').startswith('FAILED'):raise SystemExit('Required validation needs repair')
 time.sleep(5)
else:raise SystemExit('No early test slot inside budget')
name='sd3-unused-online-skip-parity8';checkpoint=R/'archived-checkpoints/sd3-det-dmd-seed42-ema1000/lora/generator'
with open('/rootDMD/path-is-unused-online-parity8.log','w') as f:
 p=subprocess.run(['timeout','--signal=TERM','--kill-after=15s','180','bash','jobs/evaluate_loaded_path_checkpoint_no_online_v2.sh',str(checkpoint),name,'dataset/pickscore','8','NUMERIC_PARITY_ONLY'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0,'Parity check incomplete; original evaluator remains default'
a=R/'runs/sd3-det-dmd-seed42-pick2048-exact/eval/step-1/rank-0';b=R/'runs'/name/'eval/step-1/rank-0'
checks=[]
for i in range(8):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 x,y=sha(a/f'{i:06d}.png'),sha(b/f'{i:06d}.png');checks.append(dict(index=i,reference_sha256=x,trial_sha256=y,equal=x==y))
result=dict(utc=datetime.now(timezone.utc).isoformat(),all_eight_PNG_bytes_exactly_equal=all(x['equal'] for x in checks),checks=checks,scope='Same nativeFP32 EMA, source scripts,512pixels,batch4,seed42 and first8PickScore prompts. Official four offline scorers unchanged. Scope is the tested prefix; no assertion of measured2048-image equality.',activation='Only final-window long evaluation may select the no-unused-online evaluator if all8match; original scripts and required cohort evaluations stay unchanged.')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
