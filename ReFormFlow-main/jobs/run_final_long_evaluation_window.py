"""Use a final15-minute opportunity only after all required work/training ends.

The existing long controller keeps its conservative25-minute gate. If it
returns without starting evaluation, this monitor can try the SAME frozen
checkpoints/prompts before the SAME hard deadline. No training is interrupted.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,time
import run_selected_deterministic_validation_v3 as v
R,ROOT,D=v.R,v.ROOT,v.D
P=R/'optional-long-final-evaluation-window-progress.json'
EVALUATOR='jobs/evaluate_loaded_path_checkpoint_v2.sh'
def record(status,**x):
 p=P.with_suffix('.pending');p.write_text(json.dumps(dict(status=status,utc=datetime.now(timezone.utc).isoformat(),**x),indent=2));p.replace(P)
def evaluate(training):
 name=training+'-pick2048-exact'
 assert not (R/'runs'/name/'config.json').exists(),'Never overwrite an existing evaluation'
 checkpoint=R/'runs'/training/'checkpoints/checkpoint-3000/lora/generator'
 assert (checkpoint/'adapter_model.safetensors').is_file()
 with (Path('/rootDMD')/f'path-is-{name}-final-window.log').open('w') as log:
  subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(int(D-time.time()-180)),
   'bash',EVALUATOR,str(checkpoint),name,'dataset/pickscore','2048',training],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 return name
record('WAITING_FOR_REQUIRED_TRAINING_AND_VALIDATION')
try:
 while time.time()<D-900:
  long=v.read(R/'optional-faithful-long-progress.json');state=long.get('status','')
  if state=='OPTIONAL_FAITHFUL_LONG_CHECK_COMPLETE':record('NOT_NEEDED_PARENT_COMPLETED');break
  if state.startswith('FAILED') or state.startswith('NOT_LAUNCHED'):
   record('NOT_LAUNCHED_PARENT_NOT_READY',parent=state);break
  if state in {'BUDGET_STOP_BEFORE_LONG_TEST','BUDGET_STOP_BEFORE_COMPLETE_LONG_TEST'}:
   if v.read(v.PROGRESS).get('status')!='ALL_PLANNED_VALIDATION_COMPLETED':time.sleep(5);continue
   training=long.get('training',[])
   assert len(training)==2
   if any((R/'runs'/(n+'-pick2048-exact')/'config.json').exists() for n in training):
    record('NOT_LAUNCHED_EXISTING_EVALUATION_PRESERVED');break
   # At this point the original controller has joined all training and test
   # futures; a budget-stop status precedes its clean exit. No train is live.
   time.sleep(2)
   old=R/'optional-faithful-long-before-final-evaluation-window.json'
   assert not old.exists();old.write_text(json.dumps(long,indent=2))
   parity=v.read(R/'unused-online-score-parity-audit.json')
   if parity.get('all_eight_PNG_bytes_exactly_equal'):
    EVALUATOR='jobs/evaluate_loaded_path_checkpoint_no_online_v2.sh'
   record('RUNNING_SAME_FROZEN_3000_FULL_TEST',training=training,evaluator=EVALUATOR)
   with ThreadPoolExecutor(max_workers=2) as pool:reports=list(pool.map(evaluate,training))
   v.compare(reports[0],[reports[1]],'comparison-optional-faithful-long3000')
   result=dict(status='OPTIONAL_FAITHFUL_LONG_CHECK_COMPLETE',utc=datetime.now(timezone.utc).isoformat(),faithful_case=v.read(R/'deterministic-selection.json')['pure_teacher_alternative']['case'],reports=reports,training=training,archived=long.get('archived',[]),prefix_audit='deterministic-full-prefix-reproduction.json',evaluation_window='Same full2048 test launched in final15-to25-minute window; no score-based selection',evaluator=EVALUATOR)
   target=R/'optional-faithful-long-progress.json';pending=target.with_suffix('.pending');pending.write_text(json.dumps(result,indent=2));pending.replace(target)
   record('SAME_FROZEN_LONG_TEST_COMPLETED',reports=reports);break
  time.sleep(5)
 else:record('NOT_LAUNCHED_FINAL_TIME_WINDOW_CLOSED')
except Exception as e:
 record('FINAL_WINDOW_INCOMPLETE_OR_FAILED',error=repr(e))
