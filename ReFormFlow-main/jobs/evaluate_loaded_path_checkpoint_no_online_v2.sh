#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
bash jobs/evaluate_loaded_path_checkpoint_no_online.sh "$@"
python scripts/audit_loaded_checkpoint_evaluation.py --run-dir "logs/path-ratio-round1/runs/$2"
python - "$2" <<'PY'
from pathlib import Path
import json,sys,shutil
r=Path('logs/path-ratio-round1');run=r/'runs'/sys.argv[1];p=run/'eval/step-1/final-metrics.json';x=json.loads(p.read_text());before=x['table'].copy()
x['protocol']['unused_online_auxiliary_score_skipped']=True
x['protocol']['actual_scoring']='Identical original four offline scorers and saved PNGs; only redundant in-loop PickScore omitted'
assert before==x['table'];p.write_text(json.dumps(x,indent=2,ensure_ascii=False));shutil.copy2(p,r/'evidence'/sys.argv[1]/p.name)
PY
