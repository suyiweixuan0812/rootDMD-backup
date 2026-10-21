#!/usr/bin/env bash
# Evaluate an ALREADY-EMA checkpoint with zero optimizer and zero extra EMA updates.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
checkpoint="${1:?EMA adapter directory}"
name="${2:?unique evaluation run name}"
dataset="${3:?prompt dataset directory}"
count="${4:?number of prompts}"
label="${5:?human-readable original training method}"
run="$REFORMDMD_ROOT/logs/path-ratio-round1/runs/$name"
evidence="$REFORMDMD_ROOT/logs/path-ratio-round1/evidence/$name"
if [[ -f "$run/config.json" ]]; then echo "Refusing to overwrite $run" >&2;exit 2;fi
mkdir -p "$evidence"
printf 'CHECKPOINT_EVALUATION\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
python -u scripts/train_rtdmd_path_is_sd3.py \
  --config=config/checkpoint_eval_no_online.py:eval \
  --config.run_name="$name" --config.lora.generator_path="$checkpoint" \
  --config.eval.dataset="$dataset" --config.eval.final_num_prompts="$count" \
  --config.train.ema=False --config.eval.use_ema=True
python -u scripts/score_rtdmd_path_is_images.py --run-dir "$run" --step 1 --expected-count "$count" --batch-size 8
python - "$run" "$label" "$checkpoint" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
run,label,checkpoint=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
history=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
assert not any('generator/loss_dm' in r for r in history)
source=load_file(str(checkpoint/'adapter_model.safetensors'))
exported=load_file(str(run/'checkpoints/checkpoint-1/lora/generator/adapter_model.safetensors'))
assert source.keys()==exported.keys() and all(torch.equal(source[k],exported[k]) for k in source)
folder=run/'eval/step-1'
p=folder/'final-metrics.json'
r=json.loads(p.read_text())
r['table']['Method']=label
r['table']['Post-Train']=label
r['protocol']['checkpoint_evaluation_only']=dict(path=str(checkpoint),optimizer_updates=0,
    ema_arithmetic_updates=0,all_exported_tensors_exactly_equal=True,tensor_count=len(source),
    source_sha256=hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest())
r['protocol']['interpretation']='Loaded EMA adapter evaluated unchanged; unused sampler configuration is not a training record.'
p.write_text(json.dumps(r,indent=2,ensure_ascii=False))
with (folder/'final-metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(r['table']));writer.writeheader();writer.writerow(r['table'])
(folder/'final-metrics.md').write_text(label+'\n\n'+json.dumps(r['table'],ensure_ascii=False,indent=2)+'\n')
(run/'checkpoint-audit.json').write_text(json.dumps(r['protocol']['checkpoint_evaluation_only'],indent=2))
print(json.dumps(r['table'],ensure_ascii=False),flush=True)
PY
cp "$run/config.json" "$run/metrics.jsonl" "$run/checkpoint-audit.json" "$evidence/"
cp "$run/eval/step-1/final-metrics.json" "$run/eval/step-1/final-metrics.csv" "$run/eval/step-1/per-image-scores.json" "$evidence/"
printf 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION\n' > "$evidence/status.txt"
