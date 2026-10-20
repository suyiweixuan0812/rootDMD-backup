#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
preset="${1:?paired, time, paired_time}"
seed="${2:-42}"
case "$preset" in paired|time|paired_time);; *)exit 2;;esac
name="sd3-medium-path-is-details-$preset-seed$seed"
run="logs/path-ratio-round1/runs/$name"
evidence="logs/path-ratio-round1/evidence/$name"
if [[ -f "$run/config.json" ]];then echo "Refusing overwrite $run" >&2;exit 2;fi
mkdir -p "$evidence"
printf 'DETAIL_TRAINING_AND_DEV_SCORING\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
python -u scripts/train_rtdmd_path_is_details_sd3.py \
 --config="config/rtdmd_path_is_details.py:$preset" --config.seed="$seed" --config.run_name="$name"
python -u scripts/score_sd3_comparison_images.py --run-dir "$run" --step 1000 --expected-count 64 --batch-size 8
python scripts/audit_path_detail_report.py --run-dir "$run"
cp "$run/config.json" "$run/metrics.jsonl" "$evidence/"
cp "$run/eval/step-1000/final-metrics.json" "$run/eval/step-1000/final-metrics.csv" "$run/eval/step-1000/per-image-scores.json" "$evidence/"
printf 'COMPLETED_DEV64\n' > "$evidence/status.txt"
