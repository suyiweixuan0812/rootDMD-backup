#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
run="logs/path-ratio-round1/runs/sd3-medium-path-is-train010-sweeps8"
evidence="logs/path-ratio-round1/evidence/sd3-medium-path-is-train010-sweeps8"
if [[ -f "$run/config.json" ]];then echo "Refusing overwrite $run" >&2;exit 2;fi
mkdir -p "$evidence"
printf 'SAMPLING_DETAIL_TRAINING\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
python -u scripts/train_rtdmd_path_is_refined_sd3.py --config=config/rtdmd_path_is_sampling_ablation.py:sweeps8
python -u scripts/score_rtdmd_path_is_images.py --run-dir "$run" --step 1000 --expected-count 64 --batch-size 8
cp "$run/config.json" "$run/metrics.jsonl" "$evidence/"
cp "$run/eval/step-1000/final-metrics.json" "$run/eval/step-1000/final-metrics.csv" "$run/eval/step-1000/per-image-scores.json" "$evidence/"
printf 'COMPLETED_DEV64\n' > "$evidence/status.txt"
