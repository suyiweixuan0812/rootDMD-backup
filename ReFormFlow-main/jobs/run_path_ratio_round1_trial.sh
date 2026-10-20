#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
preset="${1:?train000, train025 or baseline_eval}"
case "$preset" in
  train000|train025) label="sd3-medium-path-is-${preset}";step=1000;;
  baseline_eval) label=dmd-ema-dev64;step=1;;
  *) exit 2;;
esac
run="$REFORMDMD_ROOT/logs/path-ratio-round1/runs/$label"
evidence="$REFORMDMD_ROOT/logs/path-ratio-round1/evidence/$label"
if [[ -f "$run/config.json" ]]; then echo "Refusing to overwrite $run" >&2;exit 2;fi
mkdir -p "$run" "$evidence"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
printf 'TRAINING_AND_DEV_IMAGE_GENERATION\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
python -u scripts/train_rtdmd_path_is_sd3.py --config="config/rtdmd_path_is_pickscore.py:$preset"
printf 'SCORING_DEV64\n' > "$evidence/status.txt"
python -u scripts/score_rtdmd_path_is_images.py --run-dir "$run" --step "$step" --expected-count 64 --batch-size 8
cp "$run/config.json" "$run/metrics.jsonl" "$evidence/"
cp "$run/eval/step-$step/final-metrics.json" "$run/eval/step-$step/final-metrics.csv" "$run/eval/step-$step/per-image-scores.json" "$evidence/"
printf 'COMPLETED_DEV64\n' > "$evidence/status.txt"
