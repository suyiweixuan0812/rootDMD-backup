#!/usr/bin/env bash
# Local one-A100 comparison against the already completed 1000-step DMD run.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
comparison="$REFORMDMD_ROOT/logs/rtdmd-source-is/comparison-1000"
mkdir -p "$comparison"
trap 'printf "FAILED; see /rootDMD/rtdmd-source-is-comparison.log\n" > "$comparison/status.txt"' ERR
printf 'TRAINING\n' > "$comparison/status.txt"
python scripts/check_runtime.py --method dmd --model "$SD3_MODEL" --rewards pickscore clipscore aesthetic hpsv2
bash jobs/run_rtdmd_source_is_pickscore.sh train --config.save_freq=1000
printf 'SCORING_SOURCE_IS\n' > "$comparison/status.txt"
python -u scripts/score_rtdmd_images.py \
    --run-dir logs/rtdmd-source-is/sd3-medium-rtdmd-source-is-train \
    --step 1000 --expected-count 2048 --batch-size 8
printf 'SCORING_BASELINE\n' > "$comparison/status.txt"
python -u scripts/score_dmd_images.py \
    --run-dir logs/dmd/sd3-medium-pickscore-no-reward-train \
    --step 1000 --expected-count 2048 --batch-size 8
python scripts/summarize_source_is_comparison.py
printf 'COMPLETED\n' > "$comparison/status.txt"
