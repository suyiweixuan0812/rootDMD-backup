#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
comparison="$REFORMDMD_ROOT/logs/rtdmd-posterior-mh/comparison-1000"
run="$REFORMDMD_ROOT/logs/rtdmd-posterior-mh/sd3-medium-rtdmd-posterior-mh-train"
if [[ -e "$run/config.json" ]]; then
    echo "Refusing to overwrite an existing training run: $run" >&2
    exit 2
fi
mkdir -p "$comparison"
trap 'printf "FAILED; see /rootDMD/rtdmd-posterior-mh-comparison.log\n" > "$comparison/status.txt"' ERR
python scripts/check_runtime.py --method dmd --model "$SD3_MODEL" --rewards pickscore clipscore aesthetic hpsv2
printf 'TRAINING_AND_FINAL_IMAGE_GENERATION\n' > "$comparison/status.txt"
bash jobs/run_rtdmd_posterior_mh_pickscore.sh train --config.save_freq=1000
printf 'SCORING_POSTERIOR_MH\n' > "$comparison/status.txt"
python -u scripts/score_rtdmd_posterior_mh_images.py \
    --run-dir "$run" --step 1000 --expected-count 2048 --batch-size 8
# Previous PNG scores are reused only after strict protocol/prompt matching.
python scripts/summarize_posterior_mh_comparison.py
printf 'COMPLETED\n' > "$comparison/status.txt"
