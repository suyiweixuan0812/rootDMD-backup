#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
label="${1:?025 or 050}"
case "$label" in 025) preset=train;;050) preset=sensitivity;;*) exit 2;;esac
run="$REFORMDMD_ROOT/logs/rtdmd-pcmh/sd3-medium-rtdmd-pcmh-lambda${label}-train"
if [[ -f "$run/config.json" ]]; then echo "Refusing to overwrite $run" >&2;exit 2;fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
printf 'TRAINING_AND_IMAGE_GENERATION\n' > "$run/status.txt"
trap 'printf "FAILED\n" > "$run/status.txt"' ERR
python -u scripts/train_rtdmd_pcmh_sd3.py --config="config/rtdmd_pcmh_pickscore.py:$preset" --config.save_freq=1000
printf 'SCORING\n' > "$run/status.txt"
python -u scripts/score_rtdmd_pcmh_images.py --run-dir "$run" --step 1000 --expected-count 2048 --batch-size 8
printf 'COMPLETED\n' > "$run/status.txt"
