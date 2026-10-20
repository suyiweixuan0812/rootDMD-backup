#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
preset="${1:?dmd, base, paired, time, paired_time}"
seed="${2:-42}"
extra=()
case "$preset" in
 dmd) trainer=scripts/train_dmd_sd3.py;config=config/dmd_sd3_pickscore.py:train;;
 dmd_fp32) trainer=scripts/train_dmd_sd3_fp32_observation.py;config=config/dmd_fp32_observation_control.py:train;;
 last_q2_time|paired_q2_time) trainer=scripts/train_rtdmd_path_is_query_views_sd3.py;config="config/rtdmd_path_is_query_views.py:$preset";;
 last|last_time) trainer=scripts/train_rtdmd_path_is_last_sd3.py;config="config/rtdmd_path_is_last.py:$preset";;
 adaptive_bridge|adaptive_bridge_time) trainer=scripts/train_rtdmd_path_is_adaptive_bridge_sd3.py;config="config/rtdmd_path_is_adaptive_bridge.py:$preset";;
 bridge|bridge_time) trainer=scripts/train_rtdmd_path_is_bridge_sd3.py;config="config/rtdmd_path_is_bridge.py:$preset";;
 base) trainer=scripts/train_rtdmd_path_is_details_sd3.py;config=config/rtdmd_path_is_details.py:time;extra+=(--config.posterior_time_weight.mode=none);;
 paired|time|paired_time) trainer=scripts/train_rtdmd_path_is_details_sd3.py;config="config/rtdmd_path_is_details.py:$preset";;
 *) exit 2;;
esac
name="sd3-det-$preset-seed$seed"
run="logs/path-ratio-round1/runs/$name"
evidence="logs/path-ratio-round1/evidence/$name"
if [[ -f "$run/config.json" ]];then echo "Refusing overwrite $run" >&2;exit 2;fi
mkdir -p "$evidence"
printf 'DETERMINISTIC_TRAINING_AND_DEV_SCORING\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export REWARD_CKPT_PATH="${REWARD_CKPT_PATH:-/dev/shm/dmd-eval-weights/reward_ckpts}"
python -u scripts/run_deterministic_training.py "$trainer" --config="$config" \
 --config.run_name="$name" --config.seed="$seed" \
 --config.save_dir="$REFORMDMD_ROOT/logs/path-ratio-round1/runs" \
 --config.eval.dataset="$REFORMDMD_ROOT/dataset/drawbench" \
 --config.eval.num_prompts=8 --config.eval.final_num_prompts=64 \
 --config.eval_freq=1000 --config.save_freq=1000 "${extra[@]}"
python -u scripts/score_sd3_query_view_images.py --run-dir "$run" --step 1000 --expected-count 64 --batch-size 8
cp "$run/config.json" "$run/metrics.jsonl" "$run/runtime-numerics.json" "$evidence/"
cp "$run/eval/step-1000/final-metrics.json" "$run/eval/step-1000/final-metrics.csv" "$run/eval/step-1000/per-image-scores.json" "$evidence/"
printf 'COMPLETED_DETERMINISTIC_DEV64\n' > "$evidence/status.txt"
