#!/usr/bin/env bash
# Matched training-seed confirmation. Hyperparameters must be frozen beforehand.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
method="${1:?dmd or path}"
preset="${2:?frozen training preset; ignored for dmd}"
seed="${3:?matched training seed}"
case "$method" in
 dmd) trainer=scripts/train_dmd_sd3.py; config=config/dmd_sd3_pickscore.py:train; name="dmd-seed$seed-confirm";;
 path)
  case "$preset" in
   train000|train025) trainer=scripts/train_rtdmd_path_is_sd3.py;config="config/rtdmd_path_is_pickscore.py:$preset";;
   train010|train_sigma) trainer=scripts/train_rtdmd_path_is_refined_sd3.py;config="config/rtdmd_path_is_refined.py:$preset";;
   *) exit 2;;
  esac
  name="path-is-$preset-seed$seed-confirm";;
 *) exit 2;;
esac
run="logs/path-ratio-round1/runs/$name"
evidence="logs/path-ratio-round1/evidence/$name"
if [[ -f "$run/config.json" ]]; then echo "Refusing overwrite $run" >&2;exit 2;fi
mkdir -p "$evidence"
printf 'MATCHED_SEED_TRAINING\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u "$trainer" --config="$config" --config.seed="$seed" \
 --config.run_name="$name" --config.save_dir="$REFORMDMD_ROOT/logs/path-ratio-round1/runs" \
 --config.eval.enabled=False --config.save_freq=1000
cp "$run/config.json" "$run/metrics.jsonl" "$evidence/"
printf 'MATCHED_SEED_TRAINING_COMPLETED\n' > "$evidence/status.txt"
