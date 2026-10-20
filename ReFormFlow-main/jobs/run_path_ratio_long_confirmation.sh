#!/usr/bin/env bash
# Fresh matched 10k-step runs, with the selected sampler/regularizer frozen.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
method="${1:?dmd or path}"
case "$method" in
 dmd) trainer=scripts/train_dmd_sd3.py; config=config/dmd_sd3_pickscore.py:train;name=dmd-seed42-10000-confirm;;
 path) trainer=scripts/train_rtdmd_path_is_refined_sd3.py;config=config/rtdmd_path_is_refined.py:train010;name=path-is-train010-seed42-10000-confirm;;
 *) exit 2;;
esac
run="logs/path-ratio-round1/runs/$name"
evidence="logs/path-ratio-round1/evidence/$name"
if [[ -f "$run/config.json" ]]; then echo "Refusing overwrite $run" >&2;exit 2;fi
mkdir -p "$evidence"
printf 'MATCHED_10000_STEP_TRAINING\n' > "$evidence/status.txt"
trap 'printf "FAILED\n" > "$evidence/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u "$trainer" --config="$config" --config.seed=42 \
 --config.num_epochs=10000 --config.run_name="$name" \
 --config.save_dir="$REFORMDMD_ROOT/logs/path-ratio-round1/runs" \
 --config.eval.enabled=False --config.save_freq=1000
cp "$run/config.json" "$run/metrics.jsonl" "$evidence/"
printf 'MATCHED_10000_STEP_TRAINING_COMPLETED\n' > "$evidence/status.txt"
