#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
preset="${1:-smoke}"
[[ "$preset" == smoke || "$preset" == train ]] || { echo 'Use smoke or train' >&2; exit 2; }
if [[ $# -gt 0 ]]; then shift; fi
python scripts/check_runtime.py --method dmd --model "$SD3_MODEL" --rewards pickscore
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python -u scripts/train_rtdmd_posterior_mh_sd3.py \
    --config="config/rtdmd_posterior_mh_pickscore.py:$preset" "$@"
