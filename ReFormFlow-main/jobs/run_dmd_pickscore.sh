#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
preset="${1:-smoke}"
if [[ "$preset" != smoke && "$preset" != train ]]; then
    echo "Usage: bash jobs/run_dmd_pickscore.sh [smoke|train] [config overrides...]" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
python scripts/check_runtime.py --method dmd --model "$SD3_MODEL" --rewards pickscore
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python -u scripts/train_dmd_sd3.py --config "config/dmd_sd3_pickscore.py:$preset" "$@"
