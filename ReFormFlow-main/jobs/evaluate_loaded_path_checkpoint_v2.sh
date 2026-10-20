#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$REFORMDMD_ROOT"
bash jobs/evaluate_loaded_path_checkpoint.sh "$@"
python scripts/audit_loaded_checkpoint_evaluation.py --run-dir "logs/path-ratio-round1/runs/$2"
