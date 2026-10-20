#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
: "${HF_HOME:?Set HF_HOME to the mounted PickScore cache}"
: "${SD3_MODEL:?Set SD3_MODEL to the mounted SD3-Medium directory}"
: "${DMD_SAVE_DIR:?Set DMD_SAVE_DIR to persistent output storage}"
task_runtime="${DMD_RUNTIME_DIR:-/tmp/dmd-runtime}"
python -m venv --system-site-packages "$task_runtime"
"$task_runtime/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
    --index-url https://pypi.org/simple -r requirements-dmd-runtime.txt
export DMD_PYTHON="$task_runtime/bin/python"
exec bash "$task_root/jobs/run_dmd_pickscore_8gpu.sh"
