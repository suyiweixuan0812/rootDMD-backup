#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
: "${HF_HOME:?Set HF_HOME to the mounted PickScore cache}"
: "${SD3_MODEL:?Set SD3_MODEL to the mounted SD3-Medium directory}"
: "${DMD_SAVE_DIR:?Set DMD_SAVE_DIR to persistent output storage}"
: "${REWARD_CKPT_PATH:?Set REWARD_CKPT_PATH to the mounted evaluation weights}"
task_runtime="${DMD_RUNTIME_DIR:-/tmp/dmd-runtime}"
python -m venv --system-site-packages "$task_runtime"
if [[ -n "${DMD_WHEELHOUSE:-}" ]]; then
    "$task_runtime/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
        --no-index --find-links "$DMD_WHEELHOUSE" -r requirements-dmd-eval.txt
else
    "$task_runtime/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
        --index-url https://pypi.org/simple -r requirements-dmd-eval.txt
fi
export DMD_PYTHON="$task_runtime/bin/python"
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
"$DMD_PYTHON" scripts/check_runtime.py --method dmd --model "$SD3_MODEL" \
    --rewards pickscore clipscore aesthetic hpsv2
if [[ "${DMD_NNODES:-1}" -gt 1 ]]; then
    exec bash "$task_root/jobs/run_dmd_pickscore_distributed.sh"
fi
bash "$task_root/jobs/run_dmd_pickscore_8gpu.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$DMD_PYTHON" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/score_dmd_images.py \
    --run-dir "$DMD_SAVE_DIR/sd3-dmd-8gpu-10000" --step 10000 \
    --expected-count 2048 --batch-size 8
