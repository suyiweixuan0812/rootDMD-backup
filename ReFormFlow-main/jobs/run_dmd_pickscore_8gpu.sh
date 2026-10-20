#!/usr/bin/env bash
# The selected Python must already contain requirements-dmd-runtime.txt.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${DMD_PYTHON:-python}"
cd "$task_root"
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/rootDMD/hf-cache}"
export SD3_MODEL="${SD3_MODEL:-$task_root/models/sd3-medium}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
"$task_python" - <<'PY'
import torch
count = torch.cuda.device_count()
if count != 8:
    raise SystemExit(f"This job requires exactly 8 visible GPUs; found {count}.")
print("GPU preflight passed: 8 visible GPUs.")
PY
"$task_python" scripts/check_runtime.py --method dmd --model "$SD3_MODEL" --rewards pickscore
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$task_python" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_dmd_sd3.py \
    --config=config/dmd_sd3_pickscore.py:train \
    --config.num_epochs=10000 \
    --config.train.batch_size=1 \
    --config.save_freq=1000 \
    --config.eval_freq=1000 \
    --config.save_dir="${DMD_SAVE_DIR:-$task_root/logs/dmd}" \
    --config.run_name=sd3-dmd-8gpu-10000 \
    "$@"
