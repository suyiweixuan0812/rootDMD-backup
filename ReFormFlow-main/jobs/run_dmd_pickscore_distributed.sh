#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${DMD_PYTHON:-python}"
cd "$task_root"
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
task_nodes="${DMD_NNODES:-2}"
task_gpus="${DMD_GPUS_PER_NODE:-8}"
task_rank="${MLP_ROLE_INDEX:-${NODE_RANK:-}}"
task_master="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-}}"
task_port="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29500}}"
task_config="${DMD_CONFIG:-config/dmd_sd3_pickscore.py:train}"
task_expected="${DMD_EVAL_EXPECTED_COUNT:-2048}"
: "${task_rank:?The platform must provide MLP_ROLE_INDEX or NODE_RANK}"
: "${task_master:?The platform must provide MLP_WORKER_0_HOST or MASTER_ADDR}"
if [[ -n "${MLP_WORKER_NUM:-}" && "$MLP_WORKER_NUM" != "$task_nodes" ]]; then
    echo "Expected $task_nodes worker nodes, found $MLP_WORKER_NUM" >&2
    exit 1
fi
"$task_python" - "$task_gpus" <<'PY'
import sys, torch
expected = int(sys.argv[1])
count = torch.cuda.device_count()
if count != expected:
    raise SystemExit(f"Expected {expected} visible GPUs per node; found {count}")
print(f"GPU preflight passed: {count} visible GPUs per node.")
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
launch=("$task_python" -u -m torch.distributed.run
    --nnodes="$task_nodes" --nproc_per_node="$task_gpus"
    --node_rank="$task_rank" --master_addr="$task_master" --master_port="$task_port")
echo "Starting DMD: nodes=$task_nodes GPUs_per_node=$task_gpus node_rank=$task_rank"
"${launch[@]}" scripts/train_dmd_sd3.py \
    --config="$task_config" \
    --config.num_epochs=10000 --config.train.batch_size=1 \
    --config.save_freq=1000 --config.eval_freq=1000 \
    --config.save_dir="$DMD_SAVE_DIR" --config.run_name="$DMD_RUN_NAME" "$@"
exec "${launch[@]}" scripts/score_dmd_images.py \
    --run-dir "$DMD_SAVE_DIR/$DMD_RUN_NAME" --step 10000 \
    --expected-count "$task_expected" --batch-size 8
