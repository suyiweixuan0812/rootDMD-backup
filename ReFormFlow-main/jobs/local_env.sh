#!/usr/bin/env bash
# Source this file before running the project; no packages are installed here.
export REFORMDMD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export REFORMDMD_ENV="${REFORMDMD_ENV:-/rootDMD/.venv-dmd}"
export PATH="$REFORMDMD_ENV/bin:$PATH"
export PYTHONPATH="$REFORMDMD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/rootDMD/hf-cache}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export SD3_MODEL="${SD3_MODEL:-$REFORMDMD_ROOT/models/sd3-medium}"
