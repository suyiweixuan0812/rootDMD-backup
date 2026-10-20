#!/usr/bin/env bash
# Two independent, prespecified experiments share one 80GB A100. No agent delegation.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$root/jobs/run_rtdmd_pcmh_trial.sh" 025 > /rootDMD/rtdmd-pcmh-lambda025.log 2>&1 &
first=$!
bash "$root/jobs/run_rtdmd_pcmh_trial.sh" 050 > /rootDMD/rtdmd-pcmh-lambda050.log 2>&1 &
second=$!
first_status=0
second_status=0
wait "$first" || first_status=$?
wait "$second" || second_status=$?
printf 'lambda025_exit=%s lambda050_exit=%s\n' "$first_status" "$second_status"
[[ "$first_status" == 0 && "$second_status" == 0 ]]
