"""A third fixed seed only if the predeclared, time-only budget gate opens."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import shutil
import run_selected_deterministic_validation_v3 as validation

R, D, ROOT = validation.R, validation.D, validation.ROOT
progress = R/'optional-seed44-progress.json'


def record(status, **extra):
    pending = progress.with_suffix('.pending')
    pending.write_text(json.dumps(dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2))
    pending.replace(progress)


def wait_training(cases, child):
    while True:
        states = [validation.state(f'sd3-det-{case}-seed44') for case in cases]
        if all(s == 'COMPLETED_DETERMINISTIC_DEV64' for s in states):
            return True
        if 'FAILED' in states or child.poll() not in (None, 0):
            raise RuntimeError('Third-seed training failed or stopped within budget')
        if time.time() > D-1800:
            record('BUDGET_STOP_WAITING_FOR_TRAINING', cases=cases, states=states)
            return False
        time.sleep(10)


def main():
    record('WAITING_FOR_REQUIRED_TWO_SEED_VALIDATION')
    while True:
        main_state = validation.read(validation.PROGRESS).get('status', '')
        if main_state == 'ALL_PLANNED_VALIDATION_COMPLETED':
            break
        if main_state.startswith('FAILED') or main_state.startswith('BUDGET_STOP'):
            record('NOT_LAUNCHED_REQUIRED_VALIDATION_NOT_COMPLETE', main_status=main_state)
            return
        if time.time() > D-3900:
            record('NOT_LAUNCHED_TIME_ONLY_GATE_CLOSED')
            return
        time.sleep(10)
    if D-time.time() < 3900:
        record('NOT_LAUNCHED_TIME_ONLY_GATE_CLOSED')
        return
    if shutil.disk_usage(ROOT).free < 600*1024**2:
        record('NOT_LAUNCHED_PRESERVE_PERSISTENT_STORAGE_MARGIN')
        return
    selection = validation.read(R/'deterministic-selection.json')
    best, control = selection['selected_case'], selection['paired_control']
    cases = ['dmd', 'dmd_fp32', best, control]
    record('RUNNING_FIXED_THIRD_SEED', selected_case=best, controls=cases)
    # Keep the main completion record unchanged; group updates go to this file.
    validation.PROGRESS = progress
    with (Path('/rootDMD')/'path-is-seed44-replication-queue.log').open('w') as stream:
        child = subprocess.Popen(['python', '-u', 'jobs/run_matched_replication_seed_queue.py',
            '--seed', '44', '--selected', best, '--control', control], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    first_cases, second_cases = ['dmd', best], ['dmd_fp32', control]
    if not wait_training(first_cases, child):
        return
    first = validation.group(first_cases, 44, 'dataset/pickscore', 2048, 'pick2048-exact')
    if first is None:
        return
    if not wait_training(second_cases, child):
        return
    second = validation.group(second_cases, 44, 'dataset/pickscore', 2048, 'pick2048-exact')
    if second is None:
        return
    reports = [first[0], second[0], first[1], second[1]]
    validation.comparisons(reports, 'seed44')
    # The queue normally archived every model already; join before checking to
    # avoid two archivers racing on the same native FP32 directory.
    child.wait(timeout=max(1, int(D-time.time()-120)))
    if child.returncode:
        raise RuntimeError('Third-seed persistence queue failed')
    record('OPTIONAL_FIXED_SEED44_COMPLETE', selected_case=best, reports=reports)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('FAILED_REQUIRES_REPAIR', error=repr(exc))
        raise
