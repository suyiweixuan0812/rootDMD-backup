"""Adopt existing unchanged evaluations until the user's eight-hour cutoff.

Only replace their timeout monitors, preserving the live evaluator processes,
RNG streams, checkpoints, prompts, batch size and scoring code. Terminate the
owned process groups before01:20UTC even when the full pair is incomplete.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import signal
import time
import run_selected_deterministic_validation_v3 as v

R = v.R
DEADLINE = datetime(2026, 10, 21, 1, 19, 30, tzinfo=timezone.utc).timestamp()
NAMES = [f'sd3-det-{c}-seed42-3000-confirm-pick2048-exact' for c in ['dmd', 'last']]
PROGRESS = R/'user-budget-final-evaluation-progress.json'
stopping = [False]
for sig in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(sig, lambda *_: stopping.__setitem__(0, True))


def save(status, **extra):
    p = PROGRESS.with_suffix('.pending')
    p.write_text(json.dumps(dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2))
    p.replace(PROGRESS)


def kill_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def main():
    monitors = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not args or Path(os.fsdecode(args[0])).name != 'timeout':
            continue
        for name in NAMES:
            if name.encode() in args:
                pid = int(directory.name)
                assert os.getpgid(pid) == pid
                assert b'jobs/evaluate_loaded_path_checkpoint_no_online_v2.sh' in args
                monitors[name] = pid
    assert set(monitors) == set(NAMES), monitors
    assert time.time() < DEADLINE
    save('WATCHDOG_ACTIVE_BEFORE_MONITOR_REPLACEMENT', monitors=monitors,
         hard_gpu_stop_utc='2026-10-21T01:19:30+00:00',
         original_user_budget_end_utc='2026-10-21T01:20:00+00:00',
         reason='Use final approved GPU minutes; CPU paper packaging can follow GPU stop.',
         scientific_changes=False,
         preserved='Same live evaluator PIDs/RNG, native3000EMA, all2048 prompts, batch4 and four original scorers')
    adopted = False
    try:
        adopted = True
        for pid in monitors.values():
            os.kill(pid, signal.SIGKILL)
        save('RUNNING_UNCHANGED_EVALUATIONS_WITH_USER_BUDGET', monitors=monitors)
        while time.time() < DEADLINE and not stopping[0]:
            if all(v.state(name) == 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION' for name in NAMES):
                v.compare(NAMES[0], [NAMES[1]], 'comparison-optional-faithful-long3000')
                result = dict(status='OPTIONAL_FAITHFUL_LONG_CHECK_COMPLETE',
                    utc=datetime.now(timezone.utc).isoformat(), faithful_case='last',
                    reports=NAMES, training=[n.removesuffix('-pick2048-exact') for n in NAMES],
                    archived=[n.removesuffix('-pick2048-exact') for n in NAMES],
                    prefix_audit='deterministic-full-prefix-reproduction.json',
                    evaluator='jobs/evaluate_loaded_path_checkpoint_no_online_v2.sh',
                    evaluation_window='Existing unchanged tests adopted within original8hour user budget; no regeneration or parameter changes')
                target=R/'optional-faithful-long-progress.json'
                p=target.with_suffix('.pending');p.write_text(json.dumps(result,indent=2));p.replace(target)
                save('BOTH_FULL_TESTS_COMPLETED_WITHIN_USER_BUDGET', reports=NAMES)
                return
            time.sleep(2)
        save('INCOMPLETE_AT_USER_BUDGET', reports={name:v.state(name) for name in NAMES},
             quality_claim='No long-quality comparison without both full2048 four-score reports')
    finally:
        # A replaced monitor must never leave an unbounded GPU evaluator.
        if adopted:
            for pid in monitors.values():
                kill_group(pid, signal.SIGTERM)
            time.sleep(2)
            for pid in monitors.values():
                kill_group(pid, signal.SIGKILL)


if __name__ == '__main__':
    main()
