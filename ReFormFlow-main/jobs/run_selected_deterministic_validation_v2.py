"""One bounded validation round after all ten predeclared Dev64 experiments.

This is a dependency job, not a recurring automation. Test scores never feed
back into the Dev64 selector. Every model is evaluated without an EMA update.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'logs/path-ratio-round1'
DEADLINE = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()
PROGRESS = ROUND / 'selected-deterministic-validation-progress.json'
children = []


def record(status, **kw):
    pending = PROGRESS.with_suffix('.pending')
    pending.write_text(json.dumps(dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **kw), indent=2))
    pending.replace(PROGRESS)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run(command, name, minimum=1800):
    remaining = int(DEADLINE - time.time() - 180)
    if remaining < minimum:
        record('BUDGET_STOP_BEFORE_NEW_TASK', next_task=name)
        return False
    record('RUNNING', task=name)
    with (Path('/rootDMD') / f'path-is-{name}.log').open('w') as stream:
        subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', str(remaining), *command],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return True


def persist(case, seed):
    name = f'sd3-det-{case}-seed{seed}'
    archived = f'{name}-ema1000'
    if (ROUND / 'archived-checkpoints' / archived / 'storage-manifest.json').exists():
        return
    subprocess.run(['python', 'scripts/persist_path_round_checkpoint.py', '--training-run', name,
                    '--name', archived], cwd=ROOT, check=True)


def evaluate(case, seed, dataset, count, suffix):
    training = f'sd3-det-{case}-seed{seed}'
    name = f'{training}-{suffix}'
    evidence = ROUND / 'evidence' / name
    status = evidence / 'status.txt'
    if status.exists() and status.read_text().strip() == 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION':
        return name
    checkpoint = ROUND / 'runs' / training / 'checkpoints/checkpoint-1000/lora/generator'
    ok = run(['bash', 'jobs/evaluate_loaded_path_checkpoint_v2.sh', str(checkpoint), name,
              dataset, str(count), f'Deterministic-{case}-seed{seed}'], name,
             minimum=1500 if count == 2048 else 420)
    return name if ok else None


def compare(base, candidates, out):
    cmd = ['python', 'scripts/report_path_ratio_round.py', '--baseline', str(ROUND / 'evidence' / base)]
    for name in candidates:
        cmd += ['--candidate', str(ROUND / 'evidence' / name)]
    cmd += ['--output', str(ROUND / out)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    record('WAITING_FOR_ALL_TWELVE_PREDECLARED_DEV_TRIALS')
    while True:
        a = read_json(ROUND / 'last-innovation-extension-progress.json')
        b = read_json(ROUND / 'deterministic-cohort-progress.json')
        if a.get('status') == 'BOTH_LAST_INNOVATION_DEV_TRIALS_COMPLETED':
            break
        if a.get('status') == 'CASE_FAILED' or b.get('status') == 'CASE_FAILED':
            raise RuntimeError('Dev cohort requires repair; selection not performed')
        if time.time() > DEADLINE - 3600:
            record('BUDGET_STOP_WAITING_FOR_COMPLETE_DEV_COHORT')
            return
        time.sleep(10)
    selected = ROUND / 'deterministic-selection.json'
    if not selected.exists():
        subprocess.run(['python', 'scripts/select_deterministic_path_candidate_v2.py'], cwd=ROOT, check=True)
    choice = read_json(selected)
    best = choice['selected_case']
    control = choice['paired_control']
    cases = ['dmd', 'dmd_fp32', best, control]
    # Freeze the replication plan before any new candidate test/holdout score.
    plan = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), selected_case=best,
                selected_file_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
                seed42_full_and_holdout_cases=cases, seed43_replication_cases=['dmd', 'dmd_fp32', best],
                seed43_selection_role='None: independent training replication of the already selected method',
                heldout_role='Reserved 138 unique DrawBench prompts; never used to select or retune',
                deadline_utc='2026-10-21T01:20:00+00:00')
    plan_path = ROUND / 'deterministic-validation-plan.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2))
    for case in cases:
        persist(case, 42)
    # Three training models plus one evaluation pipeline fit the inspected A100.
    # Each child is separately bounded, so there is no unbounded orphan job.
    if DEADLINE - time.time() > 5400:
        for case in ['dmd', 'dmd_fp32', best]:
            status = ROUND / 'evidence' / f'sd3-det-{case}-seed43' / 'status.txt'
            if status.exists() and status.read_text().strip() == 'COMPLETED_DETERMINISTIC_DEV64':
                continue
            log = (Path('/rootDMD') / f'path-is-det-{case}-seed43.log').open('w')
            command = ['timeout', '--signal=TERM', '--kill-after=30s', str(int(DEADLINE-time.time()-180)),
                       'bash', 'jobs/run_deterministic_comparison_trial_v3.sh', case, '43']
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            log.close()
            children.append((case, child))
    reports = []
    for case in cases:
        result = evaluate(case, 42, 'dataset/pickscore', 2048, 'pick2048-exact')
        if result is None:
            return
        reports.append(result)
    compare(reports[0], reports[1:], 'comparison-deterministic-seed42')
    compare(reports[1], reports[2:], 'comparison-deterministic-fp32-seed42')
    compare(reports[3], [reports[2]], 'comparison-deterministic-particle-vs-paired')
    heldout = []
    for case in cases:
        result = evaluate(case, 42, 'dataset/drawbench_path_holdout', 138, 'holdout138-exact')
        if result is None:
            return
        heldout.append(result)
    compare(heldout[0], heldout[1:], 'comparison-deterministic-holdout138')
    replication = []
    for case in ['dmd', 'dmd_fp32', best]:
        status = ROUND / 'evidence' / f'sd3-det-{case}-seed43' / 'status.txt'
        while not status.exists() or status.read_text().strip() != 'COMPLETED_DETERMINISTIC_DEV64':
            if status.exists() and status.read_text().strip() == 'FAILED':
                raise RuntimeError(f'Seed43 training failed: {case}')
            if time.time() > DEADLINE - 1800:
                record('BUDGET_STOP_BEFORE_COMPLETE_SEED43_TEST', case=case)
                return
            if not children:
                record('SEED43_NOT_LAUNCHED_BUDGET', case=case)
                return
            time.sleep(10)
        persist(case, 43)
        result = evaluate(case, 43, 'dataset/pickscore', 2048, 'pick2048-exact')
        if result is None:
            return
        replication.append(result)
    compare(replication[0], replication[1:], 'comparison-deterministic-seed43')
    compare(replication[1], replication[2:], 'comparison-deterministic-fp32-seed43')
    record('ALL_PLANNED_VALIDATION_COMPLETED', selected_case=best,
           seed42_reports=reports, holdout_reports=heldout, seed43_reports=replication)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('FAILED_REQUIRES_REPAIR', error=repr(exc))
        raise
