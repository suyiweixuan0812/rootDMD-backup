"""Frozen Dev64 selection, two parallel evaluators and two replication trainers.

Inference remains the unchanged checkpoint wrapper and batch/seed protocol.
This one-shot job ends within the user's eight-hour round.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'logs/path-ratio-round1'
D = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()
PROGRESS = R/'selected-deterministic-validation-progress.json'


def record(status, **extra):
    pending = PROGRESS.with_suffix('.pending')
    pending.write_text(json.dumps(dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2))
    pending.replace(PROGRESS)


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def state(name):
    p = R/'evidence'/name/'status.txt'
    return p.read_text().strip() if p.exists() else ''


def execute(command):
    subprocess.run(command, cwd=ROOT, check=True)


def archive_weights(case, seed):
    name = f'sd3-det-{case}-seed{seed}'
    if not (R/'archived-checkpoints'/f'{name}-ema1000'/'storage-manifest.json').exists():
        execute(['python', 'scripts/persist_path_round_checkpoint.py', '--training-run', name,
                 '--name', name+'-ema1000'])


def evaluate(case, seed, dataset, count, suffix):
    training = f'sd3-det-{case}-seed{seed}'
    name = training+'-'+suffix
    if state(name) == 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION':
        return name
    remaining = int(D-time.time()-180)
    if remaining < (1800 if count == 2048 else 420):
        return None
    checkpoint = R/'runs'/training/'checkpoints/checkpoint-1000/lora/generator'
    with (Path('/rootDMD')/f'path-is-{name}.log').open('w') as stream:
        subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', str(remaining),
            'bash', 'jobs/evaluate_loaded_path_checkpoint_v2.sh', str(checkpoint), name,
            dataset, str(count), f'Deterministic-{case}-seed{seed}'], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    return name


def group(cases, seed, dataset, count, suffix):
    record('RUNNING_PARALLEL_EVALUATION', cases=cases, seed=seed, dataset=dataset, count=count,
           max_parallel_evaluators=2, max_parallel_trainers=2)
    results = {}
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = {pool.submit(evaluate, c, seed, dataset, count, suffix): c for c in cases}
        for future in as_completed(work):
            case = work[future]
            try:
                results[case] = future.result()
            except Exception as exc:
                errors.append((case, repr(exc)))
    if errors:
        raise RuntimeError(errors)
    if any(results.get(c) is None for c in cases):
        record('BUDGET_STOP_BEFORE_NEW_EVALUATION', incomplete=[c for c in cases if results.get(c) is None])
        return None
    return [results[c] for c in cases]


def compare(base, candidates, output):
    command = ['python', 'scripts/report_path_ratio_round.py', '--baseline', str(R/'evidence'/base)]
    for name in candidates:
        command += ['--candidate', str(R/'evidence'/name)]
    command += ['--output', str(R/output)]
    execute(command)


def comparisons(names, suffix):
    compare(names[0], names[1:], 'comparison-deterministic-'+suffix)
    compare(names[1], names[2:], 'comparison-deterministic-fp32-'+suffix)
    compare(names[3], [names[2]], 'comparison-deterministic-particle-vs-paired-'+suffix)


def main():
    record('WAITING_FOR_ALL_TWELVE_PREDECLARED_DEV_TRIALS')
    while True:
        statuses = [read(R/p).get('status') for p in ['deterministic-cohort-progress.json',
                    'adaptive-extension-progress.json', 'last-innovation-extension-progress.json']]
        if statuses == ['ALL_FACTORIAL_AND_CONTROL_DEV_TRIALS_COMPLETED',
                        'BOTH_ADAPTIVE_DEV_TRIALS_COMPLETED', 'BOTH_LAST_INNOVATION_DEV_TRIALS_COMPLETED']:
            break
        if 'CASE_FAILED' in statuses:
            raise RuntimeError('Dev cohort requires repair before any selection')
        if time.time() > D-3600:
            record('BUDGET_STOP_WAITING_FOR_COMPLETE_DEV_COHORT')
            return
        time.sleep(10)
    selection = R/'deterministic-selection.json'
    if not selection.exists():
        execute(['python', 'scripts/select_deterministic_path_candidate_v2.py'])
    choice = read(selection)
    best, control = choice['selected_case'], choice['paired_control']
    cases = ['dmd', 'dmd_fp32', best, control]
    plan = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), selected_case=best,
        selected_file_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        seed42_full_and_holdout_cases=cases, seed43_replication_cases=cases,
        seed43_selection_role='Independent replication only; not selection or tuning',
        max_parallel_training=2, max_parallel_evaluation=2,
        inference_protocol='Unchanged eval wrapper, batch, prompts, seed, NFE and scorers',
        heldout_role='138 reserved unique DrawBench prompts, never used to retune',
        deadline_utc='2026-10-21T01:20:00+00:00')
    plan_path = R/'deterministic-validation-plan.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2))
    for case in cases:
        archive_weights(case, 42)
    replication = None
    if D-time.time() > 5400:
        with (Path('/rootDMD')/'path-is-deterministic-replication-queue.log').open('w') as stream:
            replication = subprocess.Popen(['python', '-u', 'jobs/run_deterministic_replication_queue.py',
                '--selected', best, '--control', control], cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
    tests = group(cases, 42, 'dataset/pickscore', 2048, 'pick2048-exact')
    if tests is None:
        return
    comparisons(tests, 'seed42')
    for case in ['dmd', best]:
        name = f'sd3-det-{case}-seed42-pick2048-exact'
        if not (R/'archived-images'/name/'image-storage-manifest.json').exists():
            execute(['python', 'scripts/archive_path_round_images.py', '--run-dir', str(R/'runs'/name), '--name', name])
    heldout = group(cases, 42, 'dataset/drawbench_path_holdout', 138, 'holdout138-exact')
    if heldout is None:
        return
    comparisons(heldout, 'holdout138')
    if replication is None:
        record('SEED42_AND_HOLDOUT_COMPLETE_SEED43_NOT_LAUNCHED_BUDGET', tests=tests, heldout=heldout)
        return
    while True:
        states = [state(f'sd3-det-{case}-seed43') for case in cases]
        if all(s == 'COMPLETED_DETERMINISTIC_DEV64' for s in states):
            break
        if 'FAILED' in states or replication.poll() not in (None, 0):
            raise RuntimeError('Replication training failed or stopped within budget')
        if time.time() > D-1800:
            record('BUDGET_STOP_WAITING_FOR_SEED43_TEST', completed_states=dict(zip(cases, states)))
            return
        time.sleep(10)
    for case in cases:
        archive_weights(case, 43)
    repeated = group(cases, 43, 'dataset/pickscore', 2048, 'pick2048-exact')
    if repeated is None:
        return
    comparisons(repeated, 'seed43')
    record('ALL_PLANNED_VALIDATION_COMPLETED', selected_case=best,
           seed42_reports=tests, holdout_reports=heldout, seed43_reports=repeated)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('FAILED_REQUIRES_REPAIR', error=repr(exc))
        raise
