"""Optional fixed 3000-step Teacher-path check, gated only by remaining time."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_path_round_delivery import ENTRIES
import run_selected_deterministic_validation_v3 as validation

R, D = validation.R, validation.D
progress = R/'optional-faithful-long-progress.json'


def record(status, **extra):
    pending = progress.with_suffix('.pending')
    pending.write_text(json.dumps(dict(status=status, utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2))
    pending.replace(progress)


def train(case):
    name = f'sd3-det-{case}-seed42-3000-confirm'
    run = R/'runs'/name
    assert not run.exists(), 'Never overwrite an optional training run'
    if case == 'dmd':
        trainer, config, extra = 'scripts/train_dmd_sd3.py', 'config/dmd_sd3_pickscore.py:train', []
    else:
        trainer, config, extra = ENTRIES[case]
    command = ['timeout', '--signal=TERM', '--kill-after=30s', str(int(D-time.time()-180)),
        'python', '-u', 'scripts/run_deterministic_training.py', trainer, '--config='+config,
        '--config.run_name='+name, '--config.seed=42', '--config.num_epochs=3000',
        '--config.save_dir='+str(R/'runs'), '--config.eval.enabled=False', '--config.save_freq=1000', *extra]
    with (Path('/rootDMD')/f'path-is-{name}.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    evidence = R/'evidence'/name
    evidence.mkdir(exist_ok=True)
    for file in ['config.json', 'metrics.jsonl', 'runtime-numerics.json']:
        shutil.copy2(run/file, evidence/file)
    (evidence/'status.txt').write_text('OPTIONAL_MATCHED_3000_TRAINING_COMPLETED\n')
    return name


def audit_prefix(case, name):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    source = R/'archived-checkpoints'/f'sd3-det-{case}-seed42-ema1000/lora/generator/adapter_model.safetensors'
    repeated = R/'runs'/name/'checkpoints/checkpoint-1000/lora/generator/adapter_model.safetensors'
    a, b = load_file(str(source)), load_file(str(repeated))
    assert a.keys() == b.keys()
    equal = sum(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)
    return dict(case=case, original=str(source), repeated=str(repeated), tensor_count=len(a),
        equal_tensors=equal, all_bit_exact=equal==len(a),
        max_abs_difference=max(float((a[k].double()-b[k].double()).abs().max()) for k in a),
        scope='Original FP32 EMA at1000 outer steps; report mismatch rather than assume full training reproduction')


def evaluate(name):
    output = name+'-pick2048-exact'
    if D-time.time() < 1500:
        return None
    checkpoint = R/'runs'/name/'checkpoints/checkpoint-3000/lora/generator'
    with (Path('/rootDMD')/f'path-is-{output}.log').open('w') as stream:
        subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s', str(int(D-time.time()-180)),
            'bash', 'jobs/evaluate_loaded_path_checkpoint_v2.sh', str(checkpoint), output,
            'dataset/pickscore', '2048', name], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return output


def main():
    record('WAITING_FOR_REQUIRED_TWO_SEED_VALIDATION')
    while True:
        main = validation.read(validation.PROGRESS)
        state = main.get('status', '')
        if state.startswith('FAILED') or state.startswith('BUDGET_STOP'):
            record('NOT_LAUNCHED_REQUIRED_VALIDATION_NOT_COMPLETE', main_status=state)
            return
        # The one training slot can be reused after ALL required seed43
        # checkpoints have been persisted, even while seed42 tests finish.
        # Never overlap the required training queue or exceed two evaluators
        # plus one trainer. This trigger depends on progress/time, not scores.
        plan = validation.read(R/'deterministic-validation-plan.json')
        replication = validation.read(R/'deterministic-replication-progress.json')
        required = plan.get('seed43_replication_cases', [])
        training_ready = (bool(required) and set(required) <= set(replication.get('completed', []))
            and not replication.get('running') and not replication.get('queued')
            and not replication.get('failed'))
        ready = (state == 'ALL_PLANNED_VALIDATION_COMPLETED' or
                 (state == 'RUNNING_PARALLEL_EVALUATION' and main.get('seed') == 43
                  and main.get('count') == 2048) or training_ready)
        if ready:
            break
        if time.time() > D-4800:
            record('NOT_LAUNCHED_TIME_ONLY_GATE_CLOSED')
            return
        time.sleep(10)
    if D-time.time() < 4800:
        record('NOT_LAUNCHED_TIME_ONLY_GATE_CLOSED')
        return
    choice = validation.read(R/'deterministic-selection.json')
    case = choice['pure_teacher_alternative']['case']
    record('RUNNING_FRESH_MATCHED_3000_TRAINING', faithful_case=case)
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(train, c) for c in ['dmd', case]]
        names = [future.result() for future in futures]
    prefix = [audit_prefix(c, name) for c, name in zip(['dmd', case], names)]
    (R/'deterministic-full-prefix-reproduction.json').write_text(json.dumps(prefix, indent=2))
    archives = []
    for c, name in [(case, names[1]), ('dmd', names[0])]:
        source = R/'runs'/name/'checkpoints/checkpoint-3000/lora/generator/adapter_model.safetensors'
        if shutil.disk_usage(ROOT).free > source.stat().st_size+256*1024**2:
            validation.execute(['python', 'scripts/persist_path_round_checkpoint.py', '--training-run', name,
                '--step', '3000', '--name', name+'-ema3000'])
            archives.append(name)
    validation.execute(['python', 'scripts/audit_path_training_parameters.py', '--baseline',
        str(R/'runs'/names[0]/'config.json'), '--candidate', str(R/'runs'/names[1]/'config.json'),
        '--output', str(R/'parameter-audit-optional-faithful-long.json')])
    record('WAITING_FOR_MAIN_TEST_BEFORE_LONG_EVALUATION', faithful_case=case, training=names)
    while True:
        main_status = validation.read(validation.PROGRESS).get('status', '')
        if main_status == 'ALL_PLANNED_VALIDATION_COMPLETED':
            break
        if main_status.startswith('FAILED') or main_status.startswith('BUDGET_STOP'):
            record('BUDGET_STOP_MAIN_VALIDATION_NEEDS_ATTENTION', main_status=main_status, training=names, archived=archives)
            return
        if time.time() > D-1500:
            record('BUDGET_STOP_BEFORE_LONG_TEST', training=names, archived=archives)
            return
        time.sleep(10)
    record('RUNNING_FIXED_3000_PICK2048_EVALUATION', faithful_case=case, training=names)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(evaluate, names))
    if any(name is None for name in reports):
        record('BUDGET_STOP_BEFORE_COMPLETE_LONG_TEST', training=names, archived=archives)
        return
    validation.compare(reports[0], [reports[1]], 'comparison-optional-faithful-long3000')
    record('OPTIONAL_FAITHFUL_LONG_CHECK_COMPLETE', faithful_case=case, reports=reports,
           archived=archives, prefix_audit='deterministic-full-prefix-reproduction.json')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('FAILED_REQUIRES_REPAIR', error=repr(exc))
        raise
