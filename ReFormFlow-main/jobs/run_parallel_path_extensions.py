"""Run the four already frozen independent extensions on one A100.

While the first cohort still trains, allow two extra jobs; after it finishes,
allow three. This caps total development models at three. Method entry points,
launchers, seeds, numerical settings and scoring computations are unchanged.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'logs/path-ratio-round1'
D = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()
cases = ['adaptive_bridge', 'last', 'last_time', 'adaptive_bridge_time']
queue = list(cases)
running, completed, failures = {}, [], []


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):return {}


def write(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def record():
    for prefix, subset, done, file in [
            ('adaptive', ['adaptive_bridge', 'adaptive_bridge_time'], 'BOTH_ADAPTIVE_DEV_TRIALS_COMPLETED', 'adaptive-extension-progress.json'),
            ('last', ['last', 'last_time'], 'BOTH_LAST_INNOVATION_DEV_TRIALS_COMPLETED', 'last-innovation-extension-progress.json')]:
        failed = [x for x in failures if x['case'] in subset]
        if failed:
            value = dict(status='CASE_FAILED', failures=failed)
        elif all(c in completed for c in subset):
            value = dict(status=done)
        else:
            active = next((c for c in running if c in subset), None)
            value = dict(status='RUNNING' if active else 'QUEUED_PARALLEL_EXTENSION', current_case=active)
        value['orchestration'] = 'parallel-development-orchestration-revision.json'
        write(R/file, value)
    write(R/'parallel-extension-progress.json', dict(
        status='FAILED_REQUIRES_REPAIR' if failures else 'ALL_FOUR_EXTENSIONS_COMPLETED' if not queue and not running else 'RUNNING',
        running=list(running), queued=queue, completed=completed, failures=failures,
        utc=datetime.now(timezone.utc).isoformat()))


while queue or running:
    for case, child in list(running.items()):
        code = child.poll()
        if code is None:continue
        del running[case]
        if code:failures.append(dict(case=case, returncode=code))
        else:completed.append(case)
    first = read(R/'deterministic-cohort-progress.json')
    if first.get('status') == 'CASE_FAILED':
        raise RuntimeError('First cohort requires repair')
    limit = 3 if first.get('status') == 'ALL_FACTORIAL_AND_CONTROL_DEV_TRIALS_COMPLETED' else 2
    while queue and len(running) < limit and not failures:
        if D-time.time() < 1800:
            failures.extend(dict(case=c, reason='budget stop before launch') for c in queue)
            queue.clear();break
        case = queue.pop(0)
        status = R/'evidence'/f'sd3-det-{case}-seed42'/'status.txt'
        if status.exists() and status.read_text().strip() == 'COMPLETED_DETERMINISTIC_DEV64':
            completed.append(case);continue
        launcher = ('jobs/run_deterministic_comparison_trial_v2.sh' if case.startswith('adaptive') else
                    'jobs/run_deterministic_comparison_trial_v3.sh')
        with (Path('/rootDMD')/f'path-is-det-{case}-seed42.log').open('w') as stream:
            running[case] = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30s',
                str(int(D-time.time()-120)), 'bash', launcher, case, '42'], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record()
    if running:time.sleep(5)
    elif failures:break
record()
if failures:raise SystemExit(1)
