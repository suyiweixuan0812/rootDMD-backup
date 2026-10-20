"""Replicate the frozen method and controls on seed43, at most two trainers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--selected', required=True)
p.add_argument('--control', required=True)
p.add_argument('--extra-case', action='append', default=[])
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
r = root/'logs/path-ratio-round1'
deadline = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()
queue = list(dict.fromkeys(['dmd', a.selected, 'dmd_fp32', a.control, *a.extra_case]))
running, completed, failed = {}, [], []


class ExistingTrial:
    """Adopt a valid orphaned trial without replaying its optimizer steps."""
    def __init__(self, case): self.case = case
    def poll(self):
        path = r/'evidence'/f'sd3-det-{self.case}-seed43'/'status.txt'
        status = path.read_text().strip() if path.exists() else ''
        return 0 if status == 'COMPLETED_DETERMINISTIC_DEV64' else 1 if status == 'FAILED' else None


for case in list(queue):
    path = r/'evidence'/f'sd3-det-{case}-seed43'/'status.txt'
    if path.exists() and path.read_text().strip() == 'DETERMINISTIC_TRAINING_AND_DEV_SCORING':
        running[case] = ExistingTrial(case)
        queue.remove(case)


def record():
    path = r/'deterministic-replication-progress.json'
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(dict(queued=queue, running=list(running), completed=completed,
        failed=failed, max_parallel_trainers=1, utc=datetime.now(timezone.utc).isoformat()), indent=2))
    pending.replace(path)


while queue or running:
    for case, child in list(running.items()):
        code = child.poll()
        if code is None:
            continue
        del running[case]
        if code:
            failed.append(dict(case=case, returncode=code))
        else:
            archived = r/'archived-checkpoints'/f'sd3-det-{case}-seed43-ema1000/storage-manifest.json'
            if not archived.exists():
                subprocess.run(['python', 'scripts/persist_path_round_checkpoint.py', '--training-run',
                    f'sd3-det-{case}-seed43', '--name', f'sd3-det-{case}-seed43-ema1000'], cwd=root, check=True)
            completed.append(case)
    if failed:
        record()
        raise SystemExit(1)
    while queue and len(running) < 1:
        if deadline-time.time() < 2700:
            failed.extend(dict(case=case, reason='not launched: remaining training budget') for case in queue)
            queue.clear()
            break
        case = queue.pop(0)
        status = r/'evidence'/f'sd3-det-{case}-seed43'/'status.txt'
        if status.exists() and status.read_text().strip() == 'COMPLETED_DETERMINISTIC_DEV64':
            completed.append(case)
            continue
        with (Path('/rootDMD')/f'path-is-det-{case}-seed43.log').open('w') as stream:
            launcher = 'jobs/run_deterministic_comparison_trial_v4.sh' if case in {'last_q2_time','paired_q2_time'} else 'jobs/run_deterministic_comparison_trial_v3.sh'
            running[case] = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30s',
                str(int(deadline-time.time()-180)), 'bash',
                launcher, case, '43'], cwd=root,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record()
    if running:
        time.sleep(5)
record()
if failed:
    raise SystemExit(1)
