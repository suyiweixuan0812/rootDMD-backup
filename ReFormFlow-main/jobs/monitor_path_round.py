"""Bounded read-only progress stream for the authorized eight-hour round."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
r = root/'logs/path-ratio-round1'
deadline = datetime(2026, 10, 21, 1, 20, tzinfo=timezone.utc).timestamp()


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def tail_row(path):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        pass
    return {}


seen = {p.parent.name for p in (r/'evidence').glob('*/status.txt')
        if p.read_text().strip().startswith('COMPLETED')}
last_alert = None
while time.time() < deadline:
    stamp = datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
    q = read(r/'deterministic-cohort-progress.json')
    a = read(r/'adaptive-extension-progress.json')
    last = read(r/'last-innovation-extension-progress.json')
    active = (q.get('current_case') if q.get('status') == 'RUNNING' else
              a.get('current_case') if a.get('status') == 'RUNNING' else
              last.get('current_case') if last.get('status') == 'RUNNING' else None)
    parts = [stamp]
    if active:
        row = tail_row(r/'runs'/f'sd3-det-{active}-seed42'/'metrics.jsonl')
        parts.append(f'Dev {active}: {row.get("step", "loading")}/1000')
    for name in ['dmd-seed42-10000-pick2048', 'path-is-seed42-10000-pick2048']:
        status = r/'evidence'/name/'status.txt'
        if status.exists() and not status.read_text().strip().startswith('COMPLETED'):
            images = r/'runs'/name/'eval/step-1/rank-0'
            parts.append(name+': '+str(len(list(images.glob('*.png'))))+'/2048 PNGs')
    main = read(r/'selected-deterministic-validation-progress.json')
    parts.append('validation='+str(main.get('status')))
    resources = tail_row(root.parent/'development-load-guard/resource-monitor.jsonl')
    if resources:
        parts.append(f'CPU/GPU rolling {resources.get("mean_last12_cpu",0):.0f}%/{resources.get("mean_last12_gpu",0):.0f}%')
    print(' | '.join(parts), flush=True)
    for p in (r/'evidence').glob('*/status.txt'):
        if p.parent.name in seen or not p.read_text().strip().startswith('COMPLETED'):
            continue
        report = read(p.parent/'final-metrics.json')
        if report:
            print('NEW_COMPLETED '+p.parent.name+' '+json.dumps(dict(num_images=report['num_images'], **report['table']), ensure_ascii=False), flush=True)
        seen.add(p.parent.name)
    states = [q, a, last, main, read(r/'optional-seed44-progress.json')]
    errors = [s for s in states if 'FAIL' in s.get('status', '')]
    if errors and errors != last_alert:
        print('ATTENTION '+json.dumps(errors), flush=True)
        last_alert = errors
    time.sleep(min(55, max(0, deadline-time.time())))
print('Fixed round deadline reached; monitor exiting.', flush=True)
