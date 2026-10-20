"""Summarize completed training blocks; no image-quality inference from ESS."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def blocks(path, step, size):
    config = json.loads((path.parent/'config.json').read_text())
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[-1]['step'] >= step
    maximum = float(config['train']['max_grad_norm'])
    out = []
    keys = ['path_is/ess', 'path_is/path_ratio_span', 'path_is/observation_span',
            'path_is/frozen_update', 'path_is/total_mean_displacement_rms',
            'path_is/bridge_strength', 'path_is/rb_removed_variance_trace_per_coordinate']
    for start in range(0, step, size):
        end = min(start+size, step)
        subset = [x for x in rows if start < x['step'] <= end and 'generator/grad_norm' in x]
        assert len(subset) == end//config['dmd']['fake_update_ratio'] - start//config['dmd']['fake_update_ratio']
        def describe(selected):
            if not selected:
                return {'updates': 0}
            value = {'updates': len(selected), 'gradient_clip_fraction': float(np.mean([x['generator/grad_norm'] > maximum for x in selected]))}
            for k in keys:
                if k in selected[0]:
                    value[k] = float(np.mean([x[k] for x in selected]))
            return value
        item = dict(outer_step_start=start+1, outer_step_end=end, **describe(subset))
        item['by_stopping_node'] = {str(j): describe([x for x in subset if x.get('path_is/stop') == j]) for j in range(1, 5)}
        out.append(item)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--block', type=int, default=1000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    paths = {'DMD': a.baseline/'metrics.jsonl', 'Path-IS': a.candidate/'metrics.jsonl'}
    values = {name: blocks(path, a.step, a.block) for name, path in paths.items()}
    result = dict(outer_steps=a.step, block_size=a.block, methods=values,
                  sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
                  interpretation='Candidate-pool ESS, not independent-sample ESS. Completed blocks only. These training diagnostics do not establish image-quality gains.')
    (a.output/'training-drift.json').write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, key, label in zip(axes.flat,
            ['path_is/ess', 'path_is/path_ratio_span', 'path_is/frozen_update', 'gradient_clip_fraction'],
            ['Candidate-pool ESS (N=8)', 'Path log-ratio span', 'Fully frozen update fraction', 'Gradient clipping fraction']):
        for name, items in values.items():
            data = [x for x in items if key in x]
            if data:
                axis.plot([x['outer_step_end'] for x in data], [x[key] for x in data], marker='o', label=name)
        axis.set(xlabel='Outer training step', ylabel=label)
        axis.grid(alpha=.25)
        axis.legend()
    fig.suptitle('Matched training diagnostics; concurrent A100 runs')
    fig.savefig(a.output/'training-drift.png', dpi=180)
    fig.savefig(a.output/'training-drift.pdf')
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
