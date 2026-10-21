"""Describe observed training geometry without selecting new candidates."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'logs/path-ratio-round1'


def main():
    summaries = []
    for evidence in sorted((ROUND / 'evidence').glob('sd3-det-*')):
        name = evidence.name
        if 'pick2048' in name or 'holdout' in name:
            continue
        metrics, config = evidence / 'metrics.jsonl', evidence / 'config.json'
        if not metrics.is_file() or not config.is_file():
            continue
        cfg = json.loads(config.read_text())
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        updates = [row for row in rows if 'generator/grad_norm' in row]
        if not updates:
            continue
        threshold = cfg['train']['max_grad_norm']
        summary = dict(run=name, generator_updates=len(updates),
            gradient_clip_threshold=threshold,
            gradient_clip_fraction=sum(row['generator/grad_norm'] > threshold for row in updates)/len(updates),
            metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(), groups=[])
        path = [row for row in updates if 'path_is/ess' in row]
        for label, selected in [('all_updates_with_path_diagnostics', path),
                ('empty_prefix_J1', [row for row in path if row['path_is/stop'] == 1]),
                ('positive_prefix_low_noise', [row for row in path if row['path_is/stop'] > 1 and row['path_is/sigma'] < .3]),
                ('positive_prefix_medium_noise', [row for row in path if row['path_is/stop'] > 1 and .3 <= row['path_is/sigma'] < .9]),
                ('positive_prefix_high_noise', [row for row in path if row['path_is/stop'] > 1 and row['path_is/sigma'] >= .9])]:
            if not selected:
                continue
            group = dict(label=label, count=len(selected))
            for key in ['ess', 'reference_probability', 'actual_move_rate', 'move_rms',
                    'mean_displacement_rms', 'path_ratio_span', 'observation_span',
                    'rb_removed_variance_trace_per_coordinate', 'frozen_update']:
                values = [row['path_is/' + key] for row in selected if 'path_is/' + key in row]
                if values:
                    group[key] = dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values))
            summary['groups'].append(group)
        summaries.append(summary)
    output = dict(scope='Descriptive statistics from completed persisted training logs; no parameter selection or extra training.',
        caveats=['Candidate weight ESS is not an independent posterior sample count.',
            'J1 has an empty stochastic prefix; report it separately.',
            'Removed variance trace is conditional candidate readout diversity, not measured gradient variance or an image metric.',
            'Noise groups and seeds contain different finite sample counts; these are descriptive, not causal estimates.',
            'No-update outer steps are excluded from generator clipping statistics.'], runs=summaries)
    target = ROUND / 'selected-training-diagnostics.json'
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print('Wrote descriptive diagnostics for', len(summaries), 'completed training runs')


if __name__ == '__main__':
    main()
