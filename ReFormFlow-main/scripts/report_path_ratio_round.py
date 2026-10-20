"""Audited paired comparison; intervals concern prompts, not training seeds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = [('clipscore', 'CLIP Score', 100), ('aesthetic', 'Aesthetic Score', 1),
           ('pickscore', 'Pick Score', 26), ('hpsv2', 'HP Score', 100)]


def load(folder):
    folder = Path(folder)
    report = json.loads((folder / 'final-metrics.json').read_text())
    scores = json.loads((folder / 'per-image-scores.json').read_text())
    assert len(scores) == report['num_images']
    for raw, table, scale in METRICS:
        mean = float(np.mean([r['scores'][raw] for r in scores])) * scale
        assert abs(mean - report['table'][table]) < 1e-10
    return folder, report, scores


def diagnostics(folder, report):
    root = Path(__file__).resolve().parents[1]
    source = report['protocol'].get('checkpoint_evaluation_only', {}).get('path')
    run = ((root / source).resolve().parents[3] if source else
           root / 'logs/path-ratio-round1/runs' / folder.name)
    p = run / 'metrics.jsonl'
    if not p.exists(): return {}
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    updates = [r for r in rows if 'path_is/ess' in r]
    if not updates: return {}
    assert len(updates) == 100 and rows[-1]['step'] == 1000
    assert all(r['path_is/reference_replay_exact'] == 1 for r in updates)
    def summarize(selected):
        keys = ['path_is/ess', 'path_is/max_weight', 'path_is/reference_probability',
                'path_is/actual_move_rate', 'path_is/frozen_update',
                'path_is/total_mean_displacement_rms', 'path_is/path_ratio_span',
                'path_is/observation_span']
        return {'updates': len(selected), **{k: float(np.mean([r[k] for r in selected])) for k in keys}}
    result = {'overall': summarize(updates), 'by_sigma': {},
              'all_reference_replays_exact': True,
              'warning': 'Candidate-pool weight ESS is not the ESS of independent posterior samples.'}
    for lo, hi in [(0,.25),(.25,.5),(.5,.75),(.75,1)]:
        selected = [r for r in updates if lo <= r['path_is/sigma'] < hi]
        if selected: result['by_sigma'][f'{lo}<=sigma<{hi}'] = summarize(selected)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', action='append', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--bootstrap', default=10000, type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_folder, base, base_scores = load(args.baseline)
    comparison = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'baseline': base['table'], 'num_prompts': len(base_scores),
                  'bootstrap_replicates': args.bootstrap,
                  'confidence_scope': 'Paired prompt bootstrap conditional on this one trained seed; not training-seed uncertainty. No multiple-selection correction.',
                  'candidates': []}
    fields = ['test_file_sha256','resolution','seed','ema','clip','aesthetic','pickscore','hp']
    for name in args.candidate:
        folder, report, scores = load(name)
        for key in fields:
            assert report['protocol'][key] == base['protocol'][key], key
        assert [(r['index'],r['prompt']) for r in scores] == [(r['index'],r['prompt']) for r in base_scores]
        rng = np.random.default_rng(104728)
        paired = {}
        for raw, table, scale in METRICS:
            delta = np.array([r['scores'][raw]-b['scores'][raw] for b,r in zip(base_scores,scores)]) * scale
            means=[]
            for start in range(0,args.bootstrap,500):
                indices = rng.integers(len(delta),size=(min(500,args.bootstrap-start),len(delta)))
                means.extend(delta[indices].mean(1).tolist())
            paired[table] = {'mean_delta': float(delta.mean()),
                             'ci95_prompt_bootstrap': np.quantile(means,[.025,.975]).tolist(),
                             'fraction_prompts_improved': float((delta>0).mean())}
        comparison['candidates'].append({'source': str(folder), 'table': report['table'],
             'paired_difference': paired, 'sampling_diagnostics': diagnostics(folder,report),
             'report_sha256': hashlib.sha256((folder/'final-metrics.json').read_bytes()).hexdigest()})
    (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False))
    lines=['同提示词、同推理种子、同四项评分模型；所有差值=候选减DMD。',
           f'提示词数：{len(base_scores)}。置信区间仅反映这些提示词的成对抽样误差，不代表多训练种子的稳定性。',
           '候选池权重ESS不能解释为相同数量的独立后验样本。','',json.dumps(base['table'],ensure_ascii=False)]
    for r in comparison['candidates']:
        lines += ['',json.dumps(r['table'],ensure_ascii=False)]
        for key, d in r['paired_difference'].items():
            lo,hi=d['ci95_prompt_bootstrap']
            lines.append(f'{key}: {d["mean_delta"]:+.4f}, 提示词bootstrap95%区间 [{lo:+.4f}, {hi:+.4f}]')
    (args.output/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':main()
