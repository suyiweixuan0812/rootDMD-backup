"""Audit and report both prespecified v30 experiments without test-set selection."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/rtdmd-pcmh/comparison-v30'
RUNS = [
    ('DMD', 'logs/dmd/sd3-medium-pickscore-no-reward-train'),
    ('Source-IS', 'logs/rtdmd-source-is/sd3-medium-rtdmd-source-is-train'),
    ('Posterior-MH', 'logs/rtdmd-posterior-mh/sd3-medium-rtdmd-posterior-mh-train'),
    ('PC-MH lambda=.25 (primary)', 'logs/rtdmd-pcmh/sd3-medium-rtdmd-pcmh-lambda025-train'),
    ('PC-MH lambda=.50 (sensitivity)', 'logs/rtdmd-pcmh/sd3-medium-rtdmd-pcmh-lambda050-train'),
]
METRICS = [('CLIP Score', 'clipscore', 100), ('Aesthetic Score', 'aesthetic', 1),
           ('Pick Score', 'pickscore', 26), ('HP Score', 'hpsv2', 100)]


def differences(previous, current, prefix=''):
    out = {}
    for key in sorted(previous.keys() | current.keys()):
        name = f'{prefix}.{key}' if prefix else key
        a, b = previous.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(differences(a, b, name))
        elif a != b:
            out[name] = dict(previous=a, current=b)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / 'source-manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source/data changed after experiment launch: {relative}')
    reports, scores, configs, rows, audits = [], [], [], [], {}
    for label, relative in RUNS:
        run = ROOT / relative
        folder = run / 'eval/step-1000'
        report = json.loads((folder / 'final-metrics.json').read_text())
        per_image = json.loads((folder / 'per-image-scores.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        if report['num_images'] != 2048 or [r['index'] for r in per_image] != list(range(2048)):
            raise ValueError(f'Incomplete or duplicated test set: {label}')
        if reports:
            for key in ('test_file_sha256', 'resolution', 'seed', 'ema', 'clip', 'aesthetic', 'pickscore', 'hp'):
                if reports[0]['protocol'][key] != report['protocol'][key]:
                    raise ValueError(f'Protocol mismatch: {label}: {key}')
            if [(r['index'], r['prompt']) for r in scores[0]] != [(r['index'], r['prompt']) for r in per_image]:
                raise ValueError(f'Prompt pairing mismatch: {label}')
        for table_key, raw_key, scale in METRICS:
            values = np.asarray([r['scores'][raw_key] for r in per_image], dtype=np.float64)
            if not np.isfinite(values).all() or not np.isclose(values.mean() * scale, report['table'][table_key], rtol=0, atol=1e-10):
                raise ValueError(f'Mean or finite-score validation failed: {label}: {raw_key}')
        row = dict(report['table'], Method=label)
        rows.append(row)
        reports.append(report)
        scores.append(per_image)
        configs.append(config)
        if not label.startswith('PC-MH'):
            continue
        if (run / 'status.txt').read_text().strip() != 'COMPLETED':
            raise ValueError(f'Run not completed: {label}')
        history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        updates = [r for r in history if 'mh/accept_rate' in r]
        if len(updates) != 100 or history[-1]['step'] != 1000:
            raise ValueError(f'Wrong training schedule: {label}')
        if not all(np.isfinite(v) for r in history for v in r.values() if isinstance(v, (int, float))):
            raise ValueError(f'Nonfinite training metric: {label}')
        if not all(r['generator/teacher_clean_mse'] > 0 for r in updates):
            raise ValueError(f'Nonpositive normalization observed: {label}')
        mean = lambda key, selected=updates: float(np.mean([r[key] for r in selected]))
        strata = {}
        for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1)]:
            selected = [r for r in updates if lo <= r['mh/sigma'] < hi]
            if selected:
                strata[f'{lo}<=sigma<{hi}'] = dict(count=len(selected),
                    accept_rate=mean('mh/accept_rate', selected),
                    actual_image_move_rate=mean('mh/source_move_rate', selected),
                    updates_without_image_movement=float(np.mean([r['mh/source_move_rate'] == 0 for r in selected])),
                    posterior_mean_displacement_rms=mean('mh/mean_source_displacement_rms', selected))
        adapter = run / 'checkpoints/checkpoint-1000/lora/generator/adapter_model.safetensors'
        tensors = load_file(str(adapter))
        if not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values()):
            raise ValueError(f'Invalid checkpoint: {label}')
        bnorm = sum(t.double().square().sum().item() for key, t in tensors.items() if 'lora_B' in key)
        if bnorm <= 0:
            raise ValueError(f'No LoRA update: {label}')
        # Preserve prompt metadata outside the volatile image-directory symlink.
        (folder / 'prompts-persistent.json').write_text(json.dumps([
            {key: r[key] for key in ('index', 'prompt', 'metadata') if key in r} for r in per_image
        ], indent=2, ensure_ascii=False))
        audits[label] = dict(
            outer_steps=1000, generator_updates=100, fake_updates=0,
            consistency_weight=config['posterior_consistency']['weight'],
            diagnostic_means={key: mean(key) for key in updates[0] if key.startswith(('mh/', 'generator/'))},
            updates_without_image_movement=float(np.mean([r['mh/source_move_rate'] == 0 for r in updates])),
            sigma_strata=strata,
            checkpoint=dict(path=str(adapter), tensor_count=len(tensors), all_finite=True,
                lora_B_squared_norm=bnorm, sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                kind='Inference EMA LoRA; not a full training-resume checkpoint'),
            final_image_count=len(list((folder / 'rank-0').glob('*.png'))),
            images_volatile_on_host_restart=True,
        )
    paired = {}
    for index in (3, 4):
        label = RUNS[index][0]
        paired[label] = {}
        for baseline in (0, 2):
            paired[label][RUNS[baseline][0]] = {}
            for key, raw_key, scale in METRICS:
                delta = np.asarray([q['scores'][raw_key] - p['scores'][raw_key]
                    for p, q in zip(scores[baseline], scores[index])]) * scale
                paired[label][RUNS[baseline][0]][key] = dict(
                    mean_delta=float(delta.mean()), fraction_prompts_improved=float((delta > 0).mean()))
    summary = dict(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        rows=rows, protocol_verified=True, source_hashes_verified=True, num_prompts=2048,
        prespecified_primary_lambda=.25, prespecified_sensitivity_lambda=.50,
        selection='Both reported; no final-test winner substituted for the prespecified primary.',
        paired_differences=paired, new_run_audits=audits,
        config_differences_vs_DMD={RUNS[i][0]: differences(configs[0], configs[i]) for i in (3, 4)},
        config_differences_vs_posterior_MH={RUNS[i][0]: differences(configs[2], configs[i]) for i in (3, 4)},
        test_file_sha256=reports[0]['protocol']['test_file_sha256'],
        caveats=[
            'Single training seed, 100 generator updates; no stability or superiority proof.',
            'Only conditional posterior-induced raw-gradient covariance scales by (1-lambda)^2; expected update also changes.',
            'Regularization does not repair low-noise MH mixing; eight transitions do not provide a reliable MCMC ESS.',
            'Two new processes share one A100; no wall-clock speed comparison.',
            'Full new PNG sets are in /dev/shm (volatile); scores, prompts, configs and EMA LoRA persist in project.',
        ])
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['v30：相同外部参数、完整2048条PickScore测试提示词、同图四指标。',
        '主实验lambda=.25；敏感性lambda=.50；均1000外层轮/100次生成器更新，NFE=4。', '']
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False))
    for label in paired:
        lines += ['', label]
        for baseline, metrics in paired[label].items():
            lines.append('相对' + baseline + '：' + '；'.join(f'{key} {v["mean_delta"]:+.6f}' for key, v in metrics.items()))
    lines += ['', '限制：单种子短训练；不证明稳定优于DMD或原论文全部理论正确。',
        '正则改变期望更新，条件方差缩减不保证最终图像指标提高；没有修复低噪声链混合。',
        '全部图片暂存/dev/shm，重启会丢失；逐图分数、提示词元数据和EMA LoRA已持久保存。',
        'EMA LoRA仅用于推理，不能精确恢复中断训练。完整参数差异及哈希见summary.json。']
    (OUT / 'RESULTS_CN.txt').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    names = ['DMD', 'Source-IS', 'MH', 'PC-MH\n0.25', 'PC-MH\n0.50']
    for ax, (key, _, _) in zip(axes, METRICS):
        values = [r[key] for r in rows]
        bars = ax.bar(names, values, color=['#7b8794', '#d99587', '#5b8db8', '#36856c', '#85bca9'])
        ax.set(title=key + ' (higher is better)', ylabel='Same saved-PNG protocol')
        ax.tick_params(axis='x', labelsize=8)
        ax.bar_label(bars, fmt='%.3f', fontsize=7)
        ax.set_ylim(0, max(values) * 1.15)
    fig.suptitle('PickScore prompts: 2048 images, one seed, 100 generator updates')
    fig.tight_layout()
    fig.savefig(OUT / 'metrics-comparison.png', dpi=180)
    fig.savefig(OUT / 'metrics-comparison.pdf')
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
