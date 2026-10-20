"""Write a paired, same-protocol comparison and posterior-degeneracy diagnostics."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'logs/dmd/sd3-medium-pickscore-no-reward-train'
    new = root / 'logs/rtdmd-source-is/sd3-medium-rtdmd-source-is-train'
    out = root / 'logs/rtdmd-source-is/comparison-1000'
    out.mkdir(parents=True, exist_ok=True)
    reports = [json.loads((p / 'eval/step-1000/final-metrics.json').read_text()) for p in (base, new)]
    for key in ('test_file_sha256', 'resolution', 'seed', 'ema', 'clip', 'aesthetic', 'pickscore', 'hp'):
        if reports[0]['protocol'][key] != reports[1]['protocol'][key]:
            raise ValueError(f'Incompatible evaluation protocols: {key}')
    a, b = [json.loads((p / 'config.json').read_text()) for p in (base, new)]
    differences = {}
    def compare(x, y, prefix=''):
        for k in sorted(x.keys() | y.keys()):
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(x.get(k), dict) and isinstance(y.get(k), dict):
                compare(x[k], y[k], key)
            elif x.get(k) != y.get(k):
                differences[key] = {'DMD': x.get(k), 'SourceIS': y.get(k)}
    compare(a, b)
    records = [json.loads(line) for line in (new / 'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in records if 'posterior/ess' in r]
    if len(updates) != 100 or records[-1]['step'] != 1000:
        raise ValueError('Expected 100 generator updates over 1000 outer steps')
    if not all(np.isfinite(v) for r in records for v in r.values() if isinstance(v, (float, int))):
        raise ValueError('Nonfinite training metric')
    diagnostics = {k: {'mean': float(np.mean([r[k] for r in updates])),
                       'min': float(min(r[k] for r in updates)),
                       'max': float(max(r[k] for r in updates))}
                   for k in updates[0] if k.startswith('posterior/')}
    ess = np.array([r['posterior/ess'] for r in updates])
    diagnostics['fraction_updates_ess_below_2'] = float(np.mean(ess < 2))
    from safetensors.torch import load_file
    adapter = new / 'checkpoints/checkpoint-1000/lora/generator/adapter_model.safetensors'
    tensors = load_file(str(adapter))
    import torch
    if not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('Invalid final generator adapter')
    b_norm = sum(t.double().square().sum().item() for k, t in tensors.items() if 'lora_B' in k)
    if b_norm <= 0:
        raise ValueError('Generator LoRA did not update')
    checkpoint_validation = dict(path=str(adapter), tensor_count=len(tensors),
                                 lora_B_squared_norm=b_norm, all_finite=True)
    rows = [r['table'] for r in reports]
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = {}
    scores = [json.loads((p / 'eval/step-1000/per-image-scores.json').read_text()) for p in (base, new)]
    if [(r['index'], r['prompt']) for r in scores[0]] != [(r['index'], r['prompt']) for r in scores[1]]:
        raise ValueError('Per-image prompt pairing mismatch')
    for name, scale in [('clipscore', 100), ('aesthetic', 1), ('pickscore', 26), ('hpsv2', 100)]:
        delta = np.array([q['scores'][name] - p['scores'][name] for p, q in zip(*scores)]) * scale
        paired[name] = {'mean_delta_source_is_minus_dmd': float(delta.mean()),
                        'fraction_prompts_improved': float((delta > 0).mean()),
                        'note': 'One seed; no claim of statistical superiority'}
    summary = dict(
        completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        mathematical_method='Corrected source-space SNIS, NOT original reverse-CPS Scheme A',
        outer_steps=1000, generator_updates=100, source_is_fake_updates=0,
        num_test_prompts=2048, rows=rows, paired_differences=paired,
        posterior_diagnostics=diagnostics, config_differences=differences,
        checkpoint_validation=checkpoint_validation,
        train_file_sha256=hashlib.sha256((root / 'dataset/pickscore/train.txt').read_bytes()).hexdigest(),
        test_file_sha256=reports[0]['protocol']['test_file_sha256'],
        limitation='Finite-particle SNIS is biased; ESS near 1 means severe degeneration. Quality metrics do not prove the paper.',
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ['单卡 A100；相同 PickScore 测试集 2048 条；1000 外层轮 / 100 生成器更新。',
             'Source-IS 是原方案 A 的数学修正版，不是原反向 CPS 路径比公式的严格复现。', '']
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False))
    lines.extend(['', f'后验 ESS 均值：{ess.mean():.6f} / 8；ESS<2 的更新比例：{(ess<2).mean():.1%}。',
                  '有限粒子有偏，权重退化限制后验近似质量；单种子图像指标不能证明原论文成立。',
                  '完整配置差异、逐图配对差值与诊断：summary.json；表格：comparison.csv。'])
    (out / 'RESULTS_CN.txt').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    steps = [r['step'] for r in updates]
    axes[0].plot(steps, [r['generator/loss_dm'] for r in updates])
    axes[0].set(title='Generator DMD surrogate loss', xlabel='Outer step', yscale='log')
    axes[1].plot(steps, ess)
    axes[1].axhline(1, color='red', linestyle='--')
    axes[1].set(title='Importance ESS (8 particles)', xlabel='Outer step', ylim=(.8, 8.2))
    axes[2].plot(steps, [r['generator/grad_norm'] for r in updates])
    axes[2].set(title='Generator gradient norm before clipping', xlabel='Outer step')
    fig.tight_layout()
    fig.savefig(out / 'training-diagnostics.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
