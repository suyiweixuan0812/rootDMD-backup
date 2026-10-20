"""Three-way comparison with baseline DMD, previous Source-IS and posterior MH."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from safetensors.torch import load_file


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [root/'logs/dmd/sd3-medium-pickscore-no-reward-train',
             root/'logs/rtdmd-source-is/sd3-medium-rtdmd-source-is-train',
             root/'logs/rtdmd-posterior-mh/sd3-medium-rtdmd-posterior-mh-train']
    out = root/'logs/rtdmd-posterior-mh/comparison-1000'
    out.mkdir(parents=True, exist_ok=True)
    reports = [json.loads((p/'eval/step-1000/final-metrics.json').read_text()) for p in paths]
    for report in reports:
        if report['num_images'] != 2048:
            raise ValueError('Expected all 2048 test prompts')
        for key in ('test_file_sha256', 'resolution', 'seed', 'ema', 'clip', 'aesthetic', 'pickscore', 'hp'):
            if report['protocol'][key] != reports[0]['protocol'][key]:
                raise ValueError(f'Incompatible protocol: {key}')
    configs = [json.loads((p/'config.json').read_text()) for p in paths]
    differences = {}
    def compare(x, y, prefix=''):
        for k in sorted(x.keys() | y.keys()):
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(x.get(k), dict) and isinstance(y.get(k), dict):
                compare(x[k], y[k], key)
            elif x.get(k) != y.get(k):
                differences[key] = dict(DMD=x.get(k), PosteriorMH=y.get(k))
    compare(configs[0], configs[-1])
    records = [json.loads(line) for line in (paths[-1]/'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in records if 'mh/accept_rate' in r]
    if len(updates) != 100 or records[-1]['step'] != 1000:
        raise ValueError('Expected 100 generator updates over 1000 outer steps')
    if not all(np.isfinite(v) for r in records for v in r.values() if isinstance(v, (float, int))):
        raise ValueError('Nonfinite metric')
    diagnostics = {k: dict(mean=float(np.mean([r[k] for r in updates])),
                           min=float(min(r[k] for r in updates)), max=float(max(r[k] for r in updates)))
                   for k in updates[0] if k.startswith('mh/')}
    stratified = {}
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1)]:
        selected = [r for r in updates if lo <= r['mh/sigma'] < hi]
        if selected:
            stratified[f'{lo}<=sigma<{hi}'] = dict(count=len(selected),
                accept_rate=float(np.mean([r['mh/accept_rate'] for r in selected])),
                source_move_rate=float(np.mean([r['mh/source_move_rate'] for r in selected])),
                frozen_update_fraction=float(np.mean([r['mh/all_rejected'] for r in selected])),
                source_displacement_rms=float(np.mean([r['mh/mean_source_displacement_rms'] for r in selected])))
    adapter = paths[-1]/'checkpoints/checkpoint-1000/lora/generator/adapter_model.safetensors'
    tensors = load_file(str(adapter))
    if not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('Invalid adapter')
    b_norm = sum(t.double().square().sum().item() for k, t in tensors.items() if 'lora_B' in k)
    if b_norm <= 0:
        raise ValueError('Adapter did not update')
    rows = [r['table'] for r in reports]
    with (out/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = [json.loads((p/'eval/step-1000/per-image-scores.json').read_text()) for p in paths]
    for records_for_method in scores[1:]:
        if [(r['index'], r['prompt']) for r in records_for_method] != [(r['index'], r['prompt']) for r in scores[0]]:
            raise ValueError('Prompt pairing mismatch')
    deltas = {}
    for label, previous in [('DMD', scores[0]), ('SourceIS', scores[1])]:
        deltas[label] = {}
        for name, scale in [('clipscore', 100), ('aesthetic', 1), ('pickscore', 26), ('hpsv2', 100)]:
            delta = np.array([q['scores'][name] - p['scores'][name] for p, q in zip(previous, scores[-1])]) * scale
            deltas[label][name] = dict(mean_delta_mh_minus_previous=float(delta.mean()),
                                      fraction_prompts_improved=float((delta > 0).mean()))
    summary = dict(completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        method='Teacher-guided block pCN-MH, paired stationary initialization, same-observation Jacobian (paper Eq.2 interpretation)',
        outer_steps=1000, generator_updates=100, fake_updates=0, num_test_prompts=2048,
        rows=rows, paired_differences=deltas, config_differences=differences,
        mh_diagnostics=diagnostics, diagnostics_by_sigma=stratified,
        checkpoint_validation=dict(path=str(adapter), tensor_count=len(tensors), all_finite=True, lora_B_squared_norm=b_norm),
        short_chain_warning='Eight transitions cannot provide a reliable MCMC ESS. Equal MH averaging weights are not evidence of independent posterior draws.',
        inference_checkpoint='EMA generator LoRA only; exact training resume is disabled.',
        test_file_sha256=reports[0]['protocol']['test_file_sha256'],
        train_file_sha256=hashlib.sha256((root/'dataset/pickscore/train.txt').read_bytes()).hexdigest(),
        limitation='Single seed; gradient site and posterior estimator both changed; no causal ablation or superiority/convergence proof.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ['相同单卡DMD外部配置；1000外层轮/100次生成器更新；完整2048条PickScore测试提示词。',
        '三方对照：DMD、上一版Source-IS、本轮Posterior-MH。均NFE=4，无奖励训练。', '']
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False))
    lines += ['', f"MH平均接受率：{diagnostics['mh/accept_rate']['mean']:.2%}；真正图像移动率：{diagnostics['mh/source_move_rate']['mean']:.2%}。",
        f"整轮全拒绝比例：{diagnostics['mh/all_rejected']['mean']:.2%}。",
        '8步短链不能可靠估计MCMC ESS；不能把等权平均说成ESS=8。分噪声段诊断见summary.json。',
        '保留教师提议/Student后验目标/概率比校正，采用原始随机轨迹空间MH，而非原论文反向CPS路径IS。',
        '反传位置按论文同一观测点clean prediction解释修改；不是只更换候选权重。',
        '单种子试运行不证明优于DMD，不证明原论文所有理论断言；本轮没有把两个方法改动分离消融。']
    (out/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    steps = [r['step'] for r in updates]
    axes[0].plot(steps, [r['generator/loss_dm'] for r in updates])
    axes[0].set(title='Same-observation surrogate loss', xlabel='Outer step', yscale='log')
    axes[1].plot(steps, [r['mh/accept_rate'] for r in updates], label='Acceptance')
    axes[1].plot(steps, [r['mh/source_move_rate'] for r in updates], label='Image actually moves', alpha=.7)
    axes[1].set(title='MH acceptance is not ESS', xlabel='Outer step', ylim=(-.02, 1.02))
    axes[1].legend()
    axes[2].scatter([r['mh/sigma'] for r in updates],
                    [r['mh/mean_source_displacement_rms'] for r in updates], s=12)
    axes[2].set(title='Posterior mean displacement from paired source', xlabel='Corruption sigma')
    fig.tight_layout()
    fig.savefig(out/'training-diagnostics.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
