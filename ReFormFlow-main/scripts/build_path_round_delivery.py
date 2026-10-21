"""Build an honest Chinese result index from completed reports only."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import numpy as np
from report_path_ratio_round import load, METRICS

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'logs/path-ratio-round1'
OUT = ROOT.parent
CASES = ['dmd', 'dmd_fp32', 'base', 'time', 'paired', 'paired_time',
         'bridge', 'bridge_time', 'adaptive_bridge', 'adaptive_bridge_time', 'last', 'last_time', 'last_q2_time', 'paired_q2_time']
ENTRIES = {
    'base': ('scripts/train_rtdmd_path_is_details_sd3.py', 'config/rtdmd_path_is_details.py:time', ['--config.posterior_time_weight.mode=none']),
    'time': ('scripts/train_rtdmd_path_is_details_sd3.py', 'config/rtdmd_path_is_details.py:time', []),
    'bridge': ('scripts/train_rtdmd_path_is_bridge_sd3.py', 'config/rtdmd_path_is_bridge.py:bridge', []),
    'bridge_time': ('scripts/train_rtdmd_path_is_bridge_sd3.py', 'config/rtdmd_path_is_bridge.py:bridge_time', []),
    'adaptive_bridge': ('scripts/train_rtdmd_path_is_adaptive_bridge_sd3.py', 'config/rtdmd_path_is_adaptive_bridge.py:adaptive_bridge', []),
    'adaptive_bridge_time': ('scripts/train_rtdmd_path_is_adaptive_bridge_sd3.py', 'config/rtdmd_path_is_adaptive_bridge.py:adaptive_bridge_time', []),
    'last': ('scripts/train_rtdmd_path_is_last_sd3.py', 'config/rtdmd_path_is_last.py:last', []),
    'last_time': ('scripts/train_rtdmd_path_is_last_sd3.py', 'config/rtdmd_path_is_last.py:last_time', []),
    'last_q2_time': ('scripts/train_rtdmd_path_is_query_views_sd3.py', 'config/rtdmd_path_is_query_views.py:last_q2_time', []),
}


def completed(name):
    p = R/'evidence'/name
    status = p/'status.txt'
    if not status.exists() or not status.read_text().strip().startswith('COMPLETED'):
        return None
    if not (p/'final-metrics.json').exists():
        return None
    _, report, scores = load(p)
    return report


def pair_statistics(reports, candidate, comparator, suffix='pick2048-exact', count=2048):
    matched = []
    for seed in [42, 43, 44]:
        a = reports.get(f'sd3-det-{comparator}-seed{seed}-{suffix}')
        b = reports.get(f'sd3-det-{candidate}-seed{seed}-{suffix}')
        if a is None or b is None:
            continue
        assert a['num_images'] == b['num_images'] == count
        assert a['protocol']['checkpoint_training_configuration']['seed'] == b['protocol']['checkpoint_training_configuration']['seed'] == seed
        for key in ['test_file_sha256', 'resolution', 'seed', 'ema', 'clip', 'aesthetic', 'pickscore', 'hp']:
            assert a['protocol'][key] == b['protocol'][key], key
        matched.append(dict(seed=seed, baseline=a['table'], candidate=b['table'],
                            deltas={k:b['table'][k]-a['table'][k] for _, k, _ in METRICS}))
    stats = {}
    for _, key, _ in METRICS:
        data = [x['deltas'][key] for x in matched]
        stats[key] = dict(mean_delta=float(np.mean(data)) if data else None,
                         sample_std_across_seed_deltas=float(np.std(data, ddof=1)) if len(data)>1 else None)
    return dict(num_prompts=count, dataset_suffix=suffix, matched_training_seeds=[x['seed'] for x in matched], per_seed=matched, statistics=stats,
                all_seeds_improve_both_preferences=bool(matched) and all(
                    x['deltas']['Pick Score']>0 and x['deltas']['HP Score']>0 for x in matched))


def main():
    selection = json.loads((R/'deterministic-selection.json').read_text())
    best, control = selection['selected_case'], selection['paired_control']
    reports, rows = {}, []
    for folder in sorted((R/'evidence').iterdir()):
        report = completed(folder.name)
        if report is None:
            continue
        reports[folder.name] = report
        cohort = ('DEPRECATED_EXTRA_EMA_BASELINE' if folder.name == 'dmd-ema-dev64' else
                  'NUMERIC_PARITY_ONLY' if folder.name == 'sd3-unused-online-skip-parity8' else
                  'DETERMINISTIC_COHORT' if folder.name.startswith('sd3-det-') else 'EARLIER_NONDETERMINISTIC_COHORT')
        role = {64:'DrawBench development', 138:'Reserved unique holdout', 2048:'PickScore full test', 8:'Inference smoke'}.get(report['num_images'], 'Other')
        row = dict(run=folder.name, cohort=cohort, dataset_role=role, num_prompts=report['num_images'], **report['table'])
        rows.append(row)
    fields = ['run', 'cohort', 'dataset_role', 'num_prompts', 'Method', 'NFE', 'Post-Train', 'CLIP Score', 'Aesthetic Score', 'Pick Score', 'HP Score']
    with (OUT/'RTDMD_v31_results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pairs = {comparator:pair_statistics(reports, best, comparator) for comparator in ['dmd', 'dmd_fp32', control]}
    faithful = selection.get('pure_teacher_alternative', dict(case=best, paired_control=control))
    faithful_pairs = {comparator:pair_statistics(reports, faithful['case'], comparator)
                      for comparator in ['dmd', 'dmd_fp32', faithful['paired_control']]}
    holdout = {case:{comparator:pair_statistics(reports,case,comparator,
                    suffix='holdout138-exact',count=138)
                    for comparator in ['dmd','dmd_fp32',paired_case]}
               for case,paired_case in [(best,control),(faithful['case'],faithful['paired_control'])]}
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case=best,
        dev_selection_status=selection['status'], paired_control=control, comparisons=pairs, independent_holdout=holdout,
        pure_teacher_alternative=dict(**faithful, comparisons=faithful_pairs),
        statistical_scope='Every available matched trained seed is retained, including negative replications. No reliable training-seed CI is claimed from two or three seeds. Prompt bootstrap reports are conditional on the saved trained models.',
        incomplete_groups='Individual completed rows remain in CSV; pair means use every seed with both reports, with seed sets explicitly listed. Historical nondeterministic and deterministic cohorts are never averaged together.')
    long_path=R/'optional-faithful-long-progress.json'
    long_status=json.loads(long_path.read_text()) if long_path.exists() else {'status':'NOT_PLANNED'}
    summary['optional_long_confirmation']=long_status
    window_path=R/'optional-long-final-evaluation-window-progress.json'
    window=json.loads(window_path.read_text()) if window_path.exists() else {'status':'NOT_PLANNED'}
    user_window_path=R/'user-budget-final-evaluation-progress.json'
    if user_window_path.exists():
        summary['superseded_final_window_monitor']=window
        window=json.loads(user_window_path.read_text())
    summary['optional_final_evaluation_window']=window
    prefix_path=R/'deterministic-full-prefix-reproduction.json'
    prefix=(json.loads(prefix_path.read_text()) if prefix_path.exists() else
            [json.loads(p.read_text()) for case in ['dmd','last']
             if (p:=R/f'deterministic-{case}1000-prefix-live-audit.json').exists()] or None)
    summary['optional_long_prefix_reproduction']=prefix
    third=R/'optional-seed44-progress.json'
    summary['optional_third_seed_status']=json.loads(third.read_text()) if third.exists() else {'status':'NOT_LAUNCHED'}
    (R/'multiseed-validation-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [
        'RTDMD v31：本轮修改与结果说明',
        '生成时间：'+summary['created_utc'],
        '', '一、保留与改动',
        '保留实际 Student/Teacher 路径核比、粒子加权条件均值、Tweedie 接口、无 Fake 训练和四步 Student 推理。',
        '1. 零方差末步：只在正方差前缀取核比，Teacher 与 Student 测度共同采用 Student clean 读出；不新增末端噪声。',
        '2. 后验对应：路径比显式乘实际前向加噪观测似然。提议改为生成器正噪声前缀，不能称为原稿查询点反向 rollout 的逐字复现。',
        '3. 条件参考：从本次 Student 成对源构造正确后验参考，以条件 i-SIR 和 Rao-Blackwell 均值替代独立高维 SNIS；仍属相关 MCMC。',
        '4. BF16 状态按 Gaussian 舍入单元质量计算比，提议创新用 FP64，并逐次检查共同读出重放。有限 RNG/算术仍是理想核的近似。',
        '5. 新增 lambda=.10 的一致性细节改变期望更新；可选 .30/sigma 时间权重改变 surrogate 时间加权，不能说原目标完全未改。',
        '6. 同查询 Jacobian 避免与成对参考相关的源 Jacobian 问题；它是本文明确的 surrogate，不宣称等价于原 DMD 源梯度的完整 KL。',
        '7. 最后正方差块使用物理 Gaussian 坐标和直接 BF16 舍入，修复极小单元逆变换与两次舍入问题；通用创新实现配套正确的后验保护分区。',
        '8. 双查询选项固定源与时间，对两个独立观测的梯度取平均后只裁剪/更新一次；查询计算约加倍，有相同计算量无粒子对照。',
        '9. 纯Teacher入口跳过9/10无更新轮次中未使用的冻结文本编码；数据位置、逐轮种子、EMA及有效更新不变，40轮真实训练的全部更新日志一致。完整1000轮权重比较另存，不能仅凭4次更新声称已验证全程。',
        '最后块的边界：固定前缀后，更早路径比在候选间抵消；连续物理坐标目标是 M_Student×观测似然。Teacher 比值用于校正提议，不能声称完整教师路径探索已保留，或给 Student 后验增加了教师目标。BF16 的单元内提升密度及质量比仍必须保留。',
        '', '本轮固定开发候选：'+best+'；开发守门状态：'+selection['status'],
    ]
    if best.startswith('last'):
        lines += ['所选细节保持纯 Teacher CPS 路径映射与 Student/Teacher 核比，只更新最后正方差创新块；对称辅助候选必须额外保留 Gaussian 创新先验校正。',
                  '更早前缀和停止点不混合，条件方差下降有上限；前缀网络缓存先逐位核验父状态，逐查询重建。']
    elif 'bridge' in best:
        lines += ['所选细节改变提议为 Teacher/Student 桥接 Q；真实权重为 (Student/Teacher)*(Teacher/Q)*似然，即 Student/Q。',
                  '这保留路径比框架，但已经不是纯 Teacher 提议；不能把分解中的 Teacher 因子描述成从未改变原方法。纯 Teacher-only 和最后创新块代码均保留。']
    else:
        lines += ['所选细节仍使用 Teacher-only 路径映射和 OU 辅助条件 i-SIR，权重包含原路径比与观测似然。']
    lines += ['', '二、完整 PickScore2048：所有已完成的匹配训练种子']
    for comparator, pair in pairs.items():
        lines.append('对照 '+comparator+'；训练种子 '+str(pair['matched_training_seeds']))
        for item in pair['per_seed']:
            lines.append('  seed '+str(item['seed'])+'：'+json.dumps(item['deltas'], ensure_ascii=False))
        lines.append('  候选减对照的均值：'+json.dumps({k:v['mean_delta'] for k,v in pair['statistics'].items()}, ensure_ascii=False))
        lines.append('  每个已测种子 Pick/HPS 都上升：'+str(pair['all_seeds_improve_both_preferences']))
    if faithful['case'] != best:
        lines += ['', '更贴近原稿的纯Teacher版本：'+faithful['case']+'；按同一Dev规则在纯Teacher版本内选出，主完整测试不回调选择。']
        for comparator,pair in faithful_pairs.items():
            lines.append('  对照 '+comparator+'；种子 '+str(pair['matched_training_seeds'])+'；均值差 '+json.dumps(
                {k:v['mean_delta'] for k,v in pair['statistics'].items()},ensure_ascii=False))
    lines += ['', '独立138提示词验证：固定方案，不用于选择参数']
    for case,comparators in holdout.items():
        for comparator,pair in comparators.items():
            if pair['matched_training_seeds']:
                lines.append(case+' 减 '+comparator+'：'+json.dumps(
                    {k:v['mean_delta'] for k,v in pair['statistics'].items()},ensure_ascii=False))
    lines += ['', '附加固定3000轮确认：'+long_status.get('status','UNKNOWN')]
    lines.append('最后剩余窗口评分状态：'+window.get('status','UNKNOWN'))
    for training in long_status.get('training',[]):
        name=training+'-pick2048-exact'
        if name in reports:lines.append(name+'：'+json.dumps(reports[name]['table'],ensure_ascii=False))
    if long_status.get('status')=='OPTIONAL_FAITHFUL_LONG_CHECK_COMPLETE':
        for name in long_status.get('reports',[]):
            if name in reports and name not in [n+'-pick2048-exact' for n in long_status.get('training',[])]:
                lines.append(name+'：'+json.dumps(reports[name]['table'],ensure_ascii=False))
    else:
        lines.append('未完成双方完整2048评分时，不把训练结束或单方评分视为新的长程质量结论。')
    if prefix is not None:
        lines.append('1000轮原生EMA与本次从头训练前缀比较：'+json.dumps(
            [{k:x[k] for k in ['case','equal_tensors','tensor_count','all_bit_exact','max_abs_difference']} for x in prefix],ensure_ascii=False))
    lines.append('第三种子时间门槛状态：'+summary['optional_third_seed_status'].get('status','UNKNOWN'))
    lines += [
        '', '三、公平性与结论边界',
        '共同配置审计覆盖模型、512分辨率、BF16、TF32、种子、CPS、随机停止、batch1、LoRA32/alpha64、优化器、LR3e-5、EMA.99及训练轮次。',
        '原 DMD 查询噪声继承 BF16，新路径版为 FP32；因此同时保留原 DMD 与 FP32 精度对照。相同 seed 不代表移除 Fake 后随机流逐次相同。',
        '主短训练为1000轮/100次生成器更新；另有原 Teacher-only 版本双方10000轮/1000更新的长训练，不能混作新细节版长训练成绩。',
        'Dev 为 DrawBench64（61唯一文本），另留138条唯一提示词与开发、PickScore训练/测试逐字去重；不使用保留集重新选参数。',
        '本轮不同训练种子的上升与下降均保留；旧 MH/PCMH 是不同方法，不能冒充本轮路径比成绩。',
        '参考链两次未移动不等于加权均值不变；候选池权重 ESS 不等于独立后验样本 ESS。',
        '每项提示词 bootstrap 仅反映给定训练模型的提示词抽样误差，不代表训练种子稳定性。',
        '68项数学/数值测试通过；真实A100训练、原生FP32 LoRA和评分数值路径另有审计。',
        '当前入口明确限制单卡/batch1，建议先确认多种子收益与粒子额外贡献，再做2–4卡同步/批处理验证；不直接启动16卡。',
        '', '四、文件位置',
        '更新论文：'+str(OUT/'RTDMD_v31.tex')+' 与 '+str(OUT/'RTDMD_v31.pdf'),
        '完整方法说明：'+str(ROOT/'docs/RTDMD_PathRatio_Repair_CN.txt'),
        '本轮详细分析与下一步：'+str(ROOT/'docs/RTDMD_PathRatio_Conclusions_CN.txt'),
        '所有已完成指标：'+str(OUT/'RTDMD_v31_results.csv'),
        '逐图分数/配置/训练日志：'+str(R/'evidence'),
        '原始FP32 EMA权重：'+str(R/'archived-checkpoints'),
        '完整PNG持久归档：'+str(R/'archived-images')+'，具体哪些已保存以各 image-storage-manifest.json 为准。',
        '其余运行图片/未选权重在 runs 指向的内存盘，重启会丢；逐图分数和已归档权重不随内存盘丢失。',
        '公共PickScore、SD3和部分评分模型缓存也在内存盘；重启后按 dependency-fingerprints.json 及恢复说明重建。',
        '保活仅到本轮01:20 UTC截止，CPU统计按实际获配13核；保活负载不算算法训练收益。',
    ]
    (OUT/'RTDMD_v31_修改与结果说明.txt').write_text('\n'.join(lines)+'\n')
    trainer, config, extra = ENTRIES[best]
    command = ['python', 'scripts/run_deterministic_training.py', trainer, '--config='+config,
        '--config.run_name=rtdmd-v31-user-repeat-seed42', '--config.seed=42',
        '--config.save_dir=logs/path-ratio-round1/runs', '--config.eval.dataset=dataset/drawbench',
        '--config.eval.num_prompts=8', '--config.eval.final_num_prompts=64', '--config.eval_freq=1000',
        '--config.save_freq=1000', *extra]
    code = ['RTDMD v31：怎样找到与运行代码', '',
            '主训练入口：'+str(ROOT/trainer), '主配置：'+config,
            '原始DMD入口未替换：'+str(ROOT/'scripts/train_dmd_sd3.py'), '',
            '在现有开发机项目环境重新训练（run_name需始终唯一，避免覆盖）：',
            'cd '+str(ROOT), 'source jobs/local_env.sh',
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1', shlex.join(command), '',
            '保存的EMA权重可直接评测，不再做额外EMA运算：',
            shlex.join(['bash', 'jobs/evaluate_loaded_path_checkpoint_v2.sh', str(R/'archived-checkpoints'/f'sd3-det-{best}-seed42-ema1000/lora/generator'),
                        'rtdmd-v31-user-pick2048-seed42', 'dataset/pickscore', '2048', 'RTDMD-v31']), '',
            '已完成模型示例目录：'+str(R/'archived-checkpoints'/f'sd3-det-{best}-seed42-ema1000/lora/generator'),
            '这些是推理EMA权重，未保存完整优化器/RNG状态，不能声称精确断点续训。', '',
            '论文TeX为展开后的独立源文件；可用XeLaTeX/Tectonic与Fandol字体编译。PDF已经实际编译和逐页验证。',
            '如果应用的PDF链接没有响应，可打开同目录TeX的LaTeX预览，或在开发机文件系统取回PDF/ZIP。',
            '代码补丁包需要现有ReFormFlow项目及对应模型/依赖；不包含大模型和凭据。完整版本/模型指纹见 dependency-fingerprints.json。',
            '本轮一次性队列写有固定截止时间；复现实验请使用上述训练入口，不照搬已过期的队列截止日期。']
    for case in ['dmd',faithful['case']]:
        checkpoint=R/'archived-checkpoints'/f'sd3-det-{case}-seed42-3000-confirm-ema3000/lora/generator'
        if not (checkpoint/'adapter_model.safetensors').exists():
            continue
        code += ['', '已保存的固定3000轮原生EMA：'+str(checkpoint),
                 '本轮同模型2048张PNG已生成；若内存盘仍在，下轮可只补四项评分，省去重新生成：',
                 shlex.join(['python','scripts/score_rtdmd_path_is_images.py','--run-dir',
                    str(R/'runs'/f'sd3-det-{case}-seed42-3000-confirm-pick2048-exact'),
                    '--step','1','--expected-count','2048','--batch-size','8']),
                 shlex.join(['python','scripts/finalize_existing_path_scores_cpu.py','--evaluation-run',
                    f'sd3-det-{case}-seed42-3000-confirm-pick2048-exact']),
                 '下轮获准预算后可从此完整复评2048提示词；本命令不自动执行：',
                 shlex.join(['bash','jobs/evaluate_loaded_path_checkpoint_v2.sh',str(checkpoint),
                    f'rtdmd-v31-{case}-3000-user-repeat','dataset/pickscore','2048',f'RTDMD-v31-{case}-3000'])]
    if faithful['case'] != best:
        faithful_trainer,faithful_config,_ = ENTRIES[faithful['case']]
        code += ['', '更贴近原稿的纯Teacher选项：'+faithful['case'], '入口：'+str(ROOT/faithful_trainer),
                 '配置：'+faithful_config, '原生EMA目录：'+str(R/'archived-checkpoints'/f'sd3-det-{faithful["case"]}-seed42-ema1000/lora/generator')]
    ft, fc, fe = ENTRIES[faithful['case']]
    faithful_command = ['python', 'scripts/run_deterministic_training.py', ft, '--config='+fc,
        '--config.run_name=rtdmd-v31-pure-teacher-user-repeat-seed42', '--config.seed=42',
        '--config.save_dir=logs/path-ratio-round1/runs', '--config.eval.dataset=dataset/drawbench',
        '--config.eval.num_prompts=8', '--config.eval.final_num_prompts=64', '--config.eval_freq=1000',
        '--config.save_freq=1000', *fe]
    faithful_guide = ['保留纯Teacher路径比核心的代码入口', '',
        '所选纯Teacher版本：'+faithful['case'], '训练入口：'+str(ROOT/ft), '配置：'+fc,
        '保留Teacher CPS字段、完整Student/Teacher正方差路径比和观测似然；条件i-SIR更新最后一个随机过渡，不给确定性末步加噪。',
        '候选前缀固定，其核比是共同条件常数；本版只部分平均后验，更早路径和停止点不混合。',
        '这是必要测度/后验修正后的方法；并非原稿全部采样步骤保持不变。质量是否胜出以匹配种子结果和控制实验为准。', '',
        'cd '+str(ROOT), 'source jobs/local_env.sh',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
        shlex.join(faithful_command), '',
        '原生FP32 EMA目录：'+str(R/'archived-checkpoints'/f'sd3-det-{faithful["case"]}-seed42-ema1000/lora/generator'),
        '开发指标排名候选另见完整代码说明；若为Bridge，其实际提议已改变，不冒充纯Teacher版本。']
    (OUT/'RTDMD_v31_纯Teacher入口.txt').write_text('\n'.join(faithful_guide)+'\n')
    code = ['以保留纯Teacher路径为目标，先看 RTDMD_v31_纯Teacher入口.txt。下面记录固定的开发指标排名候选及其复现命令。', '']+code
    (OUT/'RTDMD_v31_代码与运行说明.txt').write_text('\n'.join(code)+'\n')
    print('Wrote result CSV, matched-seed summary, change report and runnable code guide')


if __name__ == '__main__':
    main()
