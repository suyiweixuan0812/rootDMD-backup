"""Bridge theory and completed deterministic results, read directly from evidence."""
import json

METRICS = ['CLIP Score', 'Aesthetic Score', 'Pick Score', 'HP Score']
LABELS = {'dmd': 'DMD', 'dmd_fp32': 'DMD, FP32 query', 'base': 'Path-IS',
          'time': 'Path-IS + time', 'paired': 'One-source control',
          'paired_time': 'One-source + time', 'bridge': 'Bridge',
          'bridge_time': 'Bridge + time', 'adaptive_bridge': 'Past-KL bridge',
          'adaptive_bridge_time': 'Past-KL + time', 'last':'Teacher, last block',
          'last_time':'Teacher, last + time', 'last_q2_time':'Teacher, 2 queries', 'paired_q2_time':'One-source, 2 queries'}


def render(root):
    reports, groups, computed = {}, [], []
    tex = r'''\begin{frame}{ESS 的第二个瓶颈：长路径先验失配}
\small
已有 SD3 诊断中，长路径的观测对数似然跨度可小于1，但 Teacher/Student
\textbf{路径先验比}跨度仍可超过100；只扩大坐标块或重复8次更新不能保证改善。
\begin{align}
\log w&=\sum_{k<J}\log(M_\theta/M_T)+\log L_\sigma,\\
\KL\{\N(\mu_S,\nu^2I)\Vert\N(\mu_T,\nu^2I)\}
&=\frac{\|\mu_S-\mu_T\|^2}{2\nu^2}.
\end{align}
\begin{itemize}\itemsep5pt
\item latent 各坐标的差异进入完整平方范数；维度和路径长度都可能放大失配。
\item 合理权重与正确后验并不保证有限候选池中有足够多的有效新候选。
\item 必须同时报告候选 ESS、实际读出移动、重复值及加权候选方差。
\item CFG4.5 Teacher clean 值沿用 DMD 的学习参考；未证明它等于 Teacher 四步 CPS 源的精确条件均值。
\end{itemize}
\end{frame}

\begin{frame}{为什么直接增加少量粒子难以解决高维失配}
\small
连续单步等协方差 Gaussian 的精确计算：令
$a=\|\mu_S-\mu_T\|^2/\nu^2$，$W=M_S/M_T$，则
\begin{align}
\log W\mid M_T&\sim\N(-a/2,a),\\
\E_{M_T}[W]&=1,\quad \E_{M_T}[W^2]=e^a,
\quad\mathrm{CV}^2(W)=e^a-1.
\end{align}
\begin{itemize}\itemsep5pt
\item 小的逐坐标均值差也会累加到 $a$；权重方差可迅速增长。
\item 这是固定父状态的理想单步公式，不是非线性整条 BF16 路径的闭式 ESS 公式。
\item 把8粒子增加几倍不保证覆盖目标高概率区域；16张卡也不会自动修复提议失配。
\item 后续两条细节路线：修正提议以减小失配，或固定更多上下文只做可控的条件平均。
\end{itemize}
\end{frame}

\begin{frame}{保留路径比主线的扩展：显式修正桥接提议}
\small
Teacher-only 版本保留为 $\gamma=0$ 基线。可选改进使用同一正方差 CPS 核：
\begin{align}
v_Q&=v_T+\gamma(v_\theta-v_T),\qquad 0\leq\gamma\leq1,\\
w_Q&=\frac{p_J\phi}{q_Jq_0}\prod_{k<J}
\underbrace{\frac{M_\theta}{M_T}}_{\text{原 Student/Teacher 比}}
\underbrace{\frac{M_T}{Q_\gamma}}_{\text{新增提议校正}}\ L_\sigma
=\frac{p_J\phi}{q_Jq_0}\prod_{k<J}\frac{M_\theta}{Q_\gamma}L_\sigma.
\end{align}
\begin{itemize}\itemsep5pt
\item Student 源分布、共同末端读出、观测后验和条件 i-SIR 目标保持。
\item 真正采样的提议已变为 $Q_\gamma$；不能仍用未校正的纯 $M_\theta/M_T$ 权重。
\item 日志分别保存原比和提议校正；实际用直接 $\log M_\theta-\log Q$ 避免大数相减。
\item 当 $\gamma$ 很大时，提议明显接近 Student；不能描述成完全未改动的教师提议。
\end{itemize}
\end{frame}

\begin{frame}{桥接为何可减小先验失配：结论与适用范围}
\small
理想连续等方差 Gaussian CPS 中，均值对速度线性：
\begin{align}
\mu_Q&=(1-\gamma)\mu_T+\gamma\mu_S,\\
\KL(M_S\Vert Q_\gamma)&=(1-\gamma)^2\KL(M_S\Vert M_T).
\end{align}
固定 $J$ 和共同初始分布时，用 KL 链式分解得到相应路径先验等式。
\begin{itemize}\itemsep5pt
\item BF16 舍入后只可由数据处理得到 Gaussian KL 的\textbf{上界}；不再声称严格缩放等式。
\item 这不是后验 KL、全局混合或图像质量的保证；观测似然仍会造成集中。
\item 固定桥接细节：短路径 $\gamma=.90$，$J\geq3$ 用 $.99$；高噪声扩大局部块并限创新半径 $.5$。
\item 原 Student CPS 的 $\eta=.9$ 和四步推理不变；创新空间半径是训练采样器的新参数。
\end{itemize}
\end{frame}

\begin{frame}{自适应桥接：只根据过去的查询定提议}
\small
当前 $J$ 已知后，先用\textbf{过去查询}的逐节点 Gaussian KL 代理确定提议，
再生成当前 Student 源与观测：
\begin{align}
\bar K_k&\leftarrow\max(.95\bar K_k,K_k^{\rm observed}),\quad
K_J=\sum_{k<J}\bar K_k,\\
\gamma_J&=\begin{cases}0,&K_J\leq .25,\\1-\sqrt{.25/K_J},&K_J>.25.\end{cases}
\end{align}
\begin{itemize}\itemsep5pt
\item 给定历史与 $J$，$\gamma_J$ 在当前成对采样之前固定，原条件后验参考证明仍适用。
\item 该规则在历史代理预算下保留尽可能多的 Teacher 提议成分；本次查询后才更新历史。
\item 不能根据当前参考/候选的 ESS 临时选 $\gamma$，然后直接套用固定提议的证明。
\item 历史代理有滞后，且提示词变化；$.25$ 不是当前真实路径/后验 KL 的保证。
\end{itemize}
\end{frame}
'''
    training_diagnostics = root / 'selected-training-diagnostics.json'
    if training_diagnostics.exists():
        actual = json.loads(training_diagnostics.read_text())['runs']
        lines = []
        title = '纯教师版真实训练：低噪声仍会退化'
        labels = [('positive_prefix_low_noise', r'$<.3$'),
                  ('positive_prefix_medium_noise', r'$[.3,.9)$'),
                  ('positive_prefix_high_noise', r'$\geq.9$')]
        for seed in [42, 43]:
            selected = [x for x in actual if x['run'] == f'sd3-det-last-seed{seed}']
            if not selected:
                continue
            groups_by_name = {x['label']: x for x in selected[0]['groups']}
            for key, label in labels:
                if key not in groups_by_name:
                    continue
                row = groups_by_name[key]
                values = [str(seed), label, str(row['count']),
                    f"{row['ess']['mean']:.3f}",
                    f"{row['actual_move_rate']['mean']:.3f}",
                    f"{row['rb_removed_variance_trace_per_coordinate']['mean']:.6f}"]
                lines.append(' & '.join(values) + r' \\')
                computed.extend(dict(title_contains=title, value=v, source=str(training_diagnostics))
                    for v in values[2:])
        tex += r'\begin{frame}{' + title + r'''}
\scriptsize\centering
仅统计 $J>1$ 的真实生成器更新；每次8候选、2次条件更新，均值取自训练日志。
\begin{tabular}{ccrrrr}\toprule
Seed & $\sigma$ & 更新数 & ESS & 参考移动率 & RB方差迹/坐标\\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}\par\raggedright\vspace{6pt}
\begin{itemize}\itemsep4pt
\item 低噪声的观测似然仍可使候选退化，纯Teacher细节修订没有彻底解决此问题。
\item 高噪声时 ESS 约6，但读出加权方差仍小；权重均匀不等于显著消除训练噪声。
\item $J=1$ 无路径转移，另存统计；不把它混入正方差路径比的改善结论。
\item 这些为有限训练样本的描述统计；不同噪声组不能直接作因果比较，RB迹也不是梯度方差。
\end{itemize}
\end{frame}
'''
        clip_title = '粒子以外的瓶颈：优化接口与梯度裁剪'
        clip_lines = []
        for seed in [42, 43]:
            for case in ['dmd', 'dmd_fp32', 'paired', 'last']:
                rows = [x for x in actual if x['run'] == f'sd3-det-{case}-seed{seed}']
                if not rows:
                    continue
                row = rows[0]
                value = f"{100 * row['gradient_clip_fraction']:.1f}"
                clip_lines.append(f"{seed} & {LABELS[case]} & {row['generator_updates']} & {value}" + r' \\')
                computed.append(dict(title_contains=clip_title, value=value, source=str(training_diagnostics)))
        tex += r'\begin{frame}{' + clip_title + r'''}
\scriptsize\centering
\begin{tabular}{clrr}\toprule
Seed & 训练版本 & 生成器更新数 & 裁剪率\%\\\midrule
''' + '\n'.join(clip_lines) + r'''
\bottomrule\end{tabular}\par\raggedright\vspace{6pt}
裁剪率按裁剪前梯度范数 $>1$ 的实际更新占比计算，不含未更新轮次。
\begin{itemize}\itemsep4pt
\item 同查询 Jacobian 与一致性正则改变更新；即使无粒子控制也明显更常触发裁剪。
\item 这是优化差异的证据，尚不能单独确定哪一项导致质量波动。
\item 正确后验均值、较高 ESS 或较小估计方差，均不能推出完整 DMD KL 梯度或 Adam 后质量改善。
\end{itemize}
\end{frame}
'''
    diagnostics = root / 'bridged-exact-geometry-diagnostics/diagnostics.json'
    teacher = root / 'teacher-capped-geometry-diagnostics/diagnostics.json'
    if diagnostics.exists() and teacher.exists():
        a = json.loads(teacher.read_text())['rows']
        b = json.loads(diagnostics.read_text())['rows']
        lines = []
        for sigma in [.95, .98]:
            for block in [16, 4096]:
                vals = []
                for data in [a, b]:
                    r = [x for x in data if x['source_stop'] == 4 and x['sigma'] == sigma and x['active_block_size'] == block]
                    assert len(r) == 1
                    vals.append(r[0])
                s, q = vals
                values = [f'{sigma:.2f}', str(block), f'{s["path_is_ess"]:.2f}',
                          f'{q["path_is_ess"]:.2f}', f'{q["path_is_path_ratio_span"]:.2f}',
                          f'{q["rb_removed_variance_trace_per_coordinate"]:.5f}']
                lines.append(' & '.join(values) + r' \\')
                computed.extend(dict(title_contains='长路径采样诊断', value=v, source=str(diagnostics)) for v in values[3:])
        tex += r'''\begin{frame}{长路径采样诊断：改善 ESS，不替代质量实验}
\footnotesize\centering
同一保存的 Student、固定开发提示词、$J=4$、8候选；表中为候选池权重 ESS。
\begin{tabular}{rrrrrr}\toprule
$\sigma$ & 块大小 & Teacher ESS & Bridge ESS & Bridge 比跨度 & RB 方差迹/坐标\\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}\par\raggedright\vspace{5pt}
\begin{itemize}\itemsep5pt
\item $\gamma=.99$；原 FP32 LoRA 逐张量精确载入，实际 BF16-autocast 场重验一致。
\item 每行是固定查询的一次局部池；不能外推为训练全过程的独立样本 ESS。
\item RB 数值为候选池加权读出的每坐标方差；不是后验总方差的消除比例。
\item 低噪声仍可能冻结。采样改善与最终 PickScore/HPS 必须分开验证。
\end{itemize}
\end{frame}
'''
    tex += r'''\begin{frame}{更小的改动：固定最后一个正方差创新块}
\small
纯 Teacher CPS 映射 $H=T_T(E)$ 与原权重 $w_{ST}(E)=\prod(M_S/M_T)L$ 保留。
固定更早的创新，只更新最后正方差转移（$J=1$ 时更新初始创新）的一组坐标 $B$：
\begin{align}
\pi(E_B\mid E_{-B},y,J)&\propto\phi_B(E_B)w_{ST}(E),\\
C_B\mid E_B&\sim\N(E_B,\delta^2 I),\quad E'_B\mid C_B\sim\N(C_B,\delta^2 I).
\end{align}
\begin{equation}
\log\widetilde w_i=\log w_{ST}(E_i)
-\tfrac12\{\|E_{i,B}\|^2-\|E^*_{B}\|^2\}.
\end{equation}
两个对称辅助 Gaussian 密度相消；\textbf{标准 Gaussian 先验不能再省略}。
包含正确条件参考的 i-SIR 保持原后验；没有桥接场、MH 接受率或新的末端噪声。
\end{frame}

\begin{frame}{最后块版本究竟保留了多少路径作用}
\small
固定更早父状态 $h$ 后，前缀比是所有候选的共同常数，只剩最后随机核的校正。
连续理想模型在物理坐标 $u$ 下满足
\begin{equation}
q_T(u\mid h)\frac{M_S(u\mid h)}{M_T(u\mid h)}
L_\sigma(y\mid D_\theta(u))
=M_S(u\mid h)L_\sigma(y\mid D_\theta(u)).
\end{equation}
\begin{itemize}\itemsep5pt
\item Teacher 路径比的作用是校正提议；它不会给 Student 后验额外增加一个教师目标。
\item 该局部版保留比值与条件平均框架，却没有跨整条路径利用教师候选的全部探索能力。
\item BF16 下实现保留单元质量比和单元内 Gaussian 提升密度；不能按连续式直接删除这些项。
\item Teacher CFG4.5 仍是训练 score 的参考；此作用与条件采样提议的作用不同。
\end{itemize}
\end{frame}

\begin{frame}{最后正方差块：收益、局限与精确缓存}
\small
\begin{itemize}\itemsep5pt
\item 早期创新的扰动会改变后续父状态和多个核比；固定这些父状态可避免这部分传播。
\item 单独改对称辅助、仍随机选创新层时，四个 $J=4$ SD3 诊断池仍 ESS=1；保留此负结果。
\item 固定最后层后，同组诊断 ESS 约6.25--7.51；更大半径2未带来一致方差收益，采用 $.5$。
\item 更早前缀与 $J$ 不混合，因此只部分 Rao--Blackwell 化；不能把这一采样改善等同于完整后验方差已消除。
\item 不变前缀网络输出只算一次；原缓存版本逐张量检查父状态相同。物理坐标实现直接固定前缀，逐查询重建。
\item 八个真实 SD3 查询池的 ESS、核比、移动和方差在缓存前后逐位一致；吞吐数字需另做隔离基准。
\end{itemize}
\end{frame}

\begin{frame}{新对照的选型与独立验证规则}
\small
\begin{itemize}\itemsep5pt
\item 十四组细节实验使用同一确定性运行设置、1000轮/100次生成器更新、DrawBench64 开发评测。
\item 同时匹配原 DMD 和 FP32 观测噪声 DMD；逐指标取两者较大值作为开发参考。
\item Pick/HPS 均需上升，CLIP 不低于参考 $-.10$、Aesthetic 不低于 $-.03$；排序预先固定。
\item 无粒子的成对源仅是归因对照，不能替代所请求的路径比方法。
\item 固定候选后再跑 PickScore2048、新种子和138条唯一未见 DrawBench 提示词；不把新测试结果回灌调参。
\item 原 PickScore 测试已用于此前阶段汇报，不称为全程盲测；另保留138条独立提示词。
\end{itemize}
\end{frame}
'''
    def table(title, items, note):
        nonlocal tex
        completed = []
        for label, name in items:
            path = root / 'evidence' / name / 'final-metrics.json'
            status = path.parent/'status.txt'
            if path.exists() and status.exists() and status.read_text().strip().startswith('COMPLETED'):
                report = json.loads(path.read_text())
                completed.append(('4 & ' + label + ' & ' + ' & '.join(f'{report["table"][k]:.4f}' for k in METRICS) + r' \\', name))
                reports[name] = str(path)
        if not completed:
            return
        tex += r'\begin{frame}{' + title + r'''}
\scriptsize\centering
\begin{tabular}{clrrrr}\toprule
NFE & Post-Train & CLIP$\uparrow$ & Aesthetic$\uparrow$ & Pick$\uparrow$ & HPS$\uparrow$\\\midrule
''' + '\n'.join(row for row, _ in completed) + r'''
\bottomrule\end{tabular}\par\raggedright\vspace{6pt}
''' + note + '\n' + r'\end{frame}' + '\n'
        groups.append(dict(title_contains=title, names=[name for _, name in completed]))
    for title, cases in [
            ('确定性开发集：对照与基线', ['dmd', 'dmd_fp32', 'base', 'paired', 'paired_time']),
            ('确定性开发集：采样与时间细节', ['time', 'bridge', 'bridge_time', 'adaptive_bridge', 'adaptive_bridge_time']),
            ('确定性开发集：纯教师最后正方差块', ['last', 'last_time', 'last_q2_time', 'paired_q2_time'])]:
        table(title, [(LABELS[c], f'sd3-det-{c}-seed42') for c in cases],
              r'仅 DrawBench64 开发结果；不同于 PickScore2048。所有已完成的预先声明项目均保留，不能仅列最好的一组。')
    selection = root / 'deterministic-selection.json'
    if selection.exists():
        choice = json.loads(selection.read_text())
        cases = ['dmd', 'dmd_fp32', choice['selected_case'], choice['paired_control']]
        faithful = choice.get('pure_teacher_alternative')
        if faithful and faithful['case'] not in cases:
            cases.append(faithful['case'])
            if faithful['paired_control'] not in cases:cases.append(faithful['paired_control'])
        for title, seed, suffix, note in [
                ('确定性完整 PickScore 测试：seed42', 42, 'pick2048-exact', '固定开发候选后的2048条完整测试；同推理噪声与四项评分模型。'),
                ('保留提示词验证：138条唯一提示词', 42, 'holdout138-exact', '与开发集及 PickScore 训练/测试逐字去重；结果不用于重新选择参数。'),
                ('确定性训练种子复核：seed43', 43, 'pick2048-exact', '已固定的方法重新训练；单提示词 bootstrap 区间不能代替训练种子不确定性。'),
                ('固定第三训练种子：seed44', 44, 'pick2048-exact', '本组仅由预先固定的剩余时间门槛触发；不依据前两组评分决定是否运行，也不再次调参。')]:
            active = cases
            table(title, [(LABELS[c], f'sd3-det-{c}-seed{seed}-{suffix}') for c in active], note)
    table('原路径版长训练确认：10000轮',
          [('DMD', 'dmd-seed42-10000-pick2048'), ('Path-IS', 'path-is-seed42-10000-pick2048')],
          r'原先固定的 $\lambda=.10$、Teacher-only、无时间权重；双方10000轮/1000次生成器更新。此组 CUDA 后向未启用新确定性设置，不能与新队列混为同组。')
    optional = root/'optional-faithful-long-progress.json'
    if optional.exists():
        longer = json.loads(optional.read_text())
        if longer.get('status') == 'OPTIONAL_FAITHFUL_LONG_CHECK_COMPLETE':
            table('新纯教师细节版：固定3000轮复核',
                  [(LABELS.get(c, c), name) for c, name in zip(
                      ['dmd', longer['faithful_case']], longer['reports'])],
                  r'预先固定的3000轮端点，不按中间评分选checkpoint；同生成器更新次数。双查询若启用有额外计算，本组没有3000轮FP32与无粒子控制。')
    prefix = root/'deterministic-full-prefix-reproduction.json'
    live_prefix=root/'deterministic-dmd1000-prefix-live-audit.json'
    if prefix.exists() or live_prefix.exists():
        checks = json.loads(prefix.read_text()) if prefix.exists() else [json.loads(live_prefix.read_text())]
        message = '；'.join(x['case'].replace('_', r'\_')+': '+str(x['equal_tensors'])+'/'+str(x['tensor_count']) for x in checks)
        tex += r'''\begin{frame}{完整训练前缀的重复性复核}
\small
新3000轮训练在1000轮处，与此前原生FP32 EMA逐张量比较；本页只列已完成的比较：\par
'''+message+r'''
\begin{itemize}\itemsep5pt
\item 此处报告实际一致数；不把40轮冒烟的重复性推广为完整训练已逐位复现。
\item CPU加载线程数由96限制为1；LoRA形状uniform、训练randperm及normal抽样的单独检查一致，但仍以这里的完整权重比较为准。
\item 纯Teacher入口还跳过无更新轮次的未用文本编码；40轮日志全等，完整权重比较以本表为准。
\item 只保存推理EMA，无完整优化器/RNG状态；3000轮实验从初始化重新训练。
\end{itemize}
\end{frame}
'''
    drift = root/'training-drift-long/training-drift.json'
    if drift.exists():
        data = json.loads(drift.read_text())
        assert data['outer_steps'] == 10000
        baseline = data['methods']['DMD']
        candidate = data['methods']['Path-IS']
        lines = []
        for i in [0, 2, 4, 7, 9]:
            a, b = baseline[i], candidate[i]
            vals = [str(b['outer_step_end']), f'{b["path_is/ess"]:.2f}',
                    f'{b["path_is/path_ratio_span"]:.1f}', f'{100*b["path_is/frozen_update"]:.0f}',
                    f'{100*a["gradient_clip_fraction"]:.0f}', f'{100*b["gradient_clip_fraction"]:.0f}']
            lines.append(' & '.join(vals) + r' \\')
            computed.extend(dict(title_contains='长训练中的提议漂移', value=v, source=str(drift)) for v in vals[1:])
        tex += r'''\begin{frame}{长训练中的提议漂移：直接来自训练日志}
\scriptsize\centering
Teacher-only 原路径版；每行统计截至所列轮次的最近1000轮（100次生成器更新）。
\begin{tabular}{rrrrrr}\toprule
截至轮次 & ESS/8 & 路径比跨度 & 链冻结\% & DMD裁剪\% & Path裁剪\%\\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}\par\raggedright\vspace{6pt}
固定 Teacher 提议随 Student 更新逐渐失配，是后续桥接与最后正方差块消融的动机。
链冻结指两次参考重采样均未移动，加权均值仍可能变化。分噪声/停止点诊断另存；总平均 ESS 不代表所有层均有效。此表不是图像质量或全局收敛证明。
\end{frame}
'''
    cfg_files = [root/'teacher-cfg-drift-dmd10000.json', root/'teacher-cfg-drift-oldpath10000.json']
    if all(p.exists() for p in cfg_files):
        cfg_data = [json.loads(p.read_text()) for p in cfg_files]
        assert all(x['max_sequence_length']==128 and x['source_stop']==4 for x in cfg_data)
        cfg_title = '长训练场诊断：固定教师提议的失配'
        cfg_lines = []
        for label, data, source in zip(['DMD10000', 'Old Path10000'], cfg_data, cfg_files):
            vals=[f"{data['summary'][str(c)]['mean_sum_conditional_Gaussian_KL_per_path']:.1f}" for c in [1.,4.5]]
            vals.append(f"{data['summary']['1.0']['mean_source_path_log_mass_ratio']:.1f}")
            cfg_lines.append(label+' & '+' & '.join(vals)+r' \\')
            computed.extend(dict(title_contains=cfg_title,value=v,source=str(source)) for v in vals)
        tex+=r'\begin{frame}{'+cfg_title+r'''}
\scriptsize\centering
两条旧10000轮模型；每条用2个固定开发提示词、4个种子、$J=4$，文本长度128。
\begin{tabular}{lrrr}\toprule
Student & Gaussian $K$, CFG1 & Gaussian $K$, CFG4.5 & 样本$\log(p_S/q_{T,1})$\\\midrule
'''+ '\n'.join(cfg_lines)+r'''
\bottomrule\end{tabular}\par\raggedright\vspace{5pt}
$K$ 为8条源路径的逐父状态等方差 Gaussian KL 之和的样本均值；最后一列用 BF16 单元质量计算。
\begin{itemize}\itemsep4pt
\item 健康 DMD 也可与固定 Teacher 路径明显失配；崩溃旧路径版在此组的 CFG1 代理约为其43倍。
\item 简单把提议 CFG 改成4.5，在这两组均未降低失配；本轮不据此增加新训练候选。
\item 此为机制诊断，没有新增图像质量实验；不能把先验失配代理当成后验 ESS 或唯一崩溃因果。
\end{itemize}
\end{frame}
'''
    return tex, reports, groups, computed
