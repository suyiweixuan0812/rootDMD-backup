"""Append completed negative results and implementation audits, never invent scores."""
import json
from pathlib import Path


def render(round_root):
    reports={};groups=[];computed=[]
    tex=''
    seed_summary=round_root/'comparison-two-seeds/comparison.json'
    if seed_summary.exists():
        summary=json.loads(seed_summary.read_text());items=[('DMD, seed42','dmd-ema-pick2048-exact'),
            ('Path-IS, seed42','path-is-selected-pick2048-exact'),
            ('DMD, seed43','dmd-seed43-pick2048-exact'),('Path-IS, seed43','path-is-seed43-pick2048-exact')]
        rows=[]
        for label,name in items:
            f=round_root/'evidence'/name/'final-metrics.json';r=json.loads(f.read_text());reports[name]=str(f)
            rows.append('4 & '+label+' & '+' & '.join(f'{r["table"][k]:.4f}' for k in ['CLIP Score','Aesthetic Score','Pick Score','HP Score'])+r' \\')
        d=summary['mean_deltas']
        mean_text='；'.join(label+f' {d[k]:+.4f}' for label,k in [('CLIP','CLIP Score'),('Aesthetic','Aesthetic Score'),('Pick','Pick Score'),('HPS','HP Score')])
        computed=[{'title_contains':'两组训练种子','value':f'{d[k]:+.4f}','source':str(seed_summary)} for k in d]
        tex+=r'''\begin{frame}{两组训练种子：保留未胜出的结果}
\small\centering
\begin{tabular}{clrrrr}\toprule
NFE & Post-Train & CLIP$\uparrow$ & Aesthetic$\uparrow$ & Pick$\uparrow$ & HPS$\uparrow$\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\par\raggedright\vspace{5pt}
两次训练均值差（Path-IS 减 DMD）：'''+mean_text+r'''。
\begin{alertblock}{当前结论}
seed42 的四项均值上升；seed43 的 Pick/HPS/Aesthetic 下降。
两个偏好指标的两次均值小幅改善，\textbf{仍未建立跨训练种子的稳定优势}。
本组包含 CUDA 后向非确定性；不与后续严格确定性新对照混为一组。
\end{alertblock}
\end{frame}
'''
        groups.append({'title_contains':'两组训练种子','names':[n for _,n in items]})
    tex+=r'''\begin{frame}{配置一致与数值差异：需要分别报告}
\small
\begin{itemize}\itemsep5pt
\item 模型、分辨率、LoRA、优化器、LR、EMA、batch、CPS、随机停止和训练步数通过共同配置审计。
\item 网络和源 latent 状态均 BF16；本版显式使用 \textbf{FP32 观测噪声}，原 DMD 的查询噪声继承 BF16。
\item Teacher 提议创新改为 FP64；BF16 核比用舍入单元质量。有限 RNG/算术仍是理想 Gaussian 模型的近似。
\item 原 DMD 每轮先更新 Fake；移除 Fake 后，相同 seed 的 Generator 随机流并不逐个对齐。
\item 外部参数相同不等于逐算子精度相同，也不等于相同训练计算量；上述差异不能隐藏。
\end{itemize}
\end{frame}
'''
    smoke=round_root/'deterministic-smoke-comparison.json'
    if smoke.exists():
        r=json.loads(smoke.read_text())
        tex+=r'''\begin{frame}{训练重复性与推理重复性：实际检查结果}
\small
\begin{itemize}\itemsep6pt
\item 固定保存权重的推理可逐图复现；完整 DMD 重生成 PNG 与旧保存图逐字节相同。
\item 同 seed 重训在第一次后向计算已出现微小梯度差异，发生在第一次评测之前；后续 BF16 轨迹放大差异。
\item 长训练前1000轮与短训练的 LoRA 仅2/382张量逐位相同；差异已在首个反传出现，不能归因于评测频率。
\item 启用确定性 CUDA 与固定 cuBLAS workspace 后，两次20轮小训练的 '''+str(r['equal_tensors'])+r'''/382 张量逐位相同。
\item 这仅是小规模重复检查；后续新对照需让 DMD 与各方法同时使用相同设置，并继续记录完整训练重复性。
\end{itemize}
\end{frame}
'''
    tex+=r'''\begin{frame}{必须加入的对照：没有粒子的成对 Student 源}
\small
在同一个成对训练查询中，$X^*\mid y$ 本身就是一个正确的后验样本：
\begin{equation}
\widehat m_{\rm one}=X^*,\qquad \E[X^*\mid y]=m_\theta(y).
\end{equation}
\begin{itemize}\itemsep6pt
\item 在相同查询 Jacobian、归一化与正则下，关闭全部粒子构成必要的控制。
\item 条件 i-SIR 的额外价值应体现为实际降方差/优化收益；仅比较 DMD 无法隔离这部分贡献。
\item 路径比保证 \textbf{新增 Teacher 候选}的目标正确；不是说单参考估计本身需要路径比才无偏。
\item 该控制不替代本版保留路径比的方法。若控制同样好或更好，应如实报告，不能把所有增益归因于粒子。
\end{itemize}
\end{frame}

\begin{frame}{高噪声梯度尺度：只依赖已知噪声的时间权重消融}
\small
训练日志：Path-IS 两个种子约49\%更新被裁剪，DMD约10--14\%；
高噪声 $\sigma\geq.75$ 的 Path-IS 裁剪约85--90\%。
\begin{align}
D_\theta(y,\sigma)&=y-\sigma v_\theta(y,\sigma),\qquad J_\theta D=-\sigma J_\theta v,\\
\omega(\sigma)&=.30/\sigma,\qquad g_{\lambda,\omega}=\omega(\sigma)g_\lambda.
\end{align}
\begin{itemize}\itemsep4pt
\item $.30$ 只按既有训练梯度选择；固定原状态时预计裁剪9\%/12\%，不使用测试分数拟合。
\item 路径比、观测似然与目标后验保持；$G$ 改为确定的 $\omega G$，条件均值/方差证明仍适用。
\item 该消融改变 surrogate 的时间加权及优化器动态，不能称为与原更新完全相同。
\item 新轨迹的实际裁剪和质量必须重新测量；本页的校准预测不是最终质量结论。
\end{itemize}
\end{frame}
'''
    return tex,reports,groups,computed
