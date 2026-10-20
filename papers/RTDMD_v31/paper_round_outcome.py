"""Lead with completed independent-test evidence, including failed replications."""
import json
from paper_bridge_addenda import LABELS, METRICS


def render(root):
    selected = root/'deterministic-selection.json'
    if not selected.exists():
        return '', {}, [], []
    choice = json.loads(selected.read_text())
    best = choice['selected_case']
    rows, reports, groups, values, seeds = [], {}, [], [], []
    for seed in [42, 43, 44]:
        tables = {}
        for case in ['dmd', 'dmd_fp32', best]:
            name = f'sd3-det-{case}-seed{seed}-pick2048-exact'
            p = root/'evidence'/name
            status = p/'status.txt'
            if (not status.exists() or status.read_text().strip() != 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION'):
                break
            r = json.loads((p/'final-metrics.json').read_text())
            assert r['num_images'] == 2048
            tables[case] = r['table']
        if len(tables) != 3:
            continue
        seeds.append((seed, tables))
        for case in ['dmd', 'dmd_fp32', best]:
            name = f'sd3-det-{case}-seed{seed}-pick2048-exact'
            reports[name] = str(root/'evidence'/name/'final-metrics.json')
            rows.append(f'{seed} & 4 & '+LABELS[case]+' & '+' & '.join(f'{tables[case][k]:.4f}' for k in METRICS)+r' \\')
    if not seeds:
        return '', {}, [], []
    title = '本轮完整测试结论：与两种 DMD 对照'
    gains = {k:sum(t[best][k]-t['dmd'][k] for _, t in seeds)/len(seeds) for k in METRICS}
    paired_wins = all(all(t[best][k] > max(t['dmd'][k], t['dmd_fp32'][k])
                          for k in ['Pick Score', 'HP Score']) for _, t in seeds)
    quality_guards = all(t[best]['CLIP Score'] >= max(t['dmd']['CLIP Score'], t['dmd_fp32']['CLIP Score'])-.10
                         and t[best]['Aesthetic Score'] >= max(t['dmd']['Aesthetic Score'], t['dmd_fp32']['Aesthetic Score'])-.03
                         for _, t in seeds)
    if len(seeds) >= 2 and paired_wins and quality_guards:
        conclusion = f'{len(seeds)}个已测训练种子均观察到 Pick/HPS 高于两种 DMD 对照；仍需更多种子和本细节版长训练确认。'
    elif len(seeds) >= 2 and paired_wins:
        conclusion = f'{len(seeds)}个已测训练种子的 Pick/HPS 均上升，但 CLIP/Aesthetic 守门条件没有全部保持，尚未满足整体目标。'
    elif len(seeds) >= 2:
        conclusion = '训练种子复核存在未胜出的对比，尚未建立两个偏好指标的稳定优势；不能只列较好的一次。'
    else:
        conclusion = '本页仅含一个完整训练种子；即使均值上升，也不能据此认定训练稳定性。'
    if choice['status'] != 'QUALIFIED_DEV_CANDIDATE':
        conclusion = '开发守门条件没有候选全部通过；本表为固定诊断候选的后续验证，不能称为达标方法。'+conclusion
    delta = '；'.join(label+f' {gains[k]:+.4f}' for label, k in
                     [('CLIP','CLIP Score'),('Aesthetic','Aesthetic Score'),('Pick','Pick Score'),('HPS','HP Score')])
    for v in gains.values():
        values.append(dict(title_contains=title, value=f'{v:+.4f}', sources=list(reports.values()),
                           formula='Mean across the completed paired training seeds: candidate score minus original DMD score'))
    tex = r'\begin{frame}{'+title+r'''}
\scriptsize\centering
\begin{tabular}{cclrrrr}\toprule
Seed & NFE & Post-Train & CLIP$\uparrow$ & Aesthetic$\uparrow$ & Pick$\uparrow$ & HPS$\uparrow$\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\par\raggedright\vspace{5pt}
候选减原 DMD 的已完成种子均值差：'''+delta+r'''。
\begin{alertblock}{结论的范围}
'''+conclusion+r'''
\end{alertblock}
完整2048提示词、相同推理噪声与四项模型；开发分数及旧 MH/PCMH 成绩不混入本表。
\end{frame}
'''
    groups.append(dict(title_contains=title, names=list(reports)))
    return tex, reports, groups, values
