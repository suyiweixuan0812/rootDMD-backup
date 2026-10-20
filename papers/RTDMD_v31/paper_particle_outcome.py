"""Report the particle contribution separately from the changed surrogate."""
import json
from paper_bridge_addenda import LABELS, METRICS


def render(root):
    path = root/'deterministic-selection.json'
    if not path.exists(): return '', {}, [], []
    choice = json.loads(path.read_text())
    pairs = [(choice['selected_case'], choice['paired_control'])]
    faithful = choice['pure_teacher_alternative']
    if faithful['case'] != choice['selected_case']:
        pairs.append((faithful['case'], faithful['paired_control']))
    reports, values, rows, results = {}, [], [], {}
    title = '粒子平均的额外收益：匹配无粒子对照'
    for case, control in pairs:
        results[case] = []
        for seed in [42, 43, 44]:
            data, names = [], []
            for current in [case, control]:
                name = f'sd3-det-{current}-seed{seed}-pick2048-exact'
                folder = root/'evidence'/name
                state = folder/'status.txt'
                if not state.exists() or state.read_text().strip() != 'COMPLETED_UNCHANGED_CHECKPOINT_EVALUATION': break
                report = json.loads((folder/'final-metrics.json').read_text())
                if 'checkpoint_training_configuration' not in report['protocol']: break
                assert report['num_images'] == 2048
                data.append(report); names.append(name)
            if len(data) != 2: continue
            for key in ['posterior_consistency', 'posterior_time_weight', 'test_file_sha256', 'seed', 'resolution']:
                assert data[0]['protocol'][key] == data[1]['protocol'][key], (case, control, key)
            assert data[0]['protocol']['path_is'].get('query_views', 1) == data[1]['protocol']['path_is'].get('query_views', 1)
            delta = {k:data[0]['table'][k]-data[1]['table'][k] for k in METRICS}
            results[case].append((seed, delta))
            row = f'{seed} & '+LABELS[case]+' & '+' & '.join(f'{delta[k]:+.4f}' for k in METRICS)+r' \\'
            rows.append(row)
            for k in METRICS:
                values.append(dict(title_contains=title, value=f'{delta[k]:+.4f}', sources=names,
                                   formula='Particle method minus its matched paired-source control, same trained seed and full2048 prompts'))
            for name in names: reports[name] = str(root/'evidence'/name/'final-metrics.json')
    if not rows: return '', {}, [], []
    conclusions = []
    for case, data in results.items():
        if not data: continue
        both = all(d['Pick Score'] > 0 and d['HP Score'] > 0 for _, d in data)
        if len(data) < 2:
            phrase = '仅一组已完成，尚不足判断训练种子稳定性'
        elif both:
            phrase = f'{len(data)}个已测种子的 Pick/HPS 均高于匹配控制，仍需更多种子与代价验证'
        else:
            phrase = '未在所有已测种子同时超过匹配控制，尚未建立粒子的额外稳定收益'
        conclusions.append(LABELS[case]+'：'+phrase+'。')
    tex = r'\begin{frame}{'+title+r'''}
\scriptsize\centering
\begin{tabular}{clrrrr}\toprule
Seed & 路径版减匹配控制 & $\Delta$CLIP & $\Delta$Aes & $\Delta$Pick & $\Delta$HPS\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\par\raggedright\vspace{5pt}
'''+r'\par '.join(conclusions)+r'''
\begin{itemize}\itemsep4pt
\item 控制保留同查询 Jacobian、精度、正则、时间权重及观测次数，只去掉条件粒子平均。
\item 相对 DMD 的收益不能全部归于路径比；控制与粒子版的训练计算量不同。
\item 每行使用完整2048提示词；不同历史方法与开发集不参与此表。
\end{itemize}
\end{frame}
'''
    return tex, reports, [], values
