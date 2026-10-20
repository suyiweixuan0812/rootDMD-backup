"""Summarize recorded utilization; keepalive load is not algorithm performance."""
from pathlib import Path
from datetime import datetime,timezone
import json,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent;R=ROOT/'logs/path-ratio-round1'
p=OUT/'development-load-guard/resource-monitor.jsonl'
rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
assert rows
fields={'CPU_percent_of_allocated13':'visible_container_cpu_percent_of_allocated','GPU_utilization_percent':'gpu_utilization_percent','rolling12_CPU':'mean_last12_cpu','rolling12_GPU':'mean_last12_gpu'}
summary={}
for name,key in fields.items():
 values=[float(x[key]) for x in rows if x.get(key) is not None]
 summary[name]=dict(mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values),fraction_samples_at_least50=sum(v>=50 for v in values)/len(values))
result=dict(first_recorded_utc=datetime.fromtimestamp(rows[0]['timestamp'],timezone.utc).isoformat(),last_recorded_utc=datetime.fromtimestamp(rows[-1]['timestamp'],timezone.utc).isoformat(),sample_count=len(rows),nominal_sample_interval_seconds=5,allocated_cpus=13,round_deadline_utc='2026-10-21T01:20:00+00:00',summaries=summary,scope='All visible processes sampled by nvidia-smi/procfs; independent jobs share one A100. Keepalive CPU duty and idle GPU matmul are excluded from algorithmic speed claims. Snapshots can briefly be below50 during loading; report measured samples rather than promise every instant.')
(R/'resource-utilization-summary.json').write_text(json.dumps(result,indent=2))
lines=['本轮预算与资源记录','固定截止：2026-10-21 01:20 UTC（北京时间09:20）','单张A100-SXM4 80GB；训练及评测采用独立进程并发，不改变每项实验batch1。','CPU统计按实际获配13核；保活CPU为nice19，GPU占位在正式CUDA进程存在时让出GPU。','保活只到本轮截止，不能算作算法训练成果或效率提升。',f'已记录{len(rows)}个约5秒样本；时间范围 {result["first_recorded_utc"]} 至 {result["last_recorded_utc"]}。']
for key,value in summary.items():lines.append(key+'：'+json.dumps(value,ensure_ascii=False))
(OUT/'RTDMD_v31_资源与预算.txt').write_text('\n'.join(lines)+'\n')
print('Resource samples',len(rows),'meanCPU',summary['CPU_percent_of_allocated13']['mean'],'meanGPU',summary['GPU_utilization_percent']['mean'])
