OCR / GenEval：SD3-Medium，无奖励 DMD，每套 16 卡、10000 步。

| 数据集 | 状态 | 已核实步数 | CLIP Score | Aesthetic Score | Pick Score | HP Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ocr | Running | 3865 | 待完成 | 待完成 | 待完成 | 待完成 |
| geneval | Queueing | — | 待完成 | 待完成 | 待完成 | 待完成 |

更新时间：2026-10-21T02:09:25.932716+08:00。

两个任务分别使用原始 SD3-Medium 初始化；原生 OCR / GenEval 专项分数不在本次四项指标内。

ocr 任务：`t-20261021011436-d5rtf`；vePFS：`/vepfs-mlp2/c20250506/250404014/rootDMD/sd3-dmd-ocr-geneval-16gpu-10000-20261021-011401/outputs/sd3-dmd-ocr-16gpu-10000`。

geneval 任务：`t-20261021011501-rlmj5`；vePFS：`/vepfs-mlp2/c20250506/250404014/rootDMD/sd3-dmd-ocr-geneval-16gpu-10000-20261021-011401/outputs/sd3-dmd-geneval-16gpu-10000`。

每套任务的权重：`checkpoints/checkpoint-10000/lora/generator/`；评分：`eval/step-10000/final-metrics.json`（另有 CSV、Markdown 和逐图分数）。
