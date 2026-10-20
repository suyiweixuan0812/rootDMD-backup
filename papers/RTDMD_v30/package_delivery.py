"""Package the actual source, PDF and experiment evidence without model caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import difflib
import hashlib
import json

WORK = Path('/rootDMD')
ROOT = WORK / 'ReFormFlow-main'
PAPER = Path(__file__).resolve().parent
OUT = WORK / 'deliverables'
COMP = ROOT / 'logs/rtdmd-pcmh/comparison-v30'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    tex = PAPER / 'RTDMD_SchemeA_Reorganized_Final_CN_v30.tex'
    pdf = tex.with_suffix('.pdf')
    manifest = json.loads((PAPER / 'paper-data-manifest.json').read_text())
    assert manifest['all_results_complete'] and sha(tex) == manifest['new_tex_sha256']
    assert sha(PAPER / 'v29-original-reference.tex') == manifest['source_v29_sha256']
    assert pdf.is_file() and pdf.stat().st_size > 10000
    check = json.loads((PAPER / 'pdf-validation.json').read_text())
    assert check['pdf_sha256'] == sha(pdf) and check['tex_sha256'] == sha(tex)
    assert check['numeric_table_verified'] and check['page_count'] == 32
    s = json.loads((COMP / 'summary.json').read_text())
    primary = s['rows'][3]
    old = s['rows'][2]
    baseline = s['rows'][0]
    keys = ['CLIP Score', 'Aesthetic Score', 'Pick Score', 'HP Score']
    changes = '''RTDMD v30 修改与交付说明（2026-10-21，北京时间）
================================================

结果
  主实验lambda=.25和敏感性lambda=.50均已完成；共生成4096张测试图，各2048条提示词。
  每组1000外层轮/100次生成器更新，NFE=4；一张A100，共享GPU并行，不作速度比较。
  逐图配对、协议一致性、五组均值重算、代码/数据哈希与两份LoRA有限性检查均通过。
  原DMD、Source-IS和上一版Posterior-MH结果未覆盖。
'''
    changes += '  主实验分数：' + '；'.join(f'{k}={primary[k]:.6f}' for k in keys) + '\n'
    changes += '  相对DMD：' + '；'.join(f'{k} {primary[k]-baseline[k]:+.6f}' for k in keys) + '\n'
    changes += '  相对上一版MH：' + '；'.join(f'{k} {primary[k]-old[k]:+.6f}' for k in keys) + '\n'
    changes += '''  主实验相对DMD四项提高；与上一版MH相比，HP有所下降，其他三项提高。
  lambda=.50的CLIP更高，但HP低于DMD；没有把该设置事后替换为主实验。

代码相对上一版MH的修改
  1. 新增后验去噪一致性半梯度正则；不改采样核，不增加网络前向。
     g_lambda=G[(1-lambda)*m_hat+lambda*D-r]。
  2. 新建独立训练入口和配置，预先指定主实验.25、敏感性.50。
     学习率、seed、batch、模型、分辨率、EMA、CPS、教师CFG等继承原配置。
  3. 新增一致性/教师误差日志、四指标评分标识、五方法汇总审计与参数差异报告。
  4. 新增6项测试；与原14项合计20项通过。运行方法见RUN_RTDMD_PCMH_V30_CN.txt。
  5. 为磁盘容量限制，将新全量PNG链接到内存盘；其余必要证据和EMA LoRA保存在项目中。

论文相对提供的v29的修改
  1. 保留Tweedie条件均值目标、教师引导、当前Student目标和不训练Fake的核心方向。
  2. 明确原反向路径两个数学缺口：零方差末步核可能奇异；反向端点律不自动等于前向后验。
     不添加终端噪声。在真实生成随机变量空间定义后验，避免奇异末步密度比。
  3. 补写成对轨迹的条件平稳初始化、局部pCN-MH接受比中心校正及Rao-Blackwell均值。
     这些是此前Posterior-MH修订，本轮整合进论文；不是原全路径IS的严格复现。
  4. 明确同一观测点Jacobian与原DMD源Jacobian的区别；停止梯度、归一化、时间权重及教师近似。
  5. 新增Student后验一致性正则及两条推导：条件协方差系数(1-lambda)^2、期望更新新增项。
     说明它不是原Fake一致性正则，也不是不改变原KL梯度的无代价降方差。
  6. 写入全部五方法的真实四指标表、主实验相对DMD/旧MH差值、敏感性权衡与实际移动率。
     单种子短训练、低噪声混合、无0步MH消融等限制一并说明；没有全局收敛/稳定优越性声明。
  7. 重组为31个内容/附录frame及标题页，共32页；保留中文Beamer风格。
     清理原稿重复的end-document，改用可移植Fandol字体；原v29备份不变。
     PDF编译、页面边界/文本和数值表检查记录见pdf-validation.json。

交付位置
  /rootDMD/papers/RTDMD_v30/：新版tex、PDF、源稿备份、构建脚本、来源哈希、校验与逐行diff。
  /rootDMD/ReFormFlow-main/logs/rtdmd-pcmh/comparison-v30/：完整对照表和审计。
  /rootDMD/ReFormFlow-main/RUN_RTDMD_PCMH_V30_CN.txt：代码入口、运行和结果查看方法。
  /rootDMD/deliverables/RTDMD_v30_Paper_20261021.zip：论文包。
  /rootDMD/deliverables/RTDMD_v30_Code_PickScore_20261021.zip：代码与提示词包，不含基础模型和评分模型。
  /rootDMD/deliverables/RTDMD_v30_Results_20261021.zip：配置、训练记录、逐图四指标、提示词、审计和日志。
  EMA LoRA原件留在两个run的checkpoints/checkpoint-1000/lora/generator/，不在小型结果包中重复存储。
  /dev/shm/rtdmd-v30-images/RTDMD_v30_FullPNG_20261021.zip：两组全部4096张PNG，约2GB。
  完整图片和该大包在内存盘，会随宿主机重启丢失；如需长期保留请先下载。
  小型论文、代码、结果包以及原始分数、提示词和权重位于项目磁盘，不依赖这些PNG继续存在。

编译与科学结论
  新tex自包含，使用XeLaTeX和TeX发行版Fandol字体，无外部图片依赖。
  build_paper.py按当前服务器已保存的评分JSON生成表格；单独编译tex不需要运行Python。
  本轮四项高于DMD是一次实测结果，不能单凭此验证原方案全部理论，亦不能推断16卡收益。
'''
    (PAPER / 'CHANGES_v30_CN.txt').write_text(changes)
    diff = difflib.unified_diff((PAPER / 'v29-original-reference.tex').read_text().splitlines(True),
        tex.read_text().splitlines(True), fromfile='v29-original-reference.tex', tofile=tex.name)
    (PAPER / 'v29-to-v30.diff').write_text(''.join(diff))
    archives = []
    path = OUT / 'RTDMD_v30_Paper_20261021.zip'
    with ZipFile(path, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        for name in [tex.name, pdf.name, 'v29-original-reference.tex', 'build_paper.py',
                     'package_delivery.py', 'validate_pdf.py', 'paper-data-manifest.json', 'pdf-validation.json',
                     'CHANGES_v30_CN.txt', 'v29-to-v30.diff', 'compile.log']:
            z.write(PAPER / name, 'RTDMD_v30/' + name)
        for name in ['comparison.csv', 'summary.json', 'metrics-comparison.png', 'metrics-comparison.pdf']:
            z.write(COMP / name, 'RTDMD_v30/experiment-evidence/' + name)
    archives.append(path)
    path = OUT / 'RTDMD_v30_Code_PickScore_20261021.zip'
    with ZipFile(path, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        candidates = []
        for folder in ['flow_grpo', 'config', 'scripts', 'jobs', 'tests', 'docs', 'dataset/pickscore']:
            candidates.extend((ROOT / folder).rglob('*'))
        candidates.extend(p for p in ROOT.iterdir() if p.is_file())
        for p in sorted(set(candidates)):
            if p.is_file() and '__pycache__' not in p.parts and (
                p.suffix in {'.py', '.sh', '.txt', '.toml', '.md', '.yaml', '.yml'} or p.name == '.gitignore'):
                z.write(p, 'ReFormFlow-main/' + str(p.relative_to(ROOT)))
        z.write(COMP / 'source-manifest.json', 'ReFormFlow-main/logs/rtdmd-pcmh/comparison-v30/source-manifest.json')
    archives.append(path)
    path = OUT / 'RTDMD_v30_Results_20261021.zip'
    with ZipFile(path, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(COMP.iterdir()):
            if p.is_file(): z.write(p, str(p.relative_to(WORK)))
        for row, relative in [
            ('DMD', 'logs/dmd/sd3-medium-pickscore-no-reward-train'),
            ('SourceIS', 'logs/rtdmd-source-is/sd3-medium-rtdmd-source-is-train'),
            ('PosteriorMH', 'logs/rtdmd-posterior-mh/sd3-medium-rtdmd-posterior-mh-train'),
            ('PCMH025', 'logs/rtdmd-pcmh/sd3-medium-rtdmd-pcmh-lambda025-train'),
            ('PCMH050', 'logs/rtdmd-pcmh/sd3-medium-rtdmd-pcmh-lambda050-train'),
        ]:
            run = ROOT / relative
            for name in ['config.json', 'metrics.jsonl', 'status.txt']:
                if (run / name).is_file(): z.write(run / name, f'runs/{row}/{name}')
            for name in ['final-metrics.json', 'final-metrics.csv', 'final-metrics.md',
                         'per-image-scores.json', 'prompts-persistent.json']:
                p = run / 'eval/step-1000' / name
                if p.is_file(): z.write(p, f'runs/{row}/eval/step-1000/{name}')
        for label in ['025', '050']:
            z.write(WORK / f'rtdmd-pcmh-lambda{label}.log', f'logs/rtdmd-pcmh-lambda{label}.log')
        for p in [ROOT / 'RUN_RTDMD_PCMH_V30_CN.txt', PAPER / 'CHANGES_v30_CN.txt']:
            z.write(p, p.name)
    archives.append(path)
    # Archive integrity and final hashes, without copying the large base models or PNGs.
    for archive in archives:
        with ZipFile(archive) as z:
            assert z.testzip() is None
    final = dict(tex=dict(path=str(tex), sha256=sha(tex)), pdf=dict(path=str(pdf), sha256=sha(pdf)),
        archives=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in archives],
        full_png_archive=dict(path='/dev/shm/rtdmd-v30-images/RTDMD_v30_FullPNG_20261021.zip',
            volatile_on_host_restart=True, images=4096),
        audit_summary=str(COMP / 'summary.json'),
        weights=[v['checkpoint'] for v in s['new_run_audits'].values()])
    (OUT / 'RTDMD_v30_Delivery_Manifest_20261021.json').write_text(json.dumps(final, indent=2, ensure_ascii=False))
    for p in archives: print(p, p.stat().st_size)


if __name__ == '__main__':
    main()
