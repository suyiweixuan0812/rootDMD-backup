"""Scientific comparison panels at fixed indices, without score-based selection."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

INDICES = [0, 256, 512, 768, 1024, 1280, 1536, 1792]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-eval', type=Path, required=True)
    p.add_argument('--candidate-eval', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    folders = [a.baseline_eval, a.candidate_eval]
    reports = [json.loads((x/'final-metrics.json').read_text()) for x in folders]
    records = [json.loads((x/'per-image-scores.json').read_text()) for x in folders]
    assert all(len(x) == 2048 for x in records)
    for key in ['test_file_sha256', 'seed', 'resolution']:
        assert reports[0]['protocol'][key] == reports[1]['protocol'][key]
    assert [x['prompt'] for x in records[0]] == [x['prompt'] for x in records[1]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(selection='Eight fixed, equally spaced test indices; no selection by metric delta',
                    indices=INDICES, sources=[], scope='Display panels only; all metrics use original PNGs')
    for group in range(4):
        indices = INDICES[group*2:group*2+2]
        fig, axes = plt.subplots(2, 2, figsize=(10, 12), layout='constrained')
        for row, index in enumerate(indices):
            for col in range(2):
                item = records[col][index]
                path = folders[col]/item['image']
                axis = axes[row, col]
                with Image.open(path) as im:
                    assert im.size == (512, 512)
                    axis.imshow(im.convert('RGB'))
                axis.axis('off')
                score = item['scores']
                title = f'{reports[col]["table"]["Post-Train"]}\nPick {score["pickscore"]*26:.3f}; HPS {score["hpsv2"]*100:.3f}'
                axis.set_title(title, fontsize=9)
                axis.text(.5, -.025, f'#{index}: '+textwrap.fill(item['prompt'], 58),
                          transform=axis.transAxes, ha='center', va='top', fontsize=8)
                manifest['sources'].append(dict(index=index, column=col, path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        fig.savefig(a.output/f'fixed-comparison-{group+1}.png', dpi=140)
        fig.savefig(a.output/f'fixed-comparison-{group+1}.pdf')
        plt.close(fig)
    (a.output/'visual-source-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
