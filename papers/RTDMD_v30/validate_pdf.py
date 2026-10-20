"""Verify the compiled document and render every page for layout inspection."""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, '/dev/shm/rtdmd-v30-tools/python')
import pymupdf
from PIL import Image, ImageDraw

PAPER = Path(__file__).resolve().parent
PREVIEW = Path('/dev/shm/rtdmd-v30-pdf-preview')


def main():
    tex = PAPER / 'RTDMD_SchemeA_Reorganized_Final_CN_v30.tex'
    pdf = tex.with_suffix('.pdf')
    source = tex.read_text()
    doc = pymupdf.open(pdf)
    assert len(doc) == source.count('\\begin{frame}') + 1 == 32
    assert source.count('\\end{document}') == 1
    assert not any(token in source for token in ['@@RESULT', '运行中：', '训练未完成', '正在运行'])
    all_text = '\n'.join(page.get_text() for page in doc)
    assert '运行中' not in all_text
    reports = json.loads((PAPER / 'paper-data-manifest.json').read_text())['reports']
    values = []
    for path in reports.values():
        report = json.loads(Path(path).read_text())
        for key in ['CLIP Score', 'Aesthetic Score', 'Pick Score', 'HP Score']:
            formatted = f'{report["table"][key]:.4f}'
            assert formatted in all_text, f'PDF table value missing: {formatted}'
            values.append(formatted)
    overflows, bad_chars = [], []
    PREVIEW.mkdir(exist_ok=True)
    thumbnails = []
    for i, page in enumerate(doc):
        for block in page.get_text('blocks'):
            x0, y0, x1, y1 = block[:4]
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                overflows.append(dict(page=i+1, bbox=list(block[:4]), text=block[4]))
        if '\ufffd' in page.get_text(): bad_chars.append(i+1)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False)
        pix.save(PREVIEW / f'page-{i+1:02d}.png')
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        img.thumbnail((480, 270))
        thumb = Image.new('RGB', (490, 296), 'white')
        thumb.paste(img, (5, 20))
        ImageDraw.Draw(thumb).text((8, 3), str(i+1), fill='black')
        thumbnails.append(thumb)
    for batch_start in (0, 16):
        sheet = Image.new('RGB', (4*490, 4*296), '#cccccc')
        for i, thumb in enumerate(thumbnails[batch_start:batch_start+16]):
            sheet.paste(thumb, ((i%4)*490, (i//4)*296))
        sheet.save(PREVIEW / f'montage-{batch_start//16+1}.png')
    result = dict(page_count=len(doc), numeric_table_verified=True, table_values=values,
        source_complete=True, text_blocks_outside_page=overflows, replacement_character_pages=bad_chars,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        tex_sha256=hashlib.sha256(tex.read_bytes()).hexdigest(),
        preview_directory=str(PREVIEW), all_pages_rendered=True,
        compiler='Tectonic 0.17.0 (XeTeX); fonts bundled through the TeX resource bundle',
        visual_review='Pending review of the rendered pages')
    (PAPER / 'pdf-validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
