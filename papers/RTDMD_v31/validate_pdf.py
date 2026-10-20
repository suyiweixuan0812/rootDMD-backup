"""Check actual PDF bytes, table numbers, page bounds and rendered pages."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,'/dev/shm/rtdmd-v30-tools/python')
import pymupdf
from PIL import Image,ImageDraw

PAPER=Path(__file__).resolve().parent
PREVIEW=Path('/dev/shm/rtdmd-v31-pdf-preview')


def main():
    tex=PAPER/'RTDMD_SchemeA_Reorganized_Final_CN_v31.tex'
    pdf=tex.with_suffix('.pdf')
    source=tex.read_text()
    assert pdf.read_bytes().startswith(b'%PDF-')
    doc=pymupdf.open(pdf)
    assert len(doc)==source.count('\\begin{frame}')+1
    assert source.count('\\end{document}')==1
    manifest=json.loads((PAPER/'paper-data-manifest.json').read_text())
    forbidden=['运行中；','训练未完成','完整测试仍运行','FINAL_NOT_SELECTED']
    all_text='\n'.join(p.get_text() for p in doc)
    assert not any(t in source or t in all_text for t in forbidden)
    values=[]
    for group in manifest['validation_table_groups']:
        normalize=lambda x:''.join(x.split())
        matching=[p for p in doc if normalize(group['title_contains']) in normalize(p.get_text().splitlines()[0])]
        assert matching, group['title_contains']
        text='\n'.join(p.get_text() for p in matching)
        for name in group['names']:
            path=Path(manifest['reports'][name]);raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==manifest['report_sha256'][name]
            report=json.loads(raw)
            for key in ['CLIP Score','Aesthetic Score','Pick Score','HP Score']:
                v=f'{report["table"][key]:.4f}'
                assert v in text,(name,key,v)
                values.append({'run':name,'metric':key,'rendered_value':v})
    PREVIEW.mkdir(exist_ok=True)
    outside=[];replacement=[];thumbs=[]
    for i,page in enumerate(doc):
        if '\ufffd' in page.get_text():replacement.append(i+1)
        for b in page.get_text('blocks'):
            if b[0]<-1 or b[1]<-1 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1:
                outside.append({'page':i+1,'bbox':list(b[:4]),'text':b[4]})
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.7,1.7),alpha=False)
        pix.save(PREVIEW/f'page-{i+1:02d}.png')
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        im.thumbnail((480,270));thumb=Image.new('RGB',(490,295),'white');thumb.paste(im,(5,20))
        ImageDraw.Draw(thumb).text((8,3),str(i+1),fill='black');thumbs.append(thumb)
    for start in range(0,len(thumbs),16):
        sheet=Image.new('RGB',(1960,1180),'#ccc')
        for j,im in enumerate(thumbs[start:start+16]):sheet.paste(im,((j%4)*490,(j//4)*295))
        sheet.save(PREVIEW/f'montage-{start//16+1}.png')
    assert not outside,outside
    assert not replacement,replacement
    result={'page_count':len(doc),'all_pages_rendered':True,'numeric_values_verified':values,
            'source_complete':True,'outside_page_text':outside,'replacement_character_pages':replacement,
            'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'tex_sha256':hashlib.sha256(tex.read_bytes()).hexdigest(),
            'preview_directory':str(PREVIEW),'visual_review':'Pending review of rendered pages'}
    (PAPER/'pdf-validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('Valid PDF:',len(doc),'pages;',len(values),'actual table values verified; all pages rendered')


if __name__=='__main__':main()
