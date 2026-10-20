"""Archive unchanged complete PNG evidence with SHA256 checks; never overwrite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from PIL import Image

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--step',type=int,default=1)
p.add_argument('--expected-count',type=int,default=2048)
p.add_argument('--name',required=True)
p.add_argument('--equivalent-png-directory',type=Path)
a=p.parse_args()
source=a.run_dir/'eval'/f'step-{a.step}'
report=json.loads((source/'final-metrics.json').read_text())
assert report['num_images']==a.expected_count
assert 'checkpoint_training_configuration' in report['protocol'], 'Finish true-training metadata audit first'
images=sorted((source/'rank-0').glob('*.png'))
assert [f.name for f in images]==[f'{i:06d}.png' for i in range(a.expected_count)]
root=Path(__file__).resolve().parents[1]/'logs/path-ratio-round1/archived-images'
root.mkdir(exist_ok=True)
dest=root/a.name
assert not dest.exists()
pending=root/(a.name+'.copying');assert not pending.exists()
needed=0 if a.equivalent_png_directory else sum(x.stat().st_size for x in images)
assert shutil.disk_usage(root).free>needed+256*1024**2, 'Preserve at least256MiB remaining disk margin'
(pending/'rank-0').mkdir(parents=True)
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
records=[]
linked=0
for f in images:
    src=(a.equivalent_png_directory/f.name) if a.equivalent_png_directory else f
    sha=digest(f)
    assert digest(src)==sha
    target=pending/'rank-0'/f.name
    try:os.link(src,target);linked+=1
    except OSError:shutil.copy2(src,target)
    assert digest(target)==sha
    with Image.open(target) as im:
        assert im.size==(report['protocol']['resolution'],)*2
        im.verify()
    records.append({'image':'rank-0/'+f.name,'sha256':sha,'bytes':target.stat().st_size})
for f in source.iterdir():
    if f.is_file():shutil.copy2(f,pending/f.name)
shutil.copy2(source/'rank-0/prompts.json',pending/'rank-0/prompts.json')
manifest={'source':str(source.resolve()),'persistent_directory':str(dest),'count':len(records),
          'all_png_bytes_unchanged':True,'all_png_verified':True,'hardlinked_existing_persistent_images':linked,
          'images':records}
(pending/'image-storage-manifest.json').write_text(json.dumps(manifest,indent=2))
pending.rename(dest)
print(dest,len(records),'verified original PNGs,',linked,'persistent hardlinks')
