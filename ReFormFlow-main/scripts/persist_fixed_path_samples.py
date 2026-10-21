"""Persist original scored PNGs at the eight fixed, predeclared indices."""
from pathlib import Path
import argparse,json,hashlib,shutil
from PIL import Image
from render_fixed_path_comparison import INDICES
p=argparse.ArgumentParser();p.add_argument('--evaluation-run',required=True);p.add_argument('--minimum-free-mib',type=int,default=256);a=p.parse_args()
R=Path(__file__).resolve().parents[1]/'logs/path-ratio-round1';folder=R/'runs'/a.evaluation_run/'eval/step-1'
report=json.loads((folder/'final-metrics.json').read_text());records=json.loads((folder/'per-image-scores.json').read_text());assert report['num_images']==len(records)==2048
selected=[records[i] for i in INDICES];size=sum((folder/r['image']).stat().st_size for r in selected)
out=R/'archived-samples'/a.evaluation_run
if (out/'sample-storage-manifest.json').exists():print('Fixed raw samples already saved',out);raise SystemExit(0)
assert shutil.disk_usage(R).free>size+a.minimum_free_mib*1024**2,'Preserve requested free-space margin'
out.mkdir(parents=True,exist_ok=True);saved=[]
for index,r in zip(INDICES,selected):
 src=folder/r['image'];dst=out/f'{index:06d}.png'
 with Image.open(src) as im:assert im.size==(512,512);im.verify()
 shutil.copy2(src,dst)
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 assert sha(src)==sha(dst)
 saved.append(dict(index=index,prompt=r['prompt'],scores=r['scores'],file=dst.name,sha256=sha(dst)))
(out/'sample-storage-manifest.json').write_text(json.dumps(dict(evaluation_run=a.evaluation_run,original_test_count=2048,stored_count=8,indices=INDICES,selection='Fixed equally spaced indices, no selection by score',original_PNG_bytes_unchanged=True,checkpoint=report['protocol']['checkpoint_evaluation_only'],records=saved),indent=2,ensure_ascii=False))
print('Persisted8original PNGs',out,size,'bytes')
