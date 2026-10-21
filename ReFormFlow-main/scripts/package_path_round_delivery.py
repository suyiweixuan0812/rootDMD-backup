"""Package reviewable source, completed evidence and paper; exclude model bytes."""
from datetime import datetime,timezone
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent;R=ROOT/'logs/path-ratio-round1';P=OUT/'papers/RTDMD_v31'
files={}
def add(p,label):
 if p.is_file():files[label]=p
for folder in ['flow_grpo','config','scripts','tests']:
 for p in sorted((ROOT/folder).rglob('*.py')):
  if '__pycache__' not in p.parts:add(p,'code/ReFormFlow-main/'+str(p.relative_to(ROOT)))
for p in sorted((ROOT/'jobs').iterdir()):
 if p.suffix in {'.py','.sh'} and any(w in p.name for w in ['path','deterministic','query','load_guard','local_env','restore_public']):
  add(p,'code/ReFormFlow-main/jobs/'+p.name)
for p in sorted(ROOT.glob('requirements*.txt')):add(p,'code/ReFormFlow-main/'+p.name)
for name in ['setup.py','README.md']:add(ROOT/name,'code/ReFormFlow-main/'+name)
for p in sorted((ROOT/'docs').glob('*PathRatio*')):add(p,'code/ReFormFlow-main/docs/'+p.name)
# Raw numeric evidence is retained; checkpoints/PNGs are separate persistent artifacts.
for p in sorted((R/'evidence').rglob('*')):
 if p.suffix in {'.json','.jsonl','.csv','.txt'}:add(p,'results/evidence/'+str(p.relative_to(R/'evidence')))
for p in sorted((R/'source-snapshots').glob('*.py')):add(p,'results/source-snapshots/'+p.name)
for p in sorted((R/'failed-attempts').rglob('*')):
 if p.suffix in {'.json','.jsonl','.txt','.py','.log'}:add(p,'results/failed-attempts/'+str(p.relative_to(R/'failed-attempts')))
for p in sorted(R.glob('*')):
 if p.suffix in {'.json','.csv','.txt','.md'}:add(p,'results/'+p.name)
for folder in ['common-training-audits','comparison-long','comparison-final','comparison-seed43','comparison-two-seeds','training-drift-long']:
 for p in sorted((R/folder).glob('*')):
  if p.suffix in {'.json','.csv','.md','.txt','.pdf','.png'}:add(p,'results/'+str(p.relative_to(R)))
for p in sorted(R.glob('comparison-deterministic-*/*')):
 if p.suffix in {'.json','.csv','.md'}:add(p,'results/'+str(p.relative_to(R)))
for p in sorted(R.glob('comparison-optional-*/*')):
 if p.suffix in {'.json','.csv','.md'}:add(p,'results/'+str(p.relative_to(R)))
for p in sorted(R.glob('comparison-faithful-*/*')):
 if p.suffix in {'.json','.csv','.md'}:add(p,'results/'+str(p.relative_to(R)))
for p in sorted((R/'archived-samples').rglob('*')):
 if p.suffix in {'.json','.png'}:add(p,'results/'+str(p.relative_to(R)))
for p in sorted(R.glob('visual-*-fixed/*')):
 if p.suffix in {'.json','.png'}:add(p,'results/'+str(p.relative_to(R)))
for p in sorted(R.glob('archived-checkpoints/*/*manifest*.json')):add(p,'results/'+str(p.relative_to(R)))
for p in sorted(R.glob('archived-images/*/*manifest*.json')):add(p,'results/'+str(p.relative_to(R)))
for name in ['RTDMD_v31.tex','RTDMD_v31.pdf','RTDMD_v31_results.csv','RTDMD_v31_修改与结果说明.txt','RTDMD_v31_代码与运行说明.txt','RTDMD_v31_纯Teacher入口.txt','RTDMD_v31_资源与预算.txt']:add(OUT/name,name)
for p in sorted(P.iterdir()):
 if p.suffix in {'.py','.json','.tex'}:add(p,'paper-build/'+p.name)
add(OUT/'path-ratio-all-tests-v31-repaired.log','results/68-tests-output.txt')
snapshots={label:p.read_bytes() for label,p in files.items()}
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Source patch and completed numeric results; no base models, trained weights, PNG collections, credentials or optimizer/RNG resume state','files':{label:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for label,data in snapshots.items()}}
assert all(p.stat().st_size<32*1024**2 for p in files.values()),'Unexpected large artifact'
zipfile_path=OUT/'RTDMD_v31_代码论文结果.zip';pending=zipfile_path.with_suffix('.zip.pending')
with zipfile.ZipFile(pending,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
 for label,data in snapshots.items():z.writestr(label,data)
with zipfile.ZipFile(pending) as z:
 assert z.testzip() is None
 for label,data in snapshots.items():assert hashlib.sha256(z.read(label)).hexdigest()==manifest['files'][label]['sha256']
pending.replace(zipfile_path)
(OUT/'RTDMD_v31_交付清单.json').write_text(json.dumps({'zip_path':str(zipfile_path),'zip_sha256':hashlib.sha256(zipfile_path.read_bytes()).hexdigest(),'zip_bytes':zipfile_path.stat().st_size,**manifest},indent=2,ensure_ascii=False))
print('Packaged',len(files),'files;',zipfile_path.stat().st_size,'bytes;',zipfile_path)
