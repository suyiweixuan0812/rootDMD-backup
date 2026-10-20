"""Verify scientific dependency bytes against the original frozen manifest."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'logs/path-ratio-round1'
original=R/'dependency-fingerprints.json';record=json.loads(original.read_text())
checks=[]
for item in record['files']:
 p=Path(item['path'])
 if not p.exists():checks.append(dict(path=str(p),exists=False,sha256_equal=False));continue
 with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
 checks.append(dict(path=str(p),exists=True,bytes=p.stat().st_size,sha256_equal=sha==item['sha256']))
versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
differences={k:dict(frozen=v,current=versions.get(k)) for k,v in record['package_versions'].items() if versions.get(k)!=v}
result=dict(verified_utc=datetime.now(timezone.utc).isoformat(),original_manifest_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),files=checks,package_version_differences=differences,all_frozen_scientific_bytes_unchanged=all(c['sha256_equal'] for c in checks),scope='Scientific files and package versions; never reads tokens or other environment values')
(R/'dependency-fingerprints-final-verification.json').write_text(json.dumps(result,indent=2))
assert result['all_frozen_scientific_bytes_unchanged'],result
assert not differences,differences
print('Verified',len(checks),'unchanged scientific files;',len(record['package_versions']),'unchanged package versions')
