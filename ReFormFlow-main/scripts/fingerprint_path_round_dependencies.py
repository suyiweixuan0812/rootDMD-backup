"""Record versions and local model bytes, never environment variables/tokens."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

root = Path(__file__).resolve().parents[1]
r = root/'logs/path-ratio-round1'
folders = [root/'models/sd3-medium', Path('/dev/shm/dmd-eval-weights/reward_ckpts')]
files = []
for folder in folders:
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix in {'.json', '.safetensors', '.pt', '.pth', '.bin', '.txt', '.model'}:
            files.append(p)
for key in ['models--openai--clip-vit-large-patch14', 'models--yuvalkirstain--PickScore_v1']:
    snapshots = Path('/rootDMD/hf-cache/hub')/key/'snapshots'
    if snapshots.exists():
        files.extend(p for p in sorted(snapshots.rglob('*')) if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin'})
records = []
for p in files:
    with p.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    records.append(dict(path=str(p), resolved=str(p.resolve()), bytes=p.stat().st_size, sha256=sha,
                        persistent_bytes=not str(p.resolve()).startswith('/dev/shm/')))
versions = sorted({(d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name')})
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
              automatic_backup_revision=revision, files=records, package_versions=dict(versions),
              scope='Version numbers and scientific model bytes only; no environment variable values or credentials. Public/RAM caches must be restored after reboot; trained archived LoRA and reports persist.')
(r/'dependency-fingerprints.json').write_text(json.dumps(result, indent=2))
print(len(records), 'scientific dependency files hashed;', len(versions), 'package versions recorded')
