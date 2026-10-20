"""Restore a regenerable public model cache after reboot, without altering weights."""
import hashlib
import json
import os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
m=json.loads((root/'logs/path-ratio-round1/public-cache-relocation.json').read_text())
dest=Path(m['ram_cache_file'])
if not dest.exists():
    os.environ.pop('HF_HUB_OFFLINE',None)
    os.environ.pop('TRANSFORMERS_OFFLINE',None)
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=m['repo_id'],filename=m['filename'],revision=m['revision'],
        local_dir=m['restore_local_dir'],cache_dir='/dev/shm/rtdmd-public-hf-cache/download-cache')
with dest.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['sha256']
original=Path(m['original_cache_file'])
if not original.is_symlink():
    assert not original.exists(), 'Never overwrite a different restored model'
    original.symlink_to(dest)
assert original.resolve()==dest
print('Public PickScore cache verified:',original)
