"""CPU-only startup gate: only one of the user-authorized region candidates trains."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,os,socket
TOKEN='teacher-c-dmd-region-race-20261030-v1'
def wait_for_allocation(root,region):
    root=Path(root)
    if (root/'started.json').exists():raise RuntimeError('Refuse duplicate started experiment')
    ready={'token':TOKEN,'region':region,'utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'pid':os.getpid(),'gpu_compute_started':False}
    with (root/'allocation-ready.json').open('x') as f:json.dump(ready,f,indent=2)
    deadline=time.monotonic()+1800
    print('ALLOCATION_READY '+json.dumps(ready),flush=True)
    while time.monotonic()<deadline:
        path=root/'allocation-decision.json'
        if path.exists():
            decision=json.loads(path.read_text())
            assert decision['token']==TOKEN and int(decision['region'])==region
            assert decision['action'] in ('run','cancel')
            print('ALLOCATION_DECISION '+json.dumps(decision),flush=True)
            return decision['action']=='run'
        time.sleep(3)
    (root/'allocation-timeout.json').write_text(json.dumps({'token':TOKEN,'region':region,'utc':datetime.now(timezone.utc).isoformat(),'gpu_compute_started':False},indent=2))
    raise RuntimeError('Allocation gate timed out; no GPU computation was started')
