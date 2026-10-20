"""Observe all visible-container CPU work and GPU use during the bounded round."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def cpu_times():
    values={}
    ticks=os.sysconf('SC_CLK_TCK')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            values[int(p.name)]=(int(fields[11])+int(fields[12]))/ticks
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):pass
    return values


def main():
    folder=Path('/rootDMD/development-load-guard')
    deadline=datetime.fromisoformat('2026-10-21T01:20:00+00:00').timestamp()
    previous,start=cpu_times(),time.monotonic()
    history=[]
    while time.time()<deadline and (folder/'enabled').exists():
        time.sleep(5)
        current,now=cpu_times(),time.monotonic()
        used=sum(max(0,value-previous.get(pid,value)) for pid,value in current.items())
        cpu=100*used/max(.01,now-start)/(os.cpu_count() or 1)
        previous,start=current,now
        output=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
        gpu,memory=[int(x.strip()) for x in output.split(',')]
        history.append((cpu,gpu))
        recent=history[-12:]
        row=dict(timestamp=time.time(),allocated_cpus=os.cpu_count(),
            visible_container_cpu_percent_of_allocated=cpu,gpu_utilization_percent=gpu,
            gpu_memory_mib=memory,mean_last12_cpu=sum(v[0] for v in recent)/len(recent),
            mean_last12_gpu=sum(v[1] for v in recent)/len(recent),
            interpretation='All visible container processes; distinct from guard-only CPU. Five-second GPU snapshots.')
        with (folder/'resource-monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


if __name__=='__main__':main()
