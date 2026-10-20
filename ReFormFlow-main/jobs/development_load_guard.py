"""User-requested, bounded development-host keepalive; NOT research evidence.

CPU BLAS load runs at low scheduling priority. GPU matrix load yields whenever
another CUDA process is present. Delete the enable file or send SIGTERM to stop.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time


def cpu_worker(index, stop):
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.nice(15)
    import numpy as np
    rng=np.random.default_rng(91800+index)
    a=rng.standard_normal((256,256),dtype=np.float32)
    b=rng.standard_normal((256,256),dtype=np.float32)
    out=np.empty_like(a)
    while not stop.is_set():
        for _ in range(24): np.matmul(a,b,out=out)
        if not np.isfinite(out[0,0]): raise FloatingPointError('Nonfinite keepalive matrix output')


def cpu_snapshot():
    with open('/proc/stat') as f:
        v=[int(x) for x in f.readline().split()[1:]]
    return sum(v[:8]),v[3]+v[4]


def process_cpu_seconds(pids):
    total=0.
    for pid in pids:
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            total+=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
        except (FileNotFoundError,IndexError):pass
    return total


def gpu_pids():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True)
    return {int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--cpu-fraction',type=float,default=.64)
    parser.add_argument('--directory',type=Path,default=Path('/rootDMD/development-load-guard'))
    args=parser.parse_args()
    if not 0<args.cpu_fraction<=.8 or args.hours<=0: parser.error('Invalid bounded load request')
    args.directory.mkdir(exist_ok=True)
    enable=args.directory/'enabled'
    if not enable.exists(): raise RuntimeError('Create the explicit enable file first')
    pid=args.directory/'guard.pid'
    if pid.exists():
        try: os.kill(int(pid.read_text()),0)
        except (ProcessLookupError,ValueError): pass
        else: raise RuntimeError('A guard is already active')
    pid.write_text(str(os.getpid()))
    os.nice(10)
    context=mp.get_context('spawn')
    stop=context.Event()
    # Do not acquire Event locks in a signal handler interrupted inside wait().
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    workers=[]
    try:
        allocated_cpus=os.cpu_count() or len(os.sched_getaffinity(0))
        count=max(1,math.ceil(allocated_cpus*args.cpu_fraction))
        for index in range(count):
            worker=context.Process(target=cpu_worker,args=(index,stop))
            worker.start()
            workers.append(worker)
        import torch
        torch.set_num_threads(1)
        # nvidia-smi reports HOST PIDs in this container, unlike os.getpid().
        before_cuda=gpu_pids()
        torch.cuda.init()
        probe=torch.zeros(1,device='cuda')
        torch.cuda.synchronize()
        added=gpu_pids()-before_cuda
        del probe
        if len(added)!=1:
            raise RuntimeError(f'Cannot safely identify own host CUDA PID: {added}')
        own_gpu_pids=added|{os.getpid()}
        deadline=time.monotonic()+args.hours*3600
        previous=cpu_snapshot()
        previous_process=process_cpu_seconds([os.getpid()]+[w.pid for w in workers])
        previous_time=time.monotonic()
        last_log=0.
        matrices=None
        external=[]
        while not stopping[0] and enable.exists() and time.monotonic()<deadline:
            external=sorted(gpu_pids()-own_gpu_pids)
            if external:
                if matrices is not None:
                    matrices=None
                    torch.cuda.empty_cache()
                stop.wait(3)
            else:
                if matrices is None:
                    a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16)
                    b=torch.randn_like(a)
                    matrices=(a,b,torch.empty_like(a))
                    del a,b
                started=time.monotonic()
                while time.monotonic()-started<.8 and not stopping[0]:
                    torch.mm(matrices[0],matrices[1],out=matrices[2])
                    torch.cuda.synchronize()
                stop.wait(.2)
            if time.monotonic()-last_log>=30:
                current=cpu_snapshot()
                usage=100*(1-(current[1]-previous[1])/max(1,current[0]-previous[0]))
                previous=current
                process_now=process_cpu_seconds([os.getpid()]+[w.pid for w in workers])
                now=time.monotonic()
                allocated_usage=100*(process_now-previous_process)/max(.01,now-previous_time)/allocated_cpus
                previous_process,previous_time=process_now,now
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
                row=dict(timestamp=time.time(),purpose='User-requested keepalive, not experiment progress',
                    cpu_workers=count,allocated_cpus=allocated_cpus,cpu_host_system_percent=usage,
                    cpu_allocated_percent=allocated_usage,gpu_snapshot=gpu,own_gpu_host_pids=sorted(own_gpu_pids),
                    yielding_gpu_to=external,remaining_seconds=max(0,deadline-time.monotonic()))
                with (args.directory/'utilization.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                last_log=time.monotonic()
    finally:
        stop.set()
        for worker in workers: worker.join(timeout=2)
        for worker in workers:
            if worker.is_alive():worker.terminate()
        pid.unlink(missing_ok=True)


if __name__=='__main__':
    main()
