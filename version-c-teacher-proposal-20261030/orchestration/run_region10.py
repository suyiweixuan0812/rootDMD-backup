"""Single authorized eight-GPU allocation: preflight, new DMD, matched-time C."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, subprocess, hashlib, fcntl, traceback

ROOT=Path(__file__).resolve().parent
BASE=Path('/project_c20250506/250404014/r16-region10-20261026')
PY=BASE/'launch/python-train'
now=lambda:datetime.now(timezone.utc).isoformat()
def write(name,data):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2));t.replace(p)

def main():
    from allocation_gate import wait_for_allocation
    if not wait_for_allocation(ROOT,10):return
    spec=json.loads((ROOT/'formal-spec.json').read_text());src=Path(spec['source'])
    for name,h in spec['source_files'].items():
        assert hashlib.sha256((src/name).read_bytes()).hexdigest()==h,name
    dev=ROOT/'development'/spec['development_revision']
    assert json.loads((dev/'completed.json').read_text())['status']=='passed'
    assert json.loads((dev/'verification.json').read_text())['status']=='passed'
    for k in ('WORLD_SIZE','NNODES','MLP_WORKER_NUM'):
        if k in os.environ:assert int(os.environ[k])==1,(k,os.environ[k])
    for k in ('RANK','LOCAL_RANK','NODE_RANK','MLP_ROLE_INDEX'):
        if k in os.environ:assert int(os.environ[k])==0,(k,os.environ[k])
    Path('/personal_fast_storage').mkdir(exist_ok=True)
    for name in ('v29_assets','ReFormDMD-v29','v29_env','v29_r16_standard_20261025'):
        dest=Path('/personal_fast_storage')/name;target=BASE/'payload/personal_fast_storage'/name
        assert target.is_dir()
        if dest.exists() or dest.is_symlink():assert dest.resolve()==target.resolve(),str(dest)
        else:dest.symlink_to(target,target_is_directory=True)
    env=dict(os.environ,PYTHONPATH=str(src),HF_HOME='/personal_fast_storage/v29_assets/huggingface',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        REWARD_CKPT_PATH='/personal_fast_storage/ReFormDMD-v29/reward_ckpts',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    with (ROOT/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'started.json').exists() and not (ROOT/'runs').exists(),'Refuse duplicate training'
        subprocess.run([str(PY),'-c','import torch; assert torch.cuda.device_count()==8; assert torch.cuda.is_bf16_supported(); print([torch.cuda.get_device_name(i) for i in range(8)])'],env=env,check=True)
        write('started.json',dict(utc=now(),region=10,gpus=8,revision=spec['revision'],github_commit=spec['github_commit']))
        phase='preflight';arm=None
        try:
            for phase,arm in [('preflight','ddp_dmd'),('preflight','ddp_teacher_c'),('training','dmd'),('training','teacher_c')]:
                write('active-arm.json',dict(phase=phase,arm=arm,utc=now()))
                output=ROOT/('preflight-runs' if phase=='preflight' else 'runs')
                with (ROOT/(arm+'.log')).open('x') as log:
                    subprocess.run([str(PY),'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
                        'scripts/train_teacher_comparison.py','--config=config/teacher_comparison.py:'+arm],
                        cwd=src,env={**env,'EXPERIMENT_OUTPUT':str(output)},stdout=log,stderr=subprocess.STDOUT,check=True)
                complete=json.loads((output/arm/'training-complete.json').read_text())
                if phase=='preflight':assert complete['outer_step']==20 and complete['generator_step']==2
                elif arm=='dmd':assert complete['outer_step']==10000 and complete['generator_step']==1000
                else:
                    target=json.loads((ROOT/'runs/dmd/training-complete.json').read_text())['training_seconds']
                    assert complete['previous_update_boundary_seconds']<target<=complete['training_seconds']
            write('training-completed.json',dict(utc=now(),status='completed'))
            phase='verification_and_plotting'
            subprocess.run([str(PY),str(ROOT/'verify_and_plot.py'),'--root',str(ROOT)],cwd=src,env={**env,'CUDA_VISIBLE_DEVICES':''},check=True)
            write('completed.json',dict(utc=now(),status='completed'))
        except BaseException as e:
            write('failed.json',dict(utc=now(),phase=phase,arm=arm,error=repr(e),traceback=traceback.format_exc(),automatic_restart=False))
            raise

if __name__=='__main__':main()
