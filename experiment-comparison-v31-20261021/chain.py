"""Idempotent last -> Bridge supervisor for the user-authorized experiments."""
import configparser,datetime,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import boto3
from botocore.config import Config
os.umask(0o077)
ROOT=Path('/rootDMD/experiment-comparison-v31-20261021')
sys.path.insert(0,str(ROOT))
from dmd_cloud_api import call
ENV={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
Path('/dev/shm/rtdmd-submit-tmp').mkdir(exist_ok=True)
ENV['TMPDIR']='/dev/shm/rtdmd-submit-tmp'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(plan):
    plan['checked_at']=now();p=ROOT/'plan.tmp';p.write_text(json.dumps(plan,indent=2));p.replace(ROOT/'plan.json')
def log(message):print(now(),message,flush=True)
def redact(s):return re.sub(r'https?://[^\s"\x27]*\?[^\s"\x27]*','[SIGNED_URL_REDACTED]',s)
def capture(job,label,limit=1500):
    p=subprocess.run(['/root/.volc/bin/mlp','job','logs','--id',job,'--instance','worker-0','--limit',str(limit)],env=ENV,capture_output=True,text=True,timeout=55)
    if p.returncode:
        log(label+' log capture retry needed: '+redact(p.stderr[-200:]));return
    target=ROOT/(label+'-worker-0.log')
    previous=set(target.read_text().splitlines()) if target.is_file() else set()
    lines=redact(p.stdout).splitlines()
    fresh=[s for s in reversed(lines) if s not in previous]
    with target.open('a') as f:
        for line in fresh:f.write(line+'\n')
def status(job):
    result=call('GetJob',{'Id':job})
    if 'Result' not in result or 'Status' not in result['Result']:raise RuntimeError('GetJob missing Status')
    return result['Result']['Status']
def submit(plan):
    smoke=Path('/dev/shm/rtdmd-v31-bridge-20261021/timing-smoke.log')
    if not smoke.is_file():smoke=ROOT/'timing-smoke.log'
    if not smoke.is_file() or 'TIMING_AND_BENCHMARK_SMOKE_OK' not in smoke.read_text():
        raise RuntimeError('Timing/benchmark smoke not yet passed; no submission performed')
    if smoke!=ROOT/'timing-smoke.log':shutil.copyfile(smoke,ROOT/'timing-smoke.log')
    assert hashlib.sha256((ROOT/'source.zip').read_bytes()).hexdigest()==plan['source_zip_sha256']
    source=Path(plan['source']);source.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ROOT/'source.zip') as z:z.extractall(source)
    shutil.copyfile(ROOT/'comparison-spec.json',source/'comparison-spec.json')
    c=configparser.ConfigParser();c.read('/root/.volc/credentials')
    s3=boto3.client('s3',endpoint_url='https://tos-s3-cn-beijing.ivolces.com',region_name='cn-beijing',
        aws_access_key_id=c['default']['access_key_id'],aws_secret_access_key=c['default']['secret_access_key'],
        config=Config(signature_version='s3v4',proxies={},s3={'addressing_style':'virtual'}))
    def put(key):return s3.generate_presigned_url('put_object',Params={'Bucket':plan['bucket'],'Key':plan['prefix']+'/'+key},ExpiresIn=86400)
    artifacts=[]
    for method in ('last','bridge'):
        for step in [0,10000,*range(1000,10000,1000)]:
            key=f'{method}/step-{step}.zip'
            folder=plan['last_root']+f'/drawbench/pickscore/step-{step}' if method=='last' else plan['bridge_root']+f'/eval/step-{step}'
            artifacts.append({'key':key,'method':method,'step':step,'folder':folder,
                'metrics_file':'metrics.json' if method=='last' else 'final-metrics.json','put_url':put(key)})
    private={'artifacts':artifacts,'index_put_url':put('index.json'),'benchmark_file':plan['benchmark_file'],
        'benchmark_put_url':put('inference-benchmark.json'),'performance_put_url':put('bridge-performance.json'),'bridge_root':plan['bridge_root']}
    (source/'jobs/relay-private.json').write_text(json.dumps(private));(source/'jobs/relay-private.json').chmod(0o600)
    template=json.loads((ROOT/'job-public.json').read_text())
    template['StorageConfig']['Credential']={'AccessKey':c['default']['access_key_id'],'SecretAccessKey':c['default']['secret_access_key']}
    (ROOT/'job-private.json').write_text(json.dumps(template))
    plan['phase']='submitting_bridge';save(plan)
    result=subprocess.run(['/root/.volc/bin/mlp','job','create','--filename',str(ROOT/'job-private.json')],env=ENV,capture_output=True,text=True,timeout=240)
    raw=result.stdout+'\n'+result.stderr;(ROOT/'create-private.log').write_text(raw)
    ids=sorted(set(re.findall(r't-\d{14}-[a-z0-9]+',raw)))
    if result.returncode or len(ids)!=1:
        plan['phase']='submission_requires_inspection';save(plan)
        raise RuntimeError('Submission result ambiguous; inspect private log and platform job list before any retry')
    plan['bridge_job_id']=ids[0];plan['phase']='bridge_submitted';plan['bridge_submitted_at']=now();save(plan)
    log('BRIDGE_SUBMITTED '+ids[0])
def main():
    lock=(ROOT/'chain.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:log('Supervisor already active');return
    (ROOT/'chain.pid').write_text(str(os.getpid()))
    plan=json.loads((ROOT/'plan.json').read_text())
    if plan.get('phase') in {'submitting_bridge','submission_requires_inspection'} and not plan.get('bridge_job_id'):
        raise RuntimeError('Unresolved previous submission; will not create duplicate')
    old=Path('/tmp/dmd-cloud-state/t-20261020220728-lmk7p-worker-0-logs.txt')
    if old.is_file() and not (ROOT/'dmd-worker-0.log').is_file():shutil.copyfile(old,ROOT/'dmd-worker-0.log')
    plan['dmd_status']=status(plan['dmd_job_id']);save(plan)
    capture(plan['last_job_id'],'last',15000)
    failures=0
    while True:
        try:
            label='bridge' if plan.get('bridge_job_id') else 'last'
            job=plan.get('bridge_job_id') or plan['last_job_id']
            current=status(job);plan[label+'_status']=current;save(plan)
            capture(job,label)
            state=current['State']
            if state=='Completed':
                capture(job,label,15000)
                if label=='last':
                    submit(plan)
                else:
                    plan['phase']='completed';plan['completed_at']=now();save(plan)
                    log('COMPARISON_EXPERIMENTS_COMPLETE');return
            elif state in {'Failed','Stopped','Canceled','Cancelled','Terminated'}:
                plan['phase']=label+'_failed';save(plan);log(label.upper()+'_FAILED '+state);return
            failures=0
        except Exception as e:
            failures+=1;log('SUPERVISOR_ERROR '+type(e).__name__+': '+redact(str(e)))
            if plan.get('phase') in {'submitting_bridge','submission_requires_inspection'} or failures>=10:
                plan['supervisor_error']=type(e).__name__+': '+redact(str(e));save(plan);raise
        time.sleep(45)
if __name__=='__main__':main()
