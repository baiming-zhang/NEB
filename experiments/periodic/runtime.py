"""Portable run-local records and total-memory guard; no private GPU-owner files."""
import json,os,threading,time
from pathlib import Path
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def guard(cp,run):
    import psutil
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    pool=cp.get_default_memory_pool();pool.set_limit(size=12*2**30)
    stop=threading.Event()
    def measure():
        free,total=cp.cuda.runtime.memGetInfo();host=psutil.virtual_memory()
        return dict(unix=time.time(),rss=psutil.Process().memory_info().rss,host_free=host.available,host_used=host.total-host.available,gpu_free=free,gpu_used=total-free,pool_reserved=pool.total_bytes(),pool_limit_bytes=pool.get_limit())
    def valid(r):return r['rss']<62_500_000_000 and r['host_used']<62_500_000_000 and r['host_free']>9*2**30 and r['gpu_free']>500*2**20 and r['gpu_used']<15_500_000_000
    if not valid(measure()):raise RuntimeError('Insufficient resources before periodic evaluation')
    def watch():
        while not stop.wait(5):
            r=measure()
            with (run/'resources.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if not valid(r):write(run/'resource_stop.json',r);os._exit(124)
    threading.Thread(target=watch,daemon=True).start();return stop
