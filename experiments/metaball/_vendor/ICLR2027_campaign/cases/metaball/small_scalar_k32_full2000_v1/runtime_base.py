
import os,sys,time,json,gc,math
from _neb_paths import Path
from contextlib import contextmanager
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen'))
from original_potential_k32_core import (cp,np,Model,NeuralPoints,network_fields,network_backward,
    potential_objective,fluid_points,config_namespace,sha,write)
from equivalent_harmonic_lift import lift_fields
from qualified_weak import Weak
import psutil
CASE=HERE.parent;D=CASE/'results/small_scalar_k32_full2000_v1'
PRIOR=CASE/'results/original_potential_k32_full2000_v1'

@contextmanager
def timed(t,name):
    cp.cuda.Stream.null.synchronize();start=time.perf_counter()
    try:yield
    finally:
        cp.cuda.Stream.null.synchronize();t[name]=t.get(name,0.)+time.perf_counter()-start

def progress(phase,**kw):
    row=dict(phase=phase,process=os.getpid(),updated_unix=time.time(),run=str(D),
        host_bytes=psutil.Process().memory_info().rss,gpu_pool_reserved=cp.get_default_memory_pool().total_bytes(),
        new_K2048_allowed=False,**kw)
    write(D/'progress.json',row);write(CASE/'status.json',row);print(json.dumps(row),flush=True)

class Pipeline:
    def __init__(self,c,geo):
        self.c=c;self.geo=cp.asarray(geo,dtype='f8');self.setup_timings={}
        with timed(self.setup_timings,'source_sampling_geometry'):
            x=np.load(CASE/'results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz')['fit']
            self.x=cp.asarray(x,dtype='f8');self.fit=NeuralPoints(x,self.geo,c,'f4')
        with timed(self.setup_timings,'weak_operator_preparation'):self.scalar=Weak(c,self.geo,self.x)
        with timed(self.setup_timings,'harmonic_forward_solve'):
            _,dg,_,_=lift_fields(self.x,self.geo,c);lf=dg.reshape(-1,1)
            self.hco=self.solve(lf,'setup_harmonic')
        with timed(self.setup_timings,'harmonic_preparation'):
            self.hf=lf-self.fit_field(self.hco);self.hm=cp.sum(self.hf*self.hf)
            assert np.isfinite(float(self.hm)) and float(self.hm)>0
            self.Gfh=self.fit_transpose(self.hf)
        setup=dict(timings=self.setup_timings,scalar=self.scalar.info,source_points=len(x),harmonic_mass=float(self.hm),solver=self.scalar.records)
        write(D/f'setup_{os.getpid()}_{time.time_ns()}.json',setup)
        write(D/'setup.json',setup)
    def fit_field(self,co):return self.scalar.X@(co[:self.scalar.n]*self.scalar.scales[:,None])+self.scalar.B@co[self.scalar.n:]
    def fit_transpose(self,a):return cp.concatenate(((self.scalar.XT@a)*self.scalar.scales[:,None],self.scalar.B.T@a))
    def solve(self,a,label,transpose=False):
        value,r=(self.scalar.transpose(a,label,180) if transpose else self.scalar.solve(a,label,180))
        if not r['converged'] or not bool(cp.all(cp.isfinite(value))):raise FloatingPointError('Unresolved weak solve: '+json.dumps(r))
        return value
    def batch(self,points,dtype='f4'):
        data=NeuralPoints(points,self.geo,self.c,dtype);x=cp.asarray(points,dtype='f8')
        G=self.scalar.operator(x,'grad');_,dg,_,_=lift_fields(x,self.geo,self.c)
        return data,G,dg.reshape(-1,1)-G@self.hco
    def reverse(self,model,data,G,h,obj,t):
        with timed(t,'harmonic_and_coefficient_reverse'):
            beta=h.T@obj['GA']/self.hm;rhs=-G.T@obj['GA']+self.Gfh@beta
        with timed(t,'poisson_transpose'):adj=self.solve(rhs,'transpose',True)
        with timed(t,'harmonic_reverse_fit'):adj-=self.hf@beta
        with timed(t,'parameter_reverse_query'):grad=network_backward(model,data,value=obj['GA'],curl=obj['GU'])
        with timed(t,'parameter_reverse_source'):grad=network_backward(model,self.fit,value=adj,grads=grad)
        return grad
    def compute(self,model,points,backward=True,dtype='f4'):
        t={};self.scalar.records=[]
        with timed(t,'sampling_geometry_operator'):data,G,h=self.batch(points,dtype)
        with timed(t,'neural_query_fields'):f=network_fields(model,data,'au')
        with timed(t,'neural_source_fields'):af=network_fields(model,self.fit,'a')['a']
        with timed(t,'poisson_forward'):co=self.solve(af,'forward')
        with timed(t,'harmonic_removal'):
            pf=af-self.fit_field(co);amp=self.hf.T@pf/self.hm;p=f['a']-G@co-h@amp
        with timed(t,'potential_objective_sort_GS'):obj=potential_objective(p,f['u'],data.n)
        result=dict(a=f['a'],u=f['u'],p=p,co=co,amp=amp,obj=obj,timings=t)
        if backward:result['grads']=self.reverse(model,data,G,h,obj,t)
        result['solvers']=list(self.scalar.records)
        return result

def state_equal(a,b):
    return a.epoch==b.epoch and all(bool(cp.array_equal(x,y)) for name in ('W','b','mW','vW','mb','vb') for x,y in zip(getattr(a,name),getattr(b,name)))

def check_sources():
    manifest=json.loads((D/'source_manifest.json').read_text())
    for path,digest in manifest.items():
        if sha(HERE/path)!=digest:raise RuntimeError('Frozen source mismatch: '+path)

def append_row(row):
    with (D/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())

def epoch_step(model,pipe,config_hash):
    tic=time.perf_counter();epoch=model.epoch+1;t={}
    with timed(t,'sampling'):points=fluid_points(pipe.c,pipe.geo,6144,908000+epoch)
    s=pipe.compute(model,points);t.update(s['timings'])
    assert bool(cp.all(cp.isfinite(s['obj']['losses'])))
    lr=1e-6+(.00028-1e-6)*.5*(1+math.cos(math.pi*(epoch-1)/1999))
    with timed(t,'adam'):norm=model.adam(*s['grads'],lr)
    if not all(bool(cp.all(cp.isfinite(x))) for name in ('W','b','mW','vW','mb','vb') for x in getattr(model,name)):raise FloatingPointError('Nonfinite optimizer state')
    row=dict(epoch=epoch,lr=lr,mean_rayleigh=float(cp.mean(s['obj']['losses'])),min_rayleigh=float(cp.min(s['obj']['losses'])),
        max_rayleigh=float(cp.max(s['obj']['losses'])),gram_max=float(s['obj']['gram_max']),order=cp.asnumpy(s['obj']['order']).tolist(),
        parameter_gradient_norm_min=float(norm.min()),parameter_gradient_norm_max=float(norm.max()),solvers=s['solvers'],
        host_bytes=psutil.Process().memory_info().rss,gpu_pool_reserved=cp.get_default_memory_pool().total_bytes())
    del s,norm

    with timed(t,'checkpoint_IO'):
        write(D/'pending_epoch.json',dict(**row,timings=t.copy(),pre_checkpoint_seconds=time.perf_counter()-tic))
        model.save(D/'checkpoint.npz',config_hash)
    row.update(timings=t,seconds=time.perf_counter()-tic)
    append_row(row)
    return row
