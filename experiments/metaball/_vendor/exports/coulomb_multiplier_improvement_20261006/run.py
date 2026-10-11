
import os,sys,time,json,gc,hashlib,traceback,threading,types
from _neb_paths import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ['CUPY_CACHE_DIR']=str(HERE/'cupy_cache')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
OLD=ROOT/'exports/coulomb_multiplier_20261006'
source=(OLD/'run.py').read_text(encoding='utf-8')

source=source.replace("HERE=Path(__file__).resolve().parent", "HERE=Path("+repr(str(HERE))+")")
source=source.replace("DATA=Path('_data/metaball/coulomb_multiplier_20261006')", "DATA=HERE/'data'")
actual_stage=sys.argv[1];sys.argv=[__file__,'qualification']
cm=types.ModuleType('coulomb_shared');cm.__file__=str(OLD/'run.py')
exec(compile(source,str(OLD/'run.py'),'exec'),cm.__dict__)
sys.argv=[__file__,actual_stage];cm.STAGE=actual_stage;cm.DEST=HERE/actual_stage;cm.DEST.mkdir(exist_ok=True)
cp,np,rt=cm.cp,cm.np,cm.rt
from span_penalty import span_augmented,span_constraint_functional
FWD=rt.grouped8.network_fields;BWD=rt.grouped8.network_backward
def tune(chunk):
    rt.grouped8.network_fields=lambda m,d,which='aud',**kw:FWD(m,d,which,chunk=chunk)
    rt.grouped8.network_backward=lambda m,d,**kw:BWD(m,d,chunk=chunk,**kw)
def relative(a,b):return float(cp.linalg.norm(a-b)/cp.maximum(cp.linalg.norm(a),1e-30))
def mer(a,d,h,lam,eta,rho,scale,span):
    if span:return span_augmented(cp,a,d,h,lam,eta,rho,scale)
    return (*cm.augmented(a,d,h,lam,eta,rho,scale),{})
def canonical(model,source,assessment):
    fs=rt.grouped8.network_fields(model,source,'aud');fq=rt.grouped8.network_fields(model,assessment,'aud')
    obj=cm.train.bounded_potential_objective(fs['a'],fs['u'],source.n);T=obj['T'];rows={}
    for name,f,n in [('source',fs,source.n),('independent',fq,assessment.n)]:
        a=f['a']@T;u=f['u']@T;d=f['d']@T
        rows[name]=dict(points=n,div_curl=cm.distribution(cp.sum(d*d,0)/cp.sum(u*u,0)),
            div_mass=cm.distribution(cp.sum(d*d,0)/cp.sum(a*a,0)))
    del fs,fq,obj,T,a,u,d;cm.clear();return rows
