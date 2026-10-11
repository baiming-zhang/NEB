"""Complete analytic bases, mapped LEB fields and held-out geometry diagnostics."""
from __future__ import annotations
import argparse,json,os,sys,time,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from train import HERE,V,load,native_context,canonical_pipeline,resource

def points(flat,coordinates):
    return np.stack((coordinates[flat%128],coordinates[flat//128%128],coordinates[flat//16384]),axis=1)

def load_model(model,path):
    with np.load(path,allow_pickle=False) as z:
        digest=str(z['config_hash']);epoch=int(z['epoch'])
        if epoch!=2000:raise ValueError('Complete 2000-update checkpoint required')
        for name in ('W','b','mW','vW','mb','vb'):
            for i,a in enumerate(getattr(model,name)):
                if a.shape!=z[f'{name}_{i}'].shape:raise ValueError('All requested modes and layers required')
    model.load(path,digest)

def field_provider(args,c,out):
    if args.method=='NKB':
        d=V/'ICLR2027_campaign/nkb_baseline_20260928';sys.path.insert(0,str(d))
        core=load('core',d/'core.py');domains=load('domains',d/'domains.py')
        domain=domains.Domain.__new__(domains.Domain);domain.xp=np;domain.periodic=False;domain.dim=3
        domain.descriptor=np.asarray(c.fixed_training_geometry).ravel();domain.lo=np.zeros(3);domain.hi=np.ones(3)
        f,_=domain.encode(np.zeros((1,3),'f8'));model=core.Model(np,f.shape[1],3,args.K,c.seed,dtype='float64');model.load(args.checkpoint)
        if model.step!=2000:raise ValueError('Complete NKB checkpoint required')
        def fields(x):
            f,_=domain.encode(np.asarray(x,'f8'));value,_,_=model.forward(f)
            return value.transpose(0,2,1)
        return fields,lambda:None
    if args.method=='LEB':
        import scipy.sparse as sp
        mapping=load('public_mapping',V/'exports/projected_potential_loss_20261008/implementation/source_curl_mapping.py')
        entry=load('public_canonical_entry',V/'exports/projected_potential_loss_20261008/support/train_quotient_entry.py')
        with np.load(args.mesh,allow_pickle=False) as z:p=z['p'].T.copy();t=z['t'].T.copy();free=z['free_edges'].copy();co=z['potential'][:,:args.K].copy();lam=z['eigenvalues'][:args.K].copy()
        if len(lam)!=args.K or np.any(lam<=0):raise ValueError('Complete positive LEB spectrum required')
        pipe=entry.MethodPipeline.__new__(entry.MethodPipeline);pipe.mapping=mapping;pipe.sp=sp;pipe.p=p;pipe.t=t;pipe.free=free;pipe.locator=None;pipe.log=lambda *a,**k:None
        pipe.radial=mapping.RadialCPU(p,t,np.asarray(c.fixed_training_geometry,'f8'),vars(c),pipe.log)
        edges=np.unique(np.sort(t[:,np.array(mapping.PAIRS)].reshape(-1,2),axis=1),axis=0);codes=edges[:,0]*len(p)+edges[:,1]
        local=np.sort(t[:,np.array(mapping.PAIRS)],axis=2);ids=np.searchsorted(codes,local[:,:,0]*len(p)+local[:,:,1]);slots=np.full(len(edges),-1,'i8');slots[free]=np.arange(len(free));pipe.cell_slots=slots[ids]
        with np.load(Path(args.physical)/'complex.npz') as z:H=z['H'].copy()
        pipe.inverse=SimpleNamespace(G=sp.load_npz(Path(args.physical)/'G.npz'),H=H)
        def fields(x):return (pipe.operator(x,'LEB_fullpoint')@co/np.sqrt(lam)[None,:]).reshape(len(x),3,args.K)
        return fields,lambda:None
    import cupy as cp
    if args.method=='VRQ':
        case=V/'ICLR2027_campaign/cases/metaball';sys.path.insert(0,str(case));d=case/'velocity_k1024_full2000_v1';sys.path.insert(0,str(d))
        core=load('core',d/'core.py');load('fused',d/'fused.py').install();load('fast_path',d/'fast_path.py').install()
        model=core.Model(c,args.K,c.seed);load_model(model,args.checkpoint);geo=cp.asarray(c.fixed_training_geometry,'f4')
        def fields(x):
            resource(cp,out);data=core.FieldData(cp.asarray(x,'f4'),geo,c);u,_=core.fields(model,data,args.K,c.mode_batch)
            return cp.asnumpy(u).reshape(len(x),3,args.K).astype('f8')
        return fields,lambda:None
    shared,cm,_=native_context(vars(c),args.K,out/'analytic_runtime',Path(args.samples))
    model=cm.train.Model(c,args.K,c.seed);load_model(model,args.checkpoint);geo=cp.asarray(c.fixed_training_geometry,'f8')
    def raw(x):
        resource(cp,out);data=cm.rt.NeuralPoints(x,geo,c,'f4');kind={'curl':'u','potential':'a','divergence':'d'}[getattr(args,'field','curl')];return cp.asnumpy(cm.rt.grouped8.network_fields(model,data,kind)[kind]).astype('f8')
    if args.method!='CCQ':return lambda x:raw(x).reshape(len(x),1 if getattr(args,'field','curl')=='divergence' else 3,args.K),lambda:None
    if getattr(args,'field','curl')!='curl':raise ValueError('CCQ represents velocity through its canonical curl lift')
    entry=load('public_canonical_entry',V/'exports/projected_potential_loss_20261008/support/train_quotient_entry.py');entry.K=args.K
    source=np.load(args.samples,allow_pickle=False)['fit'];ind=cm.rt.fluid_points(c,geo,16384,919000)
    state=dict(source_points=source,independent_points=ind)
    pipe=canonical_pipeline(entry,args,vars(c),state,cp,out)
    fitted=pipe.evaluate(raw(source),raw(ind),pipe.independent,reverse=False,label='complete_fitted_CCQ')
    Z=fitted['full_linear_fields']['lift_Z']
    return lambda x:(pipe.operator(x,'all_points')@Z).reshape(len(x),3,args.K),pipe.close

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--method',choices=['CCQ','VRQ','GCP-AL','GCP-FG-historical','NKB','LEB'],required=True)
    p.add_argument('--K',type=int,required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint');p.add_argument('--flat');p.add_argument('--coordinates');p.add_argument('--points');p.add_argument('--output',required=True)
    p.add_argument('--field',choices=['curl','potential','divergence'],default='curl')
    p.add_argument('--mesh');p.add_argument('--data-dir',default='data/metaball');p.add_argument('--samples');p.add_argument('--physical');p.add_argument('--source-operator');p.add_argument('--source-map');p.add_argument('--physical-volume',type=float,default=.7585406920021169);p.add_argument('--canonical-module');p.add_argument('--canonical-class',default='CanonicalLift');p.add_argument('--factor-options')
    args=p.parse_args();os.environ['NEB_DATA_DIR']=str(Path(args.data_dir).resolve());sys.path.insert(0,str(V))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);os.environ['CUPY_CACHE_DIR']=str(out/'cupy_cache')
    c=SimpleNamespace(**json.loads(Path(args.config).read_text()));flat=np.load(args.flat,allow_pickle=False) if args.flat else None;coord=np.load(args.coordinates,allow_pickle=False) if args.coordinates else None;explicit=np.load(args.points,allow_pickle=False) if args.points else None
    if explicit is None and (flat is None or coord is None or flat.shape!=(1599754,) or coord.shape!=(128,)):raise ValueError('Original complete grid/point order required')
    if args.method!='LEB' and not args.checkpoint:raise ValueError('Reference checkpoint required')
    if args.field!='curl' and args.method not in ('GCP-AL','GCP-FG-historical'):raise ValueError('Analytic potential and divergence use potential models')
    if explicit is not None and (explicit.ndim!=2 or explicit.shape[1]!=3 or not np.isfinite(explicit).all()):raise ValueError('Finite ordered XYZ points required')
    count=len(explicit) if explicit is not None else len(flat)
    start=time.perf_counter();get,close=field_provider(args,c,out)
    try:
        with (out/'basis.npy').open('xb') as f:
            components=1 if args.field=='divergence' else 3
            np.lib.format.write_array_header_1_0(f,dict(shape=(count,components,args.K),fortran_order=False,descr='<f8'))
            for first in range(0,count,2048):
                from run import guard
                guard();x=explicit[first:first+2048] if explicit is not None else points(flat[first:first+2048],coord);b=np.asarray(get(x),'<f8')
                if b.shape!=(len(x),components,args.K) or not np.isfinite(b).all():raise FloatingPointError('Complete finite all-mode fields required')
                f.write(b.tobytes(order='C'));del b;gc.collect()
            f.flush();os.fsync(f.fileno())
        from run import fullmass
        if explicit is None and args.field=='curl':fullmass(SimpleNamespace(basis=str(out/'basis.npy'),K=args.K,output=str(out/'mass.npz')))
        (out/'complete.json').write_text(json.dumps(dict(method=args.method,K=args.K,points=count,basis_precision='FP64',contractions='FP64',actual_seconds=time.perf_counter()-start),indent=2))
    finally:close()
if __name__=='__main__':main()
