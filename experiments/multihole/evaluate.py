"""Full-mass analytic P/PH projection of the fixed twenty quintic fields.

Use reference-generated same-domain points/weights. Two passes compute RHS
and independently integrate physical residuals; no blockwise normalization.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--basis-family',choices=['spectral','NKB'],default='spectral')
    p.add_argument('--harmonic-checkpoint',type=Path)
    p.add_argument('--causal-order',type=Path)
    p.add_argument('--total-K',type=int)
    p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--quadrature',type=Path,required=True,help='NPZ points[N,2], weights[N] from the reference generator')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    base=Path(__file__).resolve().parents[1]/'2d/_backend';top=base/'flow/cases/topology3'
    os.environ['NEB_2D_VENDOR_ROOT']=str(base)
    sys.path[:0]=[str(top/'learned_basis/positive/src'),str(top/'learned_basis/harmonic/scripts')]
    import numpy as np,cupy as cp
    from neb2d.basisnet.metaball2d import hardcoded_backend as be
    from harmonic_common import learned_harmonic_fields
    from train_harmonic import load_checkpoint
    meta=json.loads(a.checkpoint.with_suffix('.json').read_text());assert meta['epoch']==2000
    if a.basis_family=='NKB':
        from types import SimpleNamespace
        sys.path.insert(0,str(base.parent/'nkb'))
        from core import Model
        from domains import Domain
        c=meta['config'];cfg=SimpleNamespace(subcase=c['case'],modes=c['K'])
        model=Model(cp,c['input_dim'],c['dim'],c['K'],c['seed'],c['width'],c['hidden_layers'],dtype='float64');model.load(a.checkpoint)
        domain=Domain(c['case'],cp)
        assert not a.harmonic_checkpoint
    else:
        cfg=be.cfg_namespace(meta['config'])
        with np.load(a.checkpoint) as z:model=be.HardcodedModel(cfg,{k:z[k] for k in z.files})
    hmodel=None;h=0
    if a.harmonic_checkpoint:
        hm,ha=load_checkpoint(a.harmonic_checkpoint);assert hm['epoch']==2000
        hc=be.cfg_namespace(hm['config']);assert hc.subcase==cfg.subcase
        hmodel=be.HardcodedModel(hc,ha);h=hc.modes
    K=a.total_K or cfg.modes+h;positive=K-h
    assert 0<positive<=cfg.modes and h<K
    order=np.arange(cfg.modes)
    if positive<cfg.modes:
        if not a.causal_order:p.error('P(K-h)H(h) requires the actual final causal-order receipt')
        order=np.asarray(json.loads(a.causal_order.read_text())['details'][0]['causal_order']);assert sorted(order.tolist())==list(range(cfg.modes))
    selected=order[:positive]
    path=top/'random_streamfunctions/preview_random_streamfunction.py'
    spec=importlib.util.spec_from_file_location('neb_quintic',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    fields=[]
    for i in range(20):
        field=mod.RandomField(cfg.subcase,i)
        with np.load(a.fields/f'sample{i:02d}.npz') as z:
            assert np.array_equal(field.raw,z['raw_lattice']) and np.array_equal(field.seed,z['seed_hierarchy'])
            assert np.array_equal(field.constants,z['trace_constants'])
            field.scale=float(z['global_scale'])
        fields.append(field)
    with np.load(a.quadrature) as z:points=z['points'];weights=z['weights']
    assert points.shape==(len(weights),2) and np.isfinite(points).all() and np.isfinite(weights).all() and (weights>0).all()
    def basis(chunk):
        if a.basis_family=='NKB':
            f,j=domain.encode(cp.asarray(chunk));v,_,_=model.forward(f,j)
            return cp.asnumpy(v).transpose(0,2,1)
        G=np.empty((len(chunk),2,cfg.modes))
        for mode,field,_ in model.forward_stream_jet1(cfg,chunk,0,0,False):G[:,:,int(mode.mode_index)]=cp.asnumpy(field.grad[:,0])
        gradients=G[:,:,selected]
        if hmodel is not None:
            H=np.empty((len(chunk),2,h))
            for j,(_,field,_) in enumerate(learned_harmonic_fields(hmodel,hc,chunk,False)):H[:,:,j]=cp.asnumpy(field.grad[:,0])
            gradients=np.concatenate((gradients,H),axis=2)
        return np.stack((gradients[:,1,:],-gradients[:,0,:]),axis=1)
    from runtime import guard
    a.output.mkdir(parents=True,exist_ok=False);stop=guard(cp,a.output)
    try:
        start=time.perf_counter();M=np.zeros((K,K));rhs=np.zeros((K,20));norm=np.zeros(20)
        for first in range(0,len(points),2048):
            x=points[first:first+2048];w=weights[first:first+len(x)];B=basis(x).reshape(-1,K)
            T=np.stack([f.evaluate(x)[:,1:3] for f in fields],axis=2).reshape(-1,20);ww=np.repeat(w,2)
            M+=B.T@(ww[:,None]*B);rhs+=B.T@(ww[:,None]*T);norm+=np.sum(ww[:,None]*T*T,axis=0)
        ev=np.linalg.eigvalsh((M+M.T)/2);assert ev[0]>ev[-1]*1e-12
        c=np.linalg.solve(M,rhs);direct=np.zeros(20)
        for first in range(0,len(points),2048):
            x=points[first:first+2048];w=weights[first:first+len(x)];B=basis(x).reshape(-1,K)
            T=np.stack([f.evaluate(x)[:,1:3] for f in fields],axis=2).reshape(-1,20)
            direct+=np.sum(np.repeat(w,2)[:,None]*(B@c-T)**2,axis=0)
        errors=np.sqrt(direct/norm)
        np.savez(a.output/'projection.npz',mass=M,rhs=rhs,coefficients=c,target_norm_squared=norm,direct_residual_squared=direct,relative_L2=errors)
        np.savetxt(a.output/'per_case.csv',np.column_stack((np.arange(20),errors)),delimiter=',',header='case,relative_L2',comments='')
        (a.output/'summary.json').write_text(json.dumps(dict(K=K,positive=positive,harmonic=h,case=cfg.subcase,n=20,points=len(points),mean=float(errors.mean()),sampleSD=float(errors.std(ddof=1)),seconds=time.perf_counter()-start,checkpoint_epoch=2000,blockwise_GS=False,full_mass_rank=K,reference_convergence_not_implied=True),indent=2))
    finally:stop.set()
if __name__=='__main__':main()
