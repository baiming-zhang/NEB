"""Original NKB curl-form tensor, all coordinates, output slabs and full mass."""
from pathlib import Path
import importlib.util,json,sys
def build(args):
    import numpy as np,cupy as cp
    from core import Model
    from domains import Domain
    from common import guard
    meta=json.loads(args.checkpoint.with_suffix('.json').read_text());c=meta['config'];assert meta['epoch']==2000 and c['case']==args.case
    k=c['K'];model=Model(cp,c['input_dim'],c['dim'],k,c['seed'],c['width'],c['hidden_layers'],dtype='float64');model.load(args.checkpoint)
    domain=Domain(c['case'],cp)
    top=Path(__file__).resolve().parents[1]/'_vendor/ICLR2027_campaign/cases/topology3'
    path=top/'random_streamfunction_20260928/preview_random_streamfunction.py'
    spec=importlib.util.spec_from_file_location('nkb_quintic',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    fields=[]
    for sample in range(20):
        field=mod.RandomField(args.case,sample)
        with np.load(args.fields/f'sample{sample:02d}.npz') as z:
            assert np.array_equal(field.raw,z['raw_lattice']);field.scale=float(z['global_scale'])
        fields.append(field)
    with np.load(args.quadrature) as z:points=z['points'];weights=z['weights']
    assert np.isfinite(points).all() and np.isfinite(weights).all() and (weights>0).all()
    args.output.mkdir(parents=True,exist_ok=False);stop=guard(cp,args.output)
    def evaluate(x):
        f,j=domain.encode(cp.asarray(x));v,jac,_=model.forward(f,j)
        return cp.ascontiguousarray(v[:,:,0].T),cp.ascontiguousarray(v[:,:,1].T),cp.ascontiguousarray((jac[:,:,1,0]-jac[:,:,0,1]).T)
    try:
        M=cp.zeros((k,k),dtype='f8');D=cp.zeros_like(M);rhs=cp.zeros((k,20),dtype='f8')
        for first in range(0,len(points),1024):
            x=points[first:first+1024];w=cp.asarray(weights[first:first+len(x)]);u,v,omega=evaluate(x)
            M+=(u*w)@u.T+(v*w)@v.T;D+=(omega*w)@omega.T
            target=np.stack([f.evaluate(x)[:,1:3] for f in fields],axis=2)
            rhs+=(u*w)@cp.asarray(target[:,0])+(v*w)@cp.asarray(target[:,1])
        assert bool(cp.all(cp.linalg.eigvalsh((M+M.T)/2)>0))
        c0=cp.linalg.solve(M,rhs);np.savez(args.output/'mass.npz',mass=cp.asnumpy(M),curl_stiffness=cp.asnumpy(D),initial_coefficients=cp.asnumpy(c0))
        j,l=cp.triu_indices(k);out=np.lib.format.open_memmap(args.output/'packed_Q.npy',mode='w+',dtype='<f8',shape=(k,k*(k+1)//2))
        outputs=min(k,max(1,(2*2**30)//(8*k*k)))
        for firstout in range(0,k,outputs):
            count=min(outputs,k-firstout);raw=cp.zeros((k,count,k),dtype='f8')
            for first in range(0,len(points),1024):
                x=points[first:first+1024];w=cp.asarray(weights[first:first+len(x)]);u,v,omega=evaluate(x)
                for mode in range(0,k,8):
                    cross=v[mode:mode+8,None,:]*u[None,firstout:firstout+count,:]-u[mode:mode+8,None,:]*v[None,firstout:firstout+count,:]
                    raw[mode:mode+8].reshape(-1,k)[:]+=(cross*w).reshape(-1,len(x))@omega.T
            for i in range(count):out[firstout+i]=cp.asnumpy(raw[j,i,l]+cp.where(j==l,0.,raw[l,i,j]))
            out.flush();del raw;cp.get_default_memory_pool().free_all_blocks()
        del out
        (args.output/'complete.json').write_text(json.dumps(dict(K=k,points=len(points),all_coordinates=True,full_mass=True,representation='Original symmetric-input-pair curl-form Q; physical area weights',optimizer_step=2000),indent=2))
    finally:stop.set()
