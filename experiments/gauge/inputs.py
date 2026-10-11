"""Analytic potential, divergence forcing and original MFS wall inputs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
METABALL=Path(__file__).resolve().parents[1]/'metaball'
sys.path.insert(0,str(METABALL))

def lattice(n,method):
    ids=np.arange(n**3,dtype=np.int64)
    ijk=np.stack((ids//n**2,(ids//n)%n,ids%n),axis=1)
    return (-.02+1.04*(ijk+.5)/n) if method=='fmm' else (-.06+1.12*ijk/n)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--checkpoint');p.add_argument('--samples');p.add_argument('--method',choices=['fmm','fourier'],default='fourier');p.add_argument('--resolution',type=int,default=193);p.add_argument('--boundary-level',type=int,default=2048);p.add_argument('--output',required=True)
    a=p.parse_args();import cupy as cp
    from train import native_context,resource
    from evaluate import load_model
    from types import SimpleNamespace
    sys.path.insert(0,str(METABALL/'_vendor/gauge_kernels'));import mfs,extension
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);c=SimpleNamespace(**json.loads(Path(a.config).read_text()));mfs.c=c;mfs.geo=cp.asarray(c.fixed_training_geometry,'f8');mfs.center=mfs.geo[1,:3]
    sources=[];fit=[]
    for j,count in enumerate((a.boundary_level,2*a.boundary_level)):
        _,x,n=mfs.walls(mfs.directions(count));sources.append(x[j]+.075*n[j]);_,x,_=mfs.walls(mfs.directions(3*count,.47));fit.append(x[j])
    np.save(out/'boundary_sources.npy',cp.asnumpy(cp.concatenate(sources)));np.save(out/'boundary_points.npy',cp.asnumpy(cp.concatenate(fit)))
    x=lattice(a.resolution,a.method);np.save(out/'forcing_points.npy',x)
    if a.checkpoint:
        if not a.samples:raise ValueError('Training source samples required for model construction')
        _,cm,_=native_context(vars(c),int(c.modes),out/'analytic_runtime',Path(a.samples));model=cm.train.Model(c,int(c.modes),c.seed);load_model(model,a.checkpoint)
        mm=np.lib.format.open_memmap(out/'forcing.npy',mode='w+',dtype='<f8',shape=(len(x),int(c.modes)))
        for first in range(0,len(x),2048):
            resource(cp,out);xx=cp.asarray(x[first:first+2048],'f8');data=cm.rt.NeuralPoints(xx,mfs.geo,c,'f8');data.envelope=extension.extension(xx,mfs.geo,c)
            mm[first:first+len(xx)]=cp.asnumpy(cm.rt.grouped8.network_fields(model,data,'d')['d'])
        mm.flush();del mm
    resource(cp,out)
    (out/'complete.json').write_text(json.dumps(dict(method=a.method,resolution=a.resolution,boundary_level=a.boundary_level,extension_power=2,forcing_precision='FP64',K=int(c.modes)),indent=2))
if __name__=='__main__':main()
