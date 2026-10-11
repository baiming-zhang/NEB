"""Original constant-cell volume FMM and source-FFT Poisson controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--method',choices=['fmm','fourier'],required=True);p.add_argument('--resolution',type=int,required=True);p.add_argument('--order',type=int,default=6);p.add_argument('--forcing',required=True);p.add_argument('--points',required=True);p.add_argument('--output',required=True)
    p.add_argument('--boundary-points');p.add_argument('--boundary-sources')
    a=p.parse_args();import cupy as cp
    d=Path(__file__).resolve().parents[1]/'metaball';sys.path.insert(0,str(d));sys.path.insert(0,str(d/'_vendor/gauge_kernels'))
    from train import resource
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);resource(cp,out)
    forcing=np.load(a.forcing,allow_pickle=False).reshape(a.resolution**3,-1);x=cp.asarray(np.load(a.points,allow_pickle=False),'f8')
    if forcing.shape[0]!=a.resolution**3 or forcing.ndim!=2 or not np.isfinite(forcing).all():raise ValueError('Complete finite cell-major forcing columns required')
    if bool(a.boundary_points)!=bool(a.boundary_sources):raise ValueError('Both MFS boundary arrays required')
    xb=cp.asarray(np.load(a.boundary_points),'f8') if a.boundary_points else None
    boundary_values=None
    if a.method=='fmm':
        from volume_fmm import VolumeFMM
        op=VolumeFMM(a.resolution,a.order);state=op.forward(cp.asarray(forcing,'f8'));result=op.evaluate(state,x)
        arrays={k:cp.asnumpy(v) for k,v in result.items()}
        if xb is not None:boundary_values=op.evaluate(state,xb)['value']
    else:
        from fourier import ReusePoisson
        op=ReusePoisson(a.resolution);parts={k:[] for k in ('value','gradient','lap')}
        bparts=[]
        for col in range(forcing.shape[1]):
            resource(cp,out);op.prepare(cp.asarray(forcing[:,col],'f8'))
            for k in parts:parts[k].append(cp.asnumpy(op.evaluate(x,k)))
            if xb is not None:bparts.append(op.evaluate(xb,'value').copy())
        arrays={k:np.stack(v,axis=-1) for k,v in parts.items()}
        if xb is not None:boundary_values=cp.stack(bparts,axis=-1)
    if xb is not None:
        import boundary
        boundary.RUN=out;sources=cp.asarray(np.load(a.boundary_sources),'f8');factor=boundary.Boundary(sources,xb);co=factor.forward(-boundary_values)
        harmonic=boundary.mfs_evaluate(x,sources,co)
        arrays['value']+=cp.asnumpy(harmonic['value']);arrays['gradient']+=cp.asnumpy(harmonic['gradient'])
        arrays['boundary_coefficients']=cp.asnumpy(co)
    resource(cp,out);np.savez(out/'particular.npz',**arrays)
    (out/'complete.json').write_text(json.dumps(dict(method=a.method,resolution=a.resolution,far_order=a.order,columns=forcing.shape[1],points=len(x)),indent=2))
if __name__=='__main__':main()
