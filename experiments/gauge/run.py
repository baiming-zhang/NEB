"""Finite constrained and spline gauge projectors, harmonic removal and GS."""
from pathlib import Path
import argparse,json,sys,numpy as np
METABALL=Path(__file__).resolve().parents[1]/'metaball'
sys.path.insert(0,str(METABALL))
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--method',choices=['finite','spline'],required=True);p.add_argument('--resolution',type=int,default=8)
    p.add_argument('--config',required=True);p.add_argument('--samples',required=True);p.add_argument('--source-fields',required=True);p.add_argument('--query-points',required=True);p.add_argument('--query-fields',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();from train import native_context,resource,write,V
    from prepare import nodes
    from types import SimpleNamespace
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.linalg import solve_triangular
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.config).read_text());c=SimpleNamespace(**config);K=int(c.modes)
    shared,cm,_=native_context(config,K,out/'analytic_runtime',Path(a.samples));rt=cm.rt;geo=cp.asarray(c.fixed_training_geometry,'f8')
    x=cp.asarray(np.load(a.samples)['fit'],'f8');q=cp.asarray(np.load(a.query_points),'f8')
    A=cp.asarray(np.load(a.source_fields),'f8').reshape(3*len(x),K);Aq=cp.asarray(np.load(a.query_fields),'f8').reshape(3*len(q),K)
    resource(cp,out)
    if a.method=='finite':
        small=sys.modules['small'];factory=small.ScalarPoisson
        small.ScalarPoisson=lambda config,g,n,**kw:factory(config,g,a.resolution,**kw)
        try:space=small.Small(c,geo,x)
        finally:small.ScalarPoisson=factory
        co,record=space.solve(A,'gauge_projection');G=space.operator(q,'grad')
        fitted=space.fit_field(co);query=G@co
    else:
        import itertools
        base=V/'gauge_kernels/weak.py';geo1=sys.modules['first_geometry']
        from canonical_poisson_native import geometry2
        def shape_geometry(xx):return geometry2(xx,geo,c)[:3]
        ns=dict(cp=cp,np=np,itertools=itertools,csr_matrix=csr_matrix,shape_geometry=shape_geometry)
        nodes(base,['SplineSpace','Normal','solve'],ns)
        space=ns['SplineSpace'](a.resolution);X=space.operators(x,('grad',))['grad']
        diagonal=cp.asarray(X.multiply(X).T@cp.ones(3*len(x))).ravel();space.active=cp.flatnonzero(diagonal>0.);space.n=len(space.active);X=X[:,space.active].tocsr()
        normal=ns['Normal'](X,cp.ones(3*len(x)));co,record=ns['solve'](normal,X.T@A);G=space.operators(q,('grad',))['grad'][:,space.active].tocsr();fitted=X@co;query=G@co
    geo1=sys.modules['first_geometry'];h=geo1.fields(x,geo,c)[3].reshape(-1,1);hq=geo1.fields(q,geo,c)[3].reshape(-1,1)
    if a.method=='finite':hc,_=space.solve(h,'harmonic');h-=space.fit_field(hc);hq-=G@hc
    else:hc,_=ns['solve'](normal,X.T@h);h-=X@hc;hq-=G@hc
    residual=A-fitted;alpha=h.T@residual/cp.sum(h*h);canonical=residual-h@alpha;assessment=Aq-query-hq@alpha
    np.savez(out/'projection.npz',coefficients=cp.asnumpy(co),harmonic_coefficients=cp.asnumpy(alpha),source=cp.asnumpy(canonical),assessment=cp.asnumpy(assessment))
    write(out/'complete.json',dict(method=a.method,resolution=a.resolution,K=K,source_points=len(x),query_points=len(q),scalar_solve=record,source_relative_mass=cp.asnumpy(cp.sum(canonical*canonical,0)/cp.sum(A*A,0)).tolist(),query_relative_mass=cp.asnumpy(cp.sum(assessment*assessment,0)/cp.sum(Aq*Aq,0)).tolist()))
if __name__=='__main__':main()
