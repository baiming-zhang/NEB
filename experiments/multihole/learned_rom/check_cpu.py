"""CPU oracle for the original analytic tensor factorization and public dynamics."""
from pathlib import Path
import ast,contextlib,importlib,json,sys,tempfile,time,types
import numpy as np
from scipy.linalg import solve_triangular

def checks():
    started=time.perf_counter();root=Path(__file__).parent
    ns={'cp':np,'cholsolve':lambda L,b:solve_triangular(L.T,solve_triangular(L,b,lower=True),lower=False)}
    for name in ('analytic_rom_operators','analytic_rom_integrator','analytic_rom_streamed','analytic_rom_full_quadrature'):
        tree=ast.parse((root/f'{name}.py').read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        exec(compile(tree,str(root/f'{name}.py'),'exec'),ns)
    rng=np.random.default_rng(93);g=rng.normal(size=(137,8,2));h=rng.normal(size=(137,8,3));w=rng.uniform(.1,1,size=137);c=rng.normal(size=(8,20));o=-(h[:,:,0]+h[:,:,2])
    M,D,Q,_=ns['moments'](g,h,w);m,d=ns['full_linear_moments'](g,o,w,point_block=31)
    assert np.allclose(M,m,rtol=1e-13,atol=1e-12) and np.allclose(D,d,rtol=1e-13,atol=1e-12)
    direct=ns['full_quadrature_nonlinear'](g,o,w,c,point_block=31)
    tensor=np.einsum('ijk,js,ks->is',Q,c,c,optimize=True)
    assert np.allclose(direct,tensor,rtol=1e-12,atol=1e-10)
    full=ns['FullQuadratureAnalyticROM'](M,D,g,o,w,point_block=31)
    dense=ns['AnalyticROM'](M,D,Q)
    assert np.allclose(full.step(c,.0001),dense.step(c,.0001),rtol=1e-13,atol=1e-12)
    # Public argv/parser/read-write return uses NumPy CUDA-API double in this CPU test.
    cp=types.ModuleType('cupy');cp.__dict__.update(np.__dict__);cp.__name__='cupy';cp.asnumpy=np.asarray
    cp.cuda=types.SimpleNamespace(Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda:None)))
    tri=types.ModuleType('cupyx.scipy.linalg');tri.solve_triangular=solve_triangular
    replacements={'cupy':cp,'cupyx':types.ModuleType('cupyx'),'cupyx.scipy':types.ModuleType('cupyx.scipy'),'cupyx.scipy.linalg':tri}
    prefix='experiments.multihole.learned_rom.'
    saved={key:sys.modules.get(key) for key in replacements};sys.modules.update(replacements)
    from . import run
    runtime=importlib.import_module('experiments.multihole.runtime');oldguard=runtime.guard
    runtime.guard=lambda *_:types.SimpleNamespace(set=lambda:None)
    try:
        with tempfile.TemporaryDirectory(prefix='neb-learned-rom-') as temp:
            tmp=Path(temp);ops=tmp/'operators';ops.mkdir()
            np.save(ops/'gradients.npy',g);np.save(ops/'omega.npy',o)
            np.savez(ops/'mass.npz',mass=M,curl_stiffness=D,initial_coefficients=c,weights=w)
            (ops/'operators.json').write_text(json.dumps(dict(K=8,case='one_circle',CPU_synthetic_metadata=True)))
            run.main(['--mode','dynamics','--operators',str(ops),'--output',str(tmp/'trajectory'),'--dt','.0001','--final-time','.0004','--snapshot-dt','.0002'])
            with np.load(tmp/'trajectory/trajectory.npz') as z:
                assert z['coefficients'].shape==(3,8,20) and np.array_equal(z['t'],[0,.0002,.0004])
                expected=full.trajectory(c,.0001,.0004,.0002)
                assert np.allclose(z['coefficients'],expected,rtol=1e-13,atol=1e-12)
            assert json.loads((tmp/'trajectory/trajectory.json').read_text())['samples']==20
    finally:
        runtime.guard=oldguard
        for key,previous in saved.items():
            if previous is None:sys.modules.pop(key,None)
            else:sys.modules[key]=previous
        for key in list(sys.modules):
            if key.startswith(prefix) and key not in (prefix+'run',):sys.modules.pop(key,None)
    return dict(CPU_only=True,CUDA_initialized=False,NumPy_API_double=True,original_fullQ_vs_factorization=True,original_fullmass_IMEX_vs_dense=True,public_dynamics_cli_write_return=True,all20_initial_states_independent=True,actual_seconds=time.perf_counter()-started)
if __name__=='__main__':print(json.dumps(checks()))
