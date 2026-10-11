"""Small portable mesh/full-mass/polynomial/target-format checks (CPU only)."""
import json,tempfile,time,sys,subprocess
from pathlib import Path
import numpy as np
from . import run
from .fem_cpu import assemble_operators,export_quadrature
from .mesh_builder import build_domain_mesh
from .random_fields import RandomField

def checks():
    start=time.perf_counter();records=[]
    with tempfile.TemporaryDirectory(prefix='neb-reference-') as d:
        root=Path(d)
        run.main(['--mode','mesh','--case','one_circle','--output',str(root/'mesh'),'--outer-points','64','--obstacle-points','32','--max-area','.4'])
        mesh=run.load_mesh(root/'mesh/mesh.npz','one_circle')
        for degree in (2,3):
            space,M,S,basis=assemble_operators(mesh,degree)
            assert basis['passed'] and M.shape==S.shape==(space.n,space.n)
            assert np.max(np.abs((M-M.T).data),initial=0)<1e-12
            assert np.max(np.abs((S-S.T).data),initial=0)<1e-12
            one=np.ones(space.n);assert abs(one@M@one-space.area.sum())<1e-10
            assert np.linalg.norm(S@one)<1e-10
            vector=np.random.default_rng(4).normal(size=space.n);assert vector@M@vector>0
            points,weights=export_quadrature(space,root/f'q{degree}.npz')
            assert points.shape==(space.nt*25,2) and np.all(weights>0)
            assert abs(weights.sum()-one@M@one)<1e-10
            records.append(dict(degree=degree,nodes=space.n,triangles=space.nt,volume=float(weights.sum()),basis=basis))
        run.main(['--mode','operators','--case','one_circle','--mesh',str(root/'mesh/mesh.npz'),'--output',str(root/'operators')])
        assert (root/'operators/quadrature.npz').is_file()
        field=RandomField('one_circle',0);field.scale=.017
        np.savez(root/'sample00.npz',raw_x=field.x,raw_y=field.y,raw_lattice=field.raw,trace_constants=field.constants,global_scale=field.scale,seed_hierarchy=field.seed)
        restored=run.load_field(root/'sample00.npz','one_circle',0)
        query=np.array([[2.,1.],[7.,-2.]])
        assert np.array_equal(field.evaluate(query),restored.evaluate(query))
        try:run.load_field(root/'sample00.npz','circle_square',0)
        except ValueError:pass
        else:raise AssertionError('Wrong field identity accepted')
        try:run.main(['--mode','mesh','--case','one_circle','--output',str(root/'mesh')])
        except FileExistsError:pass
        else:raise AssertionError('Existing output overwritten')
    assert 'cupy' not in sys.modules
    return dict(CPU_only=True,CUDA_imported=False,mesh_fullmass_checks=records,worker_fields_format_exact=True,public_mesh_and_operator_cli_executed=True,actual_CPU_seconds=time.perf_counter()-start)
if __name__=='__main__':print(json.dumps(checks(),allow_nan=False))
