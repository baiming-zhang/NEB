"""CPU oracle for the unchanged full-space LOBPCG mathematical source."""
import ast,json,time,warnings,math
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse import linalg as splinalg
from .sparse_controls import assemble_p1,solve_modes
from .mesh_builder import build_domain_mesh
from .fem_cpu import NumpyBackend

def checks():
    start=time.perf_counter();source=Path(__file__).with_name('lobpcg_true_residual.py');tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    cp=NumpyBackend();namespace={'numpy':np,'cupy':cp,'linalg':np.linalg,'splinalg':splinalg,'warnings':warnings}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),str(source),'exec'),namespace)
    mesh=build_domain_mesh('one_circle',0,.4,64,32);free=np.setdiff1d(np.arange(len(mesh.xy)),np.unique(mesh.boundary_edges))
    Sfull,Mfull,S,M=assemble_p1(mesh.xy,mesh.tri,free,np,coo_matrix)
    count=3;initial=np.random.default_rng(20260928).normal(size=(len(free),count))
    rates,vectors,facts=solve_modes(S,M,initial,count,250,cp,diags,namespace['lobpcg'],namespace['TRUE_RESIDUAL_HISTORY'])
    from scipy.linalg import eigh
    truth=eigh(S.toarray(),M.toarray(),subset_by_index=(0,count-1))[0]
    relative=float(np.max(abs(rates-truth)/truth));assert relative<1e-8
    assert np.max(facts['relative_residuals'])<1e-8 and facts['energy_gram_max']<1e-8
    try:solve_modes(S,M,initial[:,:2],count,250,cp,diags,namespace['lobpcg'],[])
    except ValueError:pass
    else:raise AssertionError('Incomplete block accepted')
    return dict(CPU_only=True,native_CUDA_run=False,full_sparse_free_nodes=len(free),modes=count,same_original_LOBPCG_functions=True,eigenvalue_relative_error=relative,true_residual_max=float(np.max(facts['relative_residuals'])),energy_gram_max=facts['energy_gram_max'],actual_CPU_seconds=time.perf_counter()-start)
if __name__=='__main__':print(json.dumps(checks(),allow_nan=False))
