"""CPU construction and finite-element operator checks; production GT uses native CuPy."""
from pathlib import Path
import ast,math,types
import numpy as np
from scipy.sparse import coo_matrix
from .mesh_builder import DomainMesh
class NumpyBackend:
    def __getattr__(self,name):return getattr(np,name)
    def asnumpy(self,value):return np.asarray(value)
def finite_element_space(mesh,degree=3,quadrature_order=5):
    source=Path(__file__).with_name('high_order_reference.py')
    tree=ast.parse(source.read_text(encoding='utf-8'));names={'multiindices','basis','quadrature','validate_basis','Space'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    namespace={'np':np,'math':math,'cp':NumpyBackend()}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),str(source),'exec'),namespace)
    return namespace['Space'](mesh,degree,quadrature_order),namespace['validate_basis']()
def assemble_operators(mesh,degree=3,quadrature_order=5):
    space,checks=finite_element_space(mesh,degree,quadrature_order)
    shape=(space.nt,space.nl,space.nl);local_s=np.empty(shape);local_m=np.empty(shape)
    mass=np.einsum('qi,qj,q->ij',space.N,space.N,space.weights)
    for first in range(0,space.nt,1024):
        last=min(first+1024,space.nt);g=space.gradients(first,last)
        local_s[first:last]=np.einsum('tqid,tqjd,q,t->tij',g,g,space.weights,space.area[first:last])
        local_m[first:last]=mass[None,:,:]*space.area[first:last,None,None]
    rows=np.broadcast_to(space.cells[:,:,None],shape).ravel();cols=np.broadcast_to(space.cells[:,None,:],shape).ravel()
    stiffness=coo_matrix((local_s.ravel(),(rows,cols)),shape=(space.n,space.n)).tocsr()
    mass_matrix=coo_matrix((local_m.ravel(),(rows,cols)),shape=(space.n,space.n)).tocsr()
    return space,mass_matrix,stiffness,checks

def export_quadrature(space,path):
    """Physical cell-major product-Gauss points and unnormalized volume weights."""
    points=np.einsum('qi,tij->tqj',space.bary,space.corners).reshape(-1,2)
    weights=(space.area[:,None]*space.weights[None,:]).reshape(-1)
    np.savez_compressed(path,points=points,weights=weights,cell_index=np.repeat(np.arange(space.nt),len(space.weights)),barycentric=space.bary)
    return points,weights
