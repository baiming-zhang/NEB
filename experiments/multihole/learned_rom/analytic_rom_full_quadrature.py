"""Exact all-point factorization of the full analytic weak Galerkin tensor."""
import cupy as cp
from .analytic_rom_streamed import StreamedAnalyticROM

def full_quadrature_nonlinear(gradients,omega,weights,coefficients,point_block=1024):
    n,k,two=gradients.shape
    assert two==2 and omega.shape==(n,k) and weights.shape==(n,)
    c=cp.asarray(coefficients,dtype=cp.float64)
    assert c.ndim==2 and c.shape[0]==k and point_block>0
    result=cp.zeros_like(c)
    for start in range(0,n,point_block):
        stop=min(start+point_block,n)
        g=cp.asarray(gradients[start:stop],dtype=cp.float64)
        o=cp.asarray(omega[start:stop],dtype=cp.float64)
        w=cp.asarray(weights[start:stop],dtype=cp.float64)
        assert bool(cp.isfinite(g).all()) and bool(cp.isfinite(o).all())
        assert bool(cp.isfinite(w).all()) and bool((w>0).all())
        gx=g[:,:,0];gy=g[:,:,1];vorticity=o@c
        result+=gx.T@(w[:,None]*(gy@c)*vorticity)-gy.T@(w[:,None]*(gx@c)*vorticity)
    return result

class FullQuadratureAnalyticROM(StreamedAnalyticROM):
    def __init__(self,G,D,gradients,omega,weights,point_block=1024,nu=1/750):
        super().__init__(G,D,None,nu=nu)
        assert gradients.shape==(len(weights),self.K,2) and omega.shape==(len(weights),self.K)
        self.gradients=gradients;self.omega=omega;self.weights=weights;self.point_block=point_block
    def nonlinear(self,c):
        return full_quadrature_nonlinear(self.gradients,self.omega,self.weights,c,self.point_block)

def full_linear_moments(gradients,omega,weights,point_block=1024):
    n,k,two=gradients.shape;assert two==2 and omega.shape==(n,k) and weights.shape==(n,)
    G=cp.zeros((k,k),dtype=cp.float64);D=cp.zeros_like(G)
    for start in range(0,n,point_block):
        stop=min(start+point_block,n);g=cp.asarray(gradients[start:stop]);o=cp.asarray(omega[start:stop]);w=cp.asarray(weights[start:stop])
        assert bool(cp.isfinite(g).all()) and bool(cp.isfinite(o).all()) and bool(cp.isfinite(w).all()) and bool((w>0).all())
        G+=g[:,:,0].T@(w[:,None]*g[:,:,0])+g[:,:,1].T@(w[:,None]*g[:,:,1]);D+=o.T@(w[:,None]*o)
    return G,D
