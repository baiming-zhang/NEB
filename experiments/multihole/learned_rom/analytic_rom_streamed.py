"""Exact slab contraction of full weak-Hodge tensors, without mode reduction."""
import cupy as cp
from .analytic_rom_integrator import AnalyticROM


class StreamedAnalyticROM(AnalyticROM):
    def __init__(self, G, D, slab_reader, block_size=32, nu=1/750):
        self.G=cp.asarray(G,dtype=cp.float64)
        self.D=cp.asarray(D,dtype=cp.float64)
        self.K=self.G.shape[0]
        assert self.G.shape==self.D.shape==(self.K,self.K)
        assert block_size>0 and nu>=0
        for value in (self.G,self.D):
            assert bool(cp.isfinite(value).all())
            assert float(cp.linalg.norm(value-value.T))<1e-10*max(float(cp.linalg.norm(value)),1e-30)
        cp.linalg.cholesky(self.G)
        assert float(cp.linalg.eigvalsh(self.D).min())>=-1e-10*max(float(cp.linalg.norm(self.D)),1e-30)
        self.reader=slab_reader
        self.block_size=block_size
        self.nu=nu
        self.factors={}

    def nonlinear(self,c):
        assert c.ndim==2 and c.shape[0]==self.K
        result=cp.empty_like(c,dtype=cp.float64)
        for start in range(0,self.K,self.block_size):
            stop=min(start+self.block_size,self.K)
            slab=cp.asarray(self.reader(start,stop),dtype=cp.float64)
            assert slab.shape==(stop-start,self.K,self.K) and bool(cp.isfinite(slab).all())
            result[start:stop]=cp.einsum('ijk,js,ks->is',slab,c,c,optimize=True)
            del slab
        return result

    # step() and trajectory() are inherited unchanged. The full tensor's
    # physical symmetry/source audit belongs to its producer and is required
    # separately; this storage adapter does not grant physical qualification.
