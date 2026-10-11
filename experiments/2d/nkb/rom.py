import time
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
class ROM:
    def __init__(self,M,D,tensor):
        self.M=cp.asarray(M);self.D=cp.asarray(D);self.tensor=tensor;self.k=len(M);self.j,self.l=cp.triu_indices(self.k);self.factors={}
    def nonlinear(self,c):
        p=c[self.j]*c[self.l];out=cp.empty_like(c)
        for first in range(0,self.k,16):
            q=self.tensor[first:first+16]
            if isinstance(q,np.ndarray):q=cp.asarray(q,blocking=True)
            out[first:first+len(q)]=q@p
        return out
    def step(self,c,dt):
        if dt not in self.factors:self.factors[dt]=cp.linalg.cholesky(self.M+.5*dt/750*self.D)
        L=self.factors[dt]
        solve=lambda b:solve_triangular(L.T,solve_triangular(L,b,lower=True),lower=False)
        half=solve(self.M@c+.5*dt*self.nonlinear(c))
        return solve((self.M-.5*dt/750*self.D)@c+dt*self.nonlinear(half))
    def trajectory(self,c0,dt,progress):
        c=cp.asarray(c0);frames=[c.copy()];start=time.perf_counter()
        for frame in range(1,51):
            for step in range(round(.02/dt)):c=self.step(c,dt)
            if not bool(cp.isfinite(c).all()):raise ArithmeticError('Nonfinite trajectory')
            frames.append(c.copy())
            if frame%10==0:progress('autonomous_IMEX',dt=dt,frames=frame+1,seconds=time.perf_counter()-start)
        cp.cuda.get_current_stream().synchronize();return cp.stack(frames),time.perf_counter()-start
