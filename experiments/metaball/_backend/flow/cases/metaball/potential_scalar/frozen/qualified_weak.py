import time
import numpy as np
from original_potential_k32_core import cp
from canonical_poisson_native import ScalarPoisson
from canonical_poisson_polynomial import polynomial_fields
from cupyx.scipy.linalg import solve_triangular
from cupyx.scipy.sparse import csr_matrix,hstack

class Weak:
    def __init__(self,c,geo,x):
        self.c,self.geo,self.x=c,geo,x;self.base=ScalarPoisson(c,geo,32);self.n=self.base.n
        G=self.base.operator(x,'grad');self.scales=cp.sqrt(cp.asarray(G.power(2).sum(axis=0)).ravel())
        assert bool(cp.all(self.scales>0));self.X=G.multiply(1/self.scales).tocsr();self.XT=self.X.T.tocsr();del G
        _,_,self.B,_=polynomial_fields(x,geo,c);self.Q,self.R=cp.linalg.qr(self.B,mode='reduced')
        p,_,_,_=polynomial_fields(self.base.centers,geo,c)
        C=p.T/self.base.center_e[None,:]/self.scales[None,:]
        self.Qc,Rc=cp.linalg.qr(C.T,mode='reduced')
        self.diagonal=1-cp.sum((self.XT@self.Q)**2,axis=1)
        assert bool(cp.all(self.diagonal>0))
        self.info=dict(kernel_columns=self.n,polynomial_columns=10,admissible_dof=self.n,
            gradient_shape=self.X.shape,gradient_nnz=self.X.nnz,
            sparse_forward_bytes=int(self.X.data.nbytes+self.X.indices.nbytes+self.X.indptr.nbytes),
            prohibited_dense_gradient_bytes=int(3*len(x)*(self.n+10)*8),prohibited_dense_gram_bytes=int((self.n+10)**2*8),
            polynomial_QR_condition=float(cp.linalg.cond(self.R)),constraint_QR_condition=float(cp.linalg.cond(Rc)),
            jacobi_diagonal_min=float(self.diagonal.min()),jacobi_diagonal_max=float(self.diagonal.max()),
            preconditioner='One fixed projected diagonal inverse; no search, regularization or truncation')
        self.records=[]
    def P(self,w):return w-self.Qc@(self.Qc.T@w)
    def E(self,a):return a-self.Q@(self.Q.T@a)
    def H(self,w):return self.P(self.XT@self.E(self.X@self.P(w)))
    def pre(self,r):return self.P(self.P(r)/self.diagonal[:,None])
    def cg(self,b,label,seconds):
        tic=time.perf_counter();end=tic+seconds;b=self.P(b);scale=cp.linalg.norm(b,axis=0);assert bool(cp.all(scale>0))
        w=cp.zeros_like(b);r=b.copy();z=self.pre(r);p=z.copy();rz=cp.sum(r*z,axis=0)
        active=cp.ones(b.shape[1],dtype=bool);history=[];alphas=[];betas=[];positive=True
        for it in range(1,4001):
            if time.perf_counter()>end:break
            hp=self.H(p);den=cp.sum(p*hp,axis=0)
            if bool(cp.any(active & (den<=0))):positive=False;break
            alpha=cp.where(active,rz/cp.where(active,den,1),0.)
            w+=p*alpha;r-=hp*alpha
            if it%25==0:
                r=b-self.H(w);rr=cp.linalg.norm(r,axis=0)/scale
                history.append([it,time.perf_counter()-tic,*cp.asnumpy(rr).tolist()])
                if it%100==0:pass
            rr=cp.linalg.norm(r,axis=0)/scale;active=rr>1e-9
            if not bool(cp.any(active)):break
            zn=self.pre(r);rn=cp.sum(r*zn,axis=0);beta=cp.where(active,rn/cp.where(cp.abs(rz)>1e-290,rz,1),0.)
            if it<25:alphas.append(cp.asnumpy(alpha));betas.append(cp.asnumpy(beta))
            p=cp.where(active,zn+p*beta,0.);rz=rn
        w=self.P(w);true=cp.linalg.norm(b-self.H(w),axis=0)/scale
        record=dict(label=label,iterations=it,seconds=time.perf_counter()-tic,converged=bool(cp.all(true<=2e-9)),
            relative_residuals=cp.asnumpy(true).tolist(),positive_visited_search_curvatures=positive,
            termination='converged' if bool(cp.all(true<=2e-9)) else ('nonpositive_curvature' if not positive else 'budget_or_iteration_limit'),history=history)

        estimates=[]
        if len(alphas)>2:
            aa=cp.asarray(np.array(alphas));bb=cp.asarray(np.array(betas))
            for k in range(b.shape[1]):
                if bool(cp.all(aa[:,k]>0)&cp.all(bb[:-1,k]>=0)):
                    diag=1/aa[:,k];diag[1:]+=bb[:-1,k]/aa[:-1,k]
                    off=cp.sqrt(bb[:-1,k])/aa[:-1,k];T=cp.diag(diag)+cp.diag(off,1)+cp.diag(off,-1);e=cp.linalg.eigvalsh(T)
                    estimates.append(dict(rhs=k,steps=len(alphas),ritz_min=float(e[0]),ritz_max=float(e[-1]),ratio=float(e[-1]/e[0])))
        record['early_preconditioned_krylov_ritz_estimates']=estimates
        self.records.append(record)
        return w,record
    def solve(self,a,label,seconds):
        w,record=self.cg(self.P(self.XT@self.E(a)),label,seconds)
        q=solve_triangular(self.R,self.Q.T@(a-self.X@w),lower=False)
        return cp.concatenate((w/self.scales[:,None],q)),record
    def transpose(self,b,label,seconds):
        v=self.Q@solve_triangular(self.R.T,b[self.n:],lower=True)
        w,record=self.cg(self.P(b[:self.n]/self.scales[:,None]-self.XT@v),label,seconds)
        return v+self.E(self.X@w),record
    def operator(self,x,kind):
        _,v,g,l=polynomial_fields(x,self.geo,self.c)
        return hstack([self.base.operator(x,kind),csr_matrix({'value':v,'grad':g,'lap':l}[kind])],format='csr')
    def field(self,x,co):return self.operator(x,'grad')@co
