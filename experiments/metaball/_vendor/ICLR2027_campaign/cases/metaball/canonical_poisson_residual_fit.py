




from canonical_poisson_polynomial import *


class ResidualFitPoisson:
    def __init__(self,c,geo,points,budget_seconds=300):
        self.deadline=time.perf_counter()+budget_seconds
        self.base=ScalarPoisson(c,geo,32,budget_seconds=budget_seconds)
        self.c,self.geo,self.centers=c,geo,self.base.centers
        self.fit_points=points;self.n=self.base.n
        L=self.base.operator(points,'lap')
        self.scales=cp.sqrt(cp.asarray(L.power(2).sum(axis=0)).reshape(-1))
        if bool(cp.any(self.scales<=0)):raise RuntimeError('Uncovered scalar column; no truncation')
        self.Y=L.multiply(1/self.scales).tocsr();self.YT=self.Y.T.tocsr()
        _,_,_,self.B=polynomial_fields(points,geo,c)
        self.Q,self.R=cp.linalg.qr(self.B,mode='reduced')
        self.poly_condition=float(cp.linalg.cond(self.R))
        p,_,_,_=polynomial_fields(self.centers,geo,c)
        C=p.T/self.base.center_e[None,:]/self.scales[None,:]
        self.Qc,Rc=cp.linalg.qr(C.T,mode='reduced')
        self.constraint_condition=float(cp.linalg.cond(Rc))
        if max(self.poly_condition,self.constraint_condition)>1e12:raise RuntimeError('Unresolved QR conditioning')
        qt_y=(self.YT@self.Q).T
        self.diagonal=1-cp.sum(qt_y**2,axis=0)
        if bool(cp.any(self.diagonal<=0)):raise RuntimeError('Nonpositive preconditioner; no clipping')
        self.solve_records=[]

    def P(self,x):return x-self.Qc@(self.Qc.T@x)
    def E(self,x):return x-self.Q@(self.Q.T@x)
    def H(self,x):return self.P(self.YT@self.E(self.Y@self.P(x)))
    def precondition(self,x):return self.P(self.P(x)/self.diagonal[:,None])

    def normal_solve(self,b,scale,tol=1e-8,maxiter=1500):
        tic=time.perf_counter();scale=cp.maximum(scale,1e-300)
        b=self.P(b);x=cp.zeros_like(b);r=b.copy();z=self.precondition(r);p=z.copy()
        rz=cp.sum(r*z,axis=0);history=[]
        active=cp.linalg.norm(r,axis=0)>tol*scale
        iteration=0
        while bool(cp.any(active)) and iteration<maxiter:
            if time.perf_counter()>self.deadline:break
            iteration+=1;hp=self.H(p);den=cp.sum(p*hp,axis=0)
            if bool(cp.any(active & (den<=0))):raise FloatingPointError('Nonpositive constrained normal direction')
            alpha=cp.where(active,rz/cp.where(active,den,1),0.)
            x+=p*alpha;r-=hp*alpha
            if iteration%50==0:
                r=b-self.H(x)
                history.append([iteration,*cp.asnumpy(cp.linalg.norm(r,axis=0)/scale).tolist()])
            active &= cp.linalg.norm(r,axis=0)>tol*scale
            zz=self.precondition(r);rn=cp.sum(r*zz,axis=0)
            beta=cp.where(active,rn/cp.where(cp.abs(rz)>1e-290,rz,1),0.)
            p=cp.where(active,zz+p*beta,0.);rz=rn
        x=self.P(x);true=cp.linalg.norm(b-self.H(x),axis=0)/scale
        record=dict(iterations=iteration,relative_normal_residuals=cp.asnumpy(true).tolist(),
                    converged=bool(cp.all(true<=tol*2)),seconds=time.perf_counter()-tic,history=history)
        self.solve_records.append(record)
        return x

    def solve(self,b,transpose=False):
        b=cp.asarray(b,dtype='f8')
        if b.ndim==1:b=b[:,None]
        if not transpose:
            raw=self.YT@b;rhs=self.P(self.YT@self.E(b))
            w=self.normal_solve(rhs,cp.linalg.norm(raw,axis=0))
            q=solve_triangular(self.R,self.Q.T@(b-self.Y@w),lower=False)
            answer=cp.concatenate((w/self.scales[:,None],q))
            self.solve_records[-1]['fit_relative_residuals']=cp.asnumpy(cp.linalg.norm(self.Y@w+self.B@q-b,axis=0)/cp.linalg.norm(b,axis=0)).tolist()
            self.solve_records[-1]['moment_relative_residual']=float(cp.linalg.norm(self.Qc.T@w)/cp.maximum(cp.linalg.norm(w),1e-300))
            return answer


        v=self.Q@solve_triangular(self.R.T,b[self.n:],lower=True)
        rhs=self.P(b[:self.n]/self.scales[:,None]-self.YT@v)
        z=self.normal_solve(rhs,cp.linalg.norm(rhs,axis=0))
        return v+self.E(self.Y@z)

    def operator(self,x,kind):
        _,v,g,l=polynomial_fields(x,self.geo,self.c)
        return hstack([self.base.operator(x,kind),csr_matrix({'value':v,'grad':g,'lap':l}[kind])],format='csr')
