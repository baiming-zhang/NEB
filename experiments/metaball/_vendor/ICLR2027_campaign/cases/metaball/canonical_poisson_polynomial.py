





from canonical_poisson_native import *
from cupyx.scipy.sparse import hstack

POWERS=[a for degree in range(3) for a in itertools.product(range(3),repeat=3) if sum(a)==degree]

def polynomial_fields(x,geo,c):
    y=2*x-1
    e,de,le,*_=geometry2(x,geo,c)
    monomial=lambda a:cp.prod(y**cp.asarray(a)[None,:],axis=1)
    p=cp.stack([monomial(a) for a in POWERS],axis=1)
    dp=cp.zeros((len(x),10,3));lp=cp.zeros((len(x),10))
    for j,a in enumerate(POWERS):
        for k in range(3):
            if a[k]:
                b=list(a);b[k]-=1;dp[:,j,k]=2*a[k]*monomial(b)
            if a[k]>1:
                b=list(a);b[k]-=2;lp[:,j]+=4*a[k]*(a[k]-1)*monomial(b)
    value=e[:,None]*p
    gradient=de[:,None,:]*p[:,:,None]+e[:,None,None]*dp
    lap=le[:,None]*p+2*cp.sum(de[:,None,:]*dp,axis=2)+e[:,None]*lp
    return p,value,gradient.transpose(0,2,1).reshape(-1,10),lap


class PolynomialPoisson:
    def __init__(self,c,geo,resolution=32,budget_seconds=240):
        self.base=ScalarPoisson(c,geo,resolution,budget_seconds=budget_seconds)
        self.L=self.base.L
        self.centers=self.base.centers;self.n=self.base.n;self.geo=geo;self.c=c
        p,_,_,self.B=polynomial_fields(self.centers,geo,c)
        self.C=p.T/self.base.center_e[None,:]
        self.C/=cp.linalg.norm(self.C,axis=1)[:,None]
        self.Z=self.base.solve(self.B)
        self.S=self.C@self.Z
        self.condition=float(cp.linalg.cond(self.S))
        if not math.isfinite(self.condition) or self.condition>1e12:
            raise FloatingPointError('Unresolved polynomial Schur complement condition')
        self.solve_records=[]

    def operator(self,points,kind):
        _,value,grad,lap=polynomial_fields(points,self.geo,self.c)
        poly={'value':value,'grad':grad,'lap':lap}[kind]
        return hstack([self.base.operator(points,kind),csr_matrix(poly)],format='csr')

    def solve(self,b,transpose=False):
        tic=time.perf_counter();b=cp.asarray(b,dtype='f8')
        if b.ndim==1:b=b[:,None]
        if not transpose:
            r=self.base.solve(b)
            q=cp.linalg.solve(self.S,self.C@r)
            z=r-self.Z@q
            answer=cp.concatenate((z,q),axis=0)
            residual=self.base.L@z+self.B@q-b
            side=self.C@z
            relative=float(cp.linalg.norm(residual)/cp.maximum(cp.linalg.norm(b),1e-300))
            constraint=float(cp.linalg.norm(side)/cp.maximum(cp.linalg.norm(answer),1e-300))
        else:
            a,d=b[:self.n],b[self.n:]
            w=cp.linalg.solve(self.S.T,d-self.Z.T@a)
            answer=self.base.solve(a+self.C.T@w,transpose=True)
            residual=self.base.LT@answer-self.C.T@w-a
            side=self.B.T@answer-d
            relative=float(cp.sqrt(cp.sum(residual**2)+cp.sum(side**2))/cp.maximum(cp.linalg.norm(b),1e-300))
            constraint=float(cp.linalg.norm(side)/cp.maximum(cp.linalg.norm(b),1e-300))
        row=dict(transpose=transpose,rhs=b.shape[1],relative_block_residual=relative,
                 relative_constraint_residual=constraint,seconds=time.perf_counter()-tic)
        row['passed']=relative<2e-7 and constraint<2e-7
        self.solve_records.append(row)
        if not row['passed']:raise RuntimeError('Polynomial block inverse failed: '+json.dumps(row))
        return answer
