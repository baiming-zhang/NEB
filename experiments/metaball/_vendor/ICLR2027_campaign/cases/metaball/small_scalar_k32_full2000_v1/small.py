
import time
import runtime as rt
from runtime import cp,np,NeuralPoints
from canonical_poisson_native import ScalarPoisson
from canonical_poisson_polynomial import polynomial_fields
from cupyx.scipy.linalg import solve_triangular
from cupyx.scipy.sparse import csr_matrix,hstack
import first_geometry as geo1

class Small:
 def __init__(self,c,geo,x):
  self.c,self.geo,self.x=c,geo,x;self.records=[];self.timing={};t=time.perf_counter()
  self.base=ScalarPoisson(c,geo,8,radius_cells=3.,budget_seconds=600);self.n=self.base.n
  self.blocks=[];M=cp.zeros((self.n+10,self.n+10),dtype='f8')
  for first in range(0,len(x),2048):
   stop=min(first+2048,len(x));G,_=geo1.gradient(self.base,x[first:stop],geo,c)

   dense=G.toarray();M+=dense.T@dense;self.blocks.append((first,stop,G));del dense
  cp.cuda.Stream.null.synchronize();self.timing['centers_streamed_Gram_and_sparse_cache']=time.perf_counter()-t;t=time.perf_counter()
  p,_,_,_=polynomial_fields(self.base.centers,geo,c);self.C=p.T/self.base.center_e[None,:]
  norms=cp.sqrt(cp.diag(M)[:self.n]);assert bool(cp.all(norms>0))
  Q,R=cp.linalg.qr((self.C/norms[None,:]).T,mode='complete');constraint_rank=int(cp.linalg.matrix_rank(R[:10]));assert constraint_rank==10
  Z=Q[:,10:];T=cp.zeros((self.n+10,self.n),dtype='f8');T[:self.n,:self.n-10]=Z/norms[:,None];T[self.n:,self.n-10:]=cp.eye(10)
  A=T.T@M@T;d=cp.sqrt(cp.diag(A));assert bool(cp.all(d>0));self.U=T/d[None,:];self.K=self.U.T@M@self.U
  eig=cp.linalg.eigvalsh(self.K);rank=int(cp.sum(eig>eig[-1]*cp.finfo(cp.float64).eps*self.n));assert rank==self.n and float(eig[0])>0
  self.L=cp.linalg.cholesky(self.K);cp.cuda.Stream.null.synchronize();self.timing['constraint_QR_scaling_and_Cholesky']=time.perf_counter()-t
  self.info=dict(resolution=8,radius_cells=3,support_radius=self.base.radius,kernel_columns=self.n,polynomial_columns=10,constraints=10,effective_columns=self.n,constraint_rank=constraint_rank,Gram_rank=rank,Gram_min_eigenvalue=float(eig[0]),Gram_max_eigenvalue=float(eig[-1]),Gram_condition=float(eig[-1]/eig[0]),constraint_basis_relative=float(cp.linalg.norm(self.C@self.U[:self.n])/cp.linalg.norm(self.U)),sparse_cache_bytes=sum(G.data.nbytes+G.indices.nbytes+G.indptr.nbytes for _,_,G in self.blocks),max_dense_block_bytes=3*2048*(self.n+10)*8,forbidden_full_dense_bytes=3*len(x)*(self.n+10)*8,timings=self.timing,no_shift_no_truncation=True)
 def fit_field(self,co):
  out=cp.empty((3*len(self.x),co.shape[1]),dtype='f8')
  for first,stop,G in self.blocks:out[3*first:3*stop]=G@co
  return out
 def fit_transpose(self,a):
  out=cp.zeros((self.n+10,a.shape[1]),dtype='f8')
  for first,stop,G in self.blocks:out+=G.T@a[3*first:3*stop]
  return out
 def inverse(self,b):return solve_triangular(self.L.T,solve_triangular(self.L,b,lower=True),lower=False)
 def solve(self,a,label,seconds=600):
  t=time.perf_counter();rhs=self.U.T@self.fit_transpose(a);z=self.inverse(rhs);co=self.U@z
  true=self.U.T@self.fit_transpose(self.fit_field(co)-a);rr=cp.linalg.norm(true,axis=0)/cp.maximum(cp.linalg.norm(rhs,axis=0),1e-300)
  r=dict(label=label,converged=bool(cp.all(rr<=2e-9)),relative_residuals=cp.asnumpy(rr).tolist(),seconds=time.perf_counter()-t,method='one reusable constrained Cholesky',transpose=False)
  self.records.append(r);return co,r
 def transpose(self,b,label,seconds=600):
  t=time.perf_counter();rhs=self.U.T@b;z=self.inverse(rhs);co=self.U@z;out=self.fit_field(co)
  true=self.U.T@self.fit_transpose(out)-rhs;rr=cp.linalg.norm(true,axis=0)/cp.maximum(cp.linalg.norm(rhs,axis=0),1e-300)
  r=dict(label=label,converged=bool(cp.all(rr<=2e-9)),relative_residuals=cp.asnumpy(rr).tolist(),seconds=time.perf_counter()-t,method='same reusable constrained Cholesky',transpose=True)
  self.records.append(r);return out,r
 def operator(self,x,kind):
  if kind=='grad':return geo1.gradient(self.base,x,self.geo,self.c)[0]
  _,v,g,l=polynomial_fields(x,self.geo,self.c)
  return hstack([self.base.operator(x,kind),csr_matrix({'value':v,'lap':l}[kind])],format='csr')

class Pipeline(rt.Pipeline):
 def __init__(self,c,geo):
  self.c=c;self.geo=cp.asarray(geo,dtype='f8');self.setup_timings={};t=time.perf_counter()
  x=np.load(rt.CASE/'results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz')['fit'];self.x=cp.asarray(x,dtype='f8');self.fit=NeuralPoints(x,self.geo,c,'f4')
  self.scalar=Small(c,self.geo,self.x);self.setup_timings.update(self.scalar.timing)
  _,_,_,dlift=geo1.fields(self.x,self.geo,c);lf=dlift.reshape(-1,1);self.hco=self.solve(lf,'setup_harmonic');self.hf=lf-self.fit_field(self.hco);self.hm=cp.sum(self.hf*self.hf);assert float(self.hm)>0
  self.Gfh=self.fit_transpose(self.hf);cp.cuda.Stream.null.synchronize();self.setup_timings['whole_setup']=time.perf_counter()-t
 def fit_field(self,co):return self.scalar.fit_field(co)
 def fit_transpose(self,a):return self.scalar.fit_transpose(a)
 def batch(self,points,dtype='f4'):return geo1.batch(self,points,dtype)
