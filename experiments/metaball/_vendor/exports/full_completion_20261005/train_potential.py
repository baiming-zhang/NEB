





import os,sys,time,json,math,gc,hashlib,traceback,argparse
from _neb_paths import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;CASE=ROOT/'ICLR2027_campaign/cases/metaball'
sys.path.insert(0,str(ROOT/'ICLR2027_campaign/nkb_baseline_20260928'))
from common import guard,read,write
sys.path.insert(0,str(CASE/'small_scalar_k32_full2000_v1'))
import runtime as rt
import value_reverse as vr
cp,np,Model=rt.cp,rt.np,rt.Model
OUT=Path('_data/metaball/potential_capacity_20261005')
OLD=CASE/'results/small_scalar_k32_full2000_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def bounded_potential_objective(A,U,n):



 K=A.shape[1];P=A;order=cp.argsort(cp.sum(U*U,axis=0)/cp.sum(P*P,axis=0),kind='stable')
 Q=cp.empty_like(P);V=cp.empty_like(U);T=cp.zeros((K,K),dtype='f8');GA=cp.zeros_like(P);GU=cp.zeros_like(U)
 losses=cp.empty(K);masses=cp.empty(K)
 for rank,index in enumerate(cp.asnumpy(order)):
  p=P[:,index];v=U[:,index];normal=cp.sqrt(cp.dot(p,p)/n)
  a=p/normal;u=v/normal;t=cp.zeros(K,dtype='f8');t[index]=1/normal
  q=Q[:,:rank];w=V[:,:rank]
  for _ in range(2):
   alpha=q.T@a/n;a-=q@alpha;u-=w@alpha;t-=T[:,:rank]@alpha
  mass=cp.dot(a,a)/n;stiff=cp.dot(u,u)/n;losses[rank]=stiff/mass;masses[rank]=mass
  ga=-2*stiff*a/(n*mass*mass);gu=2*u/(n*mass)
  for _ in range(2):ga-=q@(q.T@ga+w.T@gu)/n
  gp=ga/normal-p*(cp.dot(ga,p)+cp.dot(gu,v))/(n*normal**3)
  GA[:,index]=gp;GU[:,index]=gu/normal
  norm=cp.sqrt(mass);Q[:,rank]=a/norm;V[:,rank]=u/norm;T[:,rank]=t/norm
  if (rank+1)%32==0:
   cp.cuda.Stream.null.synchronize();cp.get_default_memory_pool().free_all_blocks()
 if not bool(cp.all(cp.isfinite(losses)&(masses>1e-12))):raise FloatingPointError('Invalid canonical quotient/rank')
 return dict(losses=losses,order=order,T=T,Q=Q,U=V,GA=GA,GU=GU,
             gram_max=cp.max(cp.abs(Q.T@Q/n-cp.eye(K))))











class Pipeline(rt.Pipeline):
 def compute(self,model,points,backward=True,dtype='f4'):
  t={};self.scalar.records=[]
  with rt.timed(t,'query_geometry'):data,G,h=self.batch(points,dtype)
  with rt.timed(t,'query_analytic_fields'):f=rt.grouped8.network_fields(model,data,'au')
  co=cp.empty((self.scalar.n+10,model.K),dtype='f8');amp=cp.empty((1,model.K),dtype='f8')
  with rt.timed(t,'source_values_and_scalar_projection'):
   for mode in range(0,model.K,8):
    m=block_model(model,mode,mode+8)
    af=source_values(m,self.fit) if dtype=='f4' else rt.grouped8.network_fields(m,self.fit,'a')['a']
    cc=self.solve(af,'forward');pf=af-self.fit_field(cc);co[:,mode:mode+8]=cc; amp[:,mode:mode+8]=self.hf.T@pf/self.hm
    del af,pf,cc,m
  with rt.timed(t,'canonical_Rayleigh_sort_two_pass_GS'):
   p=f['a']-G@co-h@amp;obj=bounded_potential_objective(p,f['u'],data.n)
  del p,f

  if backward:
   for key in ('Q','U','T'):obj.pop(key)
   with rt.timed(t,'analytic_query_parameter_reverse'):grads=rt.grouped8.network_backward(model,data,value=obj['GA'],curl=obj['GU'])
   with rt.timed(t,'scalar_transpose_and_value_parameter_reverse'):
    for mode in range(0,model.K,8):
     ga=obj['GA'][:,mode:mode+8];beta=h.T@ga/self.hm;rhs=-G.T@ga+self.Gfh@beta
     adj=self.solve(rhs,'transpose',True)-self.hf@beta
     for first in range(0,self.fit.n,2048):
      last=min(first+2048,self.fit.n);part=self.fit.part(first,last)
      _,cache=vr.forward_value(model,part,mode,mode+8)
      av=adj[3*first:3*last].T.reshape(8,last-first,3).astype('f4')
      dw,db=vr.backward_value(model,part,mode,mode+8,cache,av)
      for i in range(len(dw)):grads[0][i][mode:mode+8]+=dw[i];grads[1][i][mode:mode+8]+=db[i]
      del cache,av,dw,db
     del adj,ga,beta,rhs
  else:grads=None
  result=dict(obj=obj,co=co,amp=amp,timings=t,solvers=list(self.scalar.records))
  if backward:result['grads']=grads
  return result
