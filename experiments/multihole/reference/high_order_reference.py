"""Native-CuPy P2/P3 refinement of the existing vorticity/circulation FEM.

Only the numerical reference space changes. Continuous random samples, physical
boundary conditions and the original IMEX time integrator are preserved.
"""
import gc
import math
import time
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import cg
from .solver_sparse import TopologyNS, SparseFactor


def sparse_solve(factor,rhs):
    """Native CG with true-residual correction; original acceptance is unchanged."""
    if rhs.ndim==2:return cp.column_stack([sparse_solve(factor,rhs[:,j]) for j in range(rhs.shape[1])])
    solution,info=cg(factor.A,rhs,x0=factor.previous,rtol=1e-12,atol=1e-14,maxiter=10000,M=factor.preconditioner)
    assert info==0,info
    target=max(1e-12,1e-10*float(cp.linalg.norm(rhs)))
    for correction in range(4):
        residual=rhs-factor.A@solution;actual=float(cp.linalg.norm(residual))
        if actual<=target:
            factor.previous=solution.copy()
            factor.true_residual_corrections=getattr(factor,'true_residual_corrections',0)+correction
            return solution
        if correction==3:raise RuntimeError(f'True CG residual {actual} exceeds unchanged target {target}')
        change,info=cg(factor.A,residual,rtol=1e-12,atol=1e-16,maxiter=10000,M=factor.preconditioner)
        assert info==0,info
        solution+=change

_advect=cp.RawKernel(r'''
extern "C" __global__ void advect(const int* cells, const double* gl,
 const double* area, const double* N, const double* dN, const double* weight,
 const double* omega,const double* psi,double* result,int nt,int nl,int nq) {
 int t=blockDim.x*blockIdx.x+threadIdx.x;if(t>=nt)return;
 double w[10],p[10],load[10];
 for(int i=0;i<nl;++i){int id=cells[t*nl+i];w[i]=omega[id];p[i]=psi[id];load[i]=0.;}
 for(int q=0;q<nq;++q){
   double wp[3]={0.,0.,0.},pp[3]={0.,0.,0.};
   for(int i=0;i<nl;++i)for(int k=0;k<3;++k){double d=dN[(q*nl+i)*3+k];wp[k]+=d*w[i];pp[k]+=d*p[i];}
   double wx=0.,wy=0.,px=0.,py=0.;
   for(int k=0;k<3;++k){wx+=wp[k]*gl[t*6+k*2];wy+=wp[k]*gl[t*6+k*2+1];px+=pp[k]*gl[t*6+k*2];py+=pp[k]*gl[t*6+k*2+1];}
   double a=(py*wx-px*wy)*weight[q]*area[t];
   for(int i=0;i<nl;++i)load[i]+=N[q*nl+i]*a;
 }
 for(int i=0;i<nl;++i)atomicAdd(result+cells[t*nl+i],load[i]);
}
''','advect',options=('--fmad=false',))


def multiindices(degree):
    p=degree
    assert p in (2,3)
    result=[(p,0,0),(0,p,0),(0,0,p)]
    for a,b in ((0,1),(1,2),(2,0)):
        for k in range(1,p):
            row=[0,0,0];row[a]=p-k;row[b]=k;result.append(tuple(row))
    result.extend((a,b,p-a-b) for a in range(1,p) for b in range(1,p-a) if p-a-b>0)
    return result


def basis(bary,degree):
    """Lagrange basis and exact derivatives with respect to barycentric coords."""
    bary=np.asarray(bary);values=[];derivatives=[]
    for alpha in multiindices(degree):
        factors=[];diff=[]
        for j,a in enumerate(alpha):
            f=np.ones(len(bary));d=np.zeros(len(bary))
            for k in range(a):
                term=(degree*bary[:,j]-k)/(k+1)
                d=d*term+f*degree/(k+1);f=f*term
            factors.append(f);diff.append(d)
        values.append(factors[0]*factors[1]*factors[2])
        derivatives.append(np.stack([diff[j]*factors[(j+1)%3]*factors[(j+2)%3] for j in range(3)],axis=1))
    return np.stack(values,axis=1),np.stack(derivatives,axis=1)


def quadrature(order=5):
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    r,s=np.meshgrid(x,x,indexing='ij');a,b=np.meshgrid(w,w,indexing='ij')
    return np.stack((1-r-(1-r)*s,r,(1-r)*s),axis=-1).reshape(-1,3),(2*a*b*(1-r)).ravel()


def validate_basis():
    checks=[]
    q,w=quadrature(5)
    for p in (2,3):
        nodes=np.asarray(multiindices(p))/p;v,d=basis(q,p);nv,_=basis(nodes,p)
        worst=0.
        for a in range(p+1):
            for b in range(p+1-a):
                f=nodes[:,1]**a*nodes[:,2]**b
                target=q[:,1]**a*q[:,2]**b
                dx=(a*q[:,1]**(a-1)*q[:,2]**b if a else np.zeros(len(q)))
                dy=(b*q[:,1]**a*q[:,2]**(b-1) if b else np.zeros(len(q)))
                worst=max(worst,abs(v@f-target).max(),abs((d[:,:,1]-d[:,:,0])@f-dx).max(),abs((d[:,:,2]-d[:,:,0])@f-dy).max())
        assert worst<1e-12 and abs(nv-np.eye(len(nodes))).max()<1e-12
        checks.append(dict(degree=p,polynomial_value_and_gradient_max_error=float(worst)))
    worst=0.
    for a in range(9):
        for b in range(9-a):
            exact=2*math.factorial(a)*math.factorial(b)/math.factorial(a+b+2)
            worst=max(worst,abs(w@(q[:,1]**a*q[:,2]**b)-exact))
    assert worst<1e-12
    return dict(passed=True,basis=checks,degree8_quadrature_max_error=float(worst))


class Space:
    def __init__(self,mesh,degree,order=5):
        self.mesh=mesh;self.degree=degree
        tri=mesh.tri;nv=len(mesh.xy);nt=len(tri)
        pairs=np.stack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]),axis=1)
        edges,indices=np.unique(np.sort(pairs,axis=2).reshape(-1,2),axis=0,return_inverse=True)
        indices=indices.reshape(-1,3)
        xy=[mesh.xy];local=[tri]
        for k in range(1,degree):xy.append((mesh.xy[edges[:,0]]*(degree-k)+mesh.xy[edges[:,1]]*k)/degree)
        for j in range(3):
            for k in range(1,degree):
                fraction=np.where(pairs[:,j,0]<pairs[:,j,1],k,degree-k)
                local.append((nv+(fraction-1)*len(edges)+indices[:,j])[:,None])
        if degree==3:
            local.append((nv+2*len(edges)+np.arange(nt))[:,None]);xy.append(mesh.xy[tri].mean(axis=1))
        self.xy_cpu=np.vstack(xy);self.cells_cpu=np.hstack(local).astype('i4')
        self.xy=cp.asarray(self.xy_cpu);self.cells=cp.asarray(self.cells_cpu)
        self.corners=cp.asarray(mesh.xy)[cp.asarray(tri)]
        self.n=len(self.xy_cpu);self.nt=nt;self.nl=self.cells.shape[1]
        keys=edges[:,0].astype('i8')*nv+edges[:,1]
        boundary_edges=np.sort(mesh.boundary_edges,axis=1)
        boundary_index=np.searchsorted(keys,boundary_edges[:,0].astype('i8')*nv+boundary_edges[:,1])
        assert np.array_equal(edges[boundary_index],boundary_edges)
        self.boundary_nodes=[]
        for j,nodes in enumerate(mesh.boundary_nodes):
            selected=boundary_index[mesh.boundary_edge_labels==j]
            self.boundary_nodes.append(cp.asarray(np.concatenate([nodes]+[nv+(k-1)*len(edges)+selected for k in range(1,degree)]),dtype='i4'))
        self.boundary_nodes=tuple(self.boundary_nodes)
        p=self.corners;det=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1])
        assert bool(cp.all(det>0));self.area=det/2
        delta=cp.stack((p[:,1]-p[:,2],p[:,2]-p[:,0],p[:,0]-p[:,1]),axis=1)
        self.grad_l=cp.stack((delta[:,:,1],-delta[:,:,0]),axis=2)/det[:,None,None]
        bary,weights=quadrature(order);N,dN=basis(bary,degree)
        self.bary=cp.asarray(bary);self.weights=cp.asarray(weights);self.N=cp.asarray(N);self.dN=cp.asarray(dN)

    def gradients(self,first,last,dN=None):
        return cp.einsum('qik,tkd->tqid',self.dN if dN is None else dN,self.grad_l[first:last])

    def imposed_interpolant(self,source,field):
        values=field.evaluate(self.xy)[:,0]-source.scale*source.constants[0]
        correction=0.
        for j,nodes in enumerate(self.boundary_nodes):
            constant=source.scale*(source.constants[j]-source.constants[0])
            correction=max(correction,float(cp.max(abs(values[nodes]-constant))));values[nodes]=constant
        return values,correction

    def analytic_error(self,psi,field,order=None,cell_errors=False):
        bary,weights=quadrature(order) if order else (cp.asnumpy(self.bary),cp.asnumpy(self.weights))
        _,dN=basis(bary,self.degree);bary=cp.asarray(bary);weights=cp.asarray(weights);dN=cp.asarray(dN)
        errors=cp.empty(self.nt);norms=cp.empty(self.nt)
        for first in range(0,self.nt,512):
            last=min(first+512,self.nt);grad=self.gradients(first,last,dN)
            g=cp.einsum('tqid,ti->tqd',grad,psi[self.cells[first:last]])
            velocity=cp.stack((g[:,:,1],-g[:,:,0]),axis=2)
            points=cp.einsum('qi,tij->tqj',bary,self.corners[first:last])
            target=field.evaluate(points.reshape(-1,2))[:,1:3].reshape(-1,len(weights),2)
            w=self.area[first:last,None]*weights[None,:]
            errors[first:last]=cp.sum(w*cp.sum((velocity-target)**2,axis=2),axis=1)
            norms[first:last]=cp.sum(w*cp.sum(target**2,axis=2),axis=1)
        result=dict(relative_velocity_L2=float(cp.sqrt(errors.sum()/norms.sum())),analytic_energy_twice=float(norms.sum()),quadrature_order=int(round(len(weights)**.5)),domain='full preserved polygon')
        return (result,cp.asnumpy(errors)) if cell_errors else result


class HighOrderNS(TopologyNS):
    def __init__(self,space,viscosity=1/750):
        self.space=space;self.mesh=space.mesh;self.nu=viscosity
        self.xy=space.xy;self.n=space.n;self.boundary_nodes=space.boundary_nodes
        self.harmonic_dimension=len(self.boundary_nodes)-1
        self.D=cp.unique(cp.concatenate(self.boundary_nodes));mask=cp.ones(self.n,dtype=cp.bool_);mask[self.D]=False;self.F=cp.where(mask)[0]
        shape=(space.nt,space.nl,space.nl);local_s=cp.empty(shape);local_m=cp.empty(shape)
        mass=cp.einsum('qi,qj,q->ij',space.N,space.N,space.weights)
        for first in range(0,space.nt,1024):
            last=min(first+1024,space.nt);g=space.gradients(first,last)
            local_s[first:last]=cp.einsum('tqid,tqjd,q,t->tij',g,g,space.weights,space.area[first:last])
            local_m[first:last]=mass[None,:,:]*space.area[first:last,None,None]
        rows=cp.broadcast_to(space.cells[:,:,None],shape).ravel();cols=cp.broadcast_to(space.cells[:,None,:],shape).ravel()
        self.S=coo_matrix((local_s.ravel(),(rows,cols)),shape=(self.n,self.n)).tocsr()
        self.M=coo_matrix((local_m.ravel(),(rows,cols)),shape=(self.n,self.n)).tocsr()
        del rows,cols,local_s,local_m,g;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        self.Sf=self.S[self.F][:,self.F].tocsr();self.Mf=self.M[self.F][:,self.F].tocsr()
        self.LS=SparseFactor(self.Sf);self.LM=SparseFactor(self.Mf)
        self.H=cp.zeros((self.n,self.harmonic_dimension))
        for j in range(self.harmonic_dimension):
            self.H[self.boundary_nodes[j+1],j]=1.
            self.H[self.F,j]=sparse_solve(self.LS,-(self.S@self.H[:,j])[self.F])
        self.period=-cp.stack([cp.sum((self.S@self.H)[nodes],axis=0) for nodes in self.boundary_nodes[1:]])
        self.harmonic_gram=self.H.T@self.S@self.H

    def zero_trace_potential(self,omega):
        psi=cp.zeros(self.n);psi[self.F]=sparse_solve(self.LS,(self.M@omega)[self.F]);return psi

    def projected_initial(self,source,field):
        sp=self.space;load=cp.zeros(self.n)
        for first in range(0,sp.nt,512):
            last=min(first+512,sp.nt);points=cp.einsum('qi,tij->tqj',sp.bary,sp.corners[first:last])
            omega=field.evaluate(points.reshape(-1,2))[:,3].reshape(-1,len(sp.weights))
            local=cp.einsum('tq,qi,q,t->ti',omega,sp.N,sp.weights,sp.area[first:last])
            cp.add.at(load,sp.cells[first:last].ravel(),local.ravel())
        omega=cp.zeros(self.n);omega[self.F]=sparse_solve(self.LM,load[self.F])
        trace=cp.asarray((np.asarray(source.constants[1:])-source.constants[0])*source.scale)
        psi=self.zero_trace_potential(omega)+self.H@trace
        return omega,self.gamma_from_state(psi,omega),psi

    def advection_unfused(self,omega,psi):
        sp=self.space;result=cp.zeros(self.n)
        for first in range(0,sp.nt,2048):
            last=min(first+2048,sp.nt);cells=sp.cells[first:last];g=sp.gradients(first,last)
            gp=cp.einsum('tqid,ti->tqd',g,psi[cells]);gw=cp.einsum('tqid,ti->tqd',g,omega[cells])
            adv=gp[:,:,1]*gw[:,:,0]-gp[:,:,0]*gw[:,:,1]
            local=cp.einsum('tq,qi,q,t->ti',adv,sp.N,sp.weights,sp.area[first:last])
            cp.add.at(result,cells.ravel(),local.ravel())
        return result

    def advection(self,omega,psi):
        sp=self.space;result=cp.zeros(self.n)
        _advect(((sp.nt+127)//128,),(128,),(sp.cells,sp.grad_l,sp.area,sp.N,sp.dN,sp.weights,
            omega,psi,result,np.int32(sp.nt),np.int32(sp.nl),np.int32(len(sp.weights))))
        return result

    def advance(self,omega0,gamma0,dt,final_time,snapshot_dt,label,timeout_seconds=1700.,frame_callback=None):
        # Same original IMEX midpoint/CN equations, with true-residual-corrected
        # native sparse solves. No projection to future reference data.
        steps=round(final_time/dt);assert abs(steps*dt-final_time)<1e-12
        stride=max(1,round(snapshot_dt/dt));implicit=SparseFactor(self.Mf+.5*dt*self.nu*self.Sf)
        omega=omega0.copy();gamma=gamma0.copy();psi=self.reconstruct(omega,gamma)
        frames=[];started=time.perf_counter()
        for step in range(steps+1):
            if step%stride==0 or step==steps:
                frames.append((step*dt,psi.copy(),omega.copy(),gamma.copy()))
                if frame_callback is not None:frame_callback(step,steps,frames[-1])
            if step==steps:break
            if time.perf_counter()-started>timeout_seconds:raise TimeoutError(f'{label} exceeded bounded {timeout_seconds} second budget')
            advection=self.advection(omega,psi)
            half=cp.zeros(self.n);half[self.F]=sparse_solve(implicit,(self.M@omega-.5*dt*advection)[self.F])
            gamma_half=gamma+self.boundary_sums(self.M@(half-omega)+.5*dt*(advection+self.nu*(self.S@half)))
            psi_half=self.reconstruct(half,gamma_half);adv_half=self.advection(half,psi_half)
            following=cp.zeros(self.n)
            following[self.F]=sparse_solve(implicit,(self.M@omega-.5*dt*self.nu*(self.S@omega)-dt*adv_half)[self.F])
            gamma=gamma+self.boundary_sums(self.M@(following-omega)+dt*(adv_half+.5*self.nu*(self.S@(following+omega))))
            omega=following;psi=self.reconstruct(omega,gamma)
            assert bool(cp.all(cp.isfinite(omega))),f'Nonfinite {label}, step {step+1}'
        cp.cuda.Stream.null.synchronize();return frames,time.perf_counter()-started
