




import torch
import scipy.sparse.linalg as sla
from campaign_core import np

def sparse(matrix,device,dtype):
    coo=matrix.tocoo()
    return torch.sparse_coo_tensor(np.array([coo.row,coo.col]),coo.data,coo.shape,device=device,dtype=dtype).coalesce()

class Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,complex_,harmonic):
        ctx.complex=complex_;ctx.harmonic=harmonic
        value=x.detach().cpu().double().numpy();c=complex_
        p=value-c.G@c.poisson.solve(c.G.T@(c.M@value))
        if harmonic:p-=c.H@(c.H.T@(c.M@p))
        return torch.as_tensor(p,device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx,gradient):
        c=ctx.complex;v=gradient.detach().cpu().double().numpy()
        p=v-c.M@(c.G@c.poisson.solve(c.G.T@v))
        if ctx.harmonic:p-=c.M@(c.H@(c.H.T@p))
        return torch.as_tensor(p,device=gradient.device,dtype=gradient.dtype),None,None

class MassInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,factor):
        ctx.factor=factor
        return torch.as_tensor(factor.solve(x.detach().cpu().double().numpy()),device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx,gradient):
        return torch.as_tensor(ctx.factor.solve(gradient.detach().cpu().double().numpy(),trans='T'),
            device=gradient.device,dtype=gradient.dtype),None

class Objective:
    def __init__(self,complex_,method,device='cpu',dtype=torch.float64):
        self.c=complex_;self.method=method
        self.mass=sparse(complex_.M,device,dtype);self.stiffness=sparse(complex_.S,device,dtype)
        self.mass_factor=sla.splu(complex_.M.tocsc())

    def canonicalize(self,raw,final=False):
        if self.method in ('full','lifted_velocity') or (final and self.method=='posthoc'):
            return Project.apply(raw,self.c,True)
        if self.method=='coulomb_only':return Project.apply(raw,self.c,False)
        return raw

    def quotient(self,a):
        sa=torch.mv(self.stiffness,a)
        if self.method=='lifted_velocity':
            return torch.dot(sa,MassInverse.apply(sa,self.mass_factor))/torch.dot(a,sa)
        return torch.dot(a,sa)/torch.dot(a,torch.mv(self.mass,a))

    def orthogonalize(self,raw,previous):
        a=self.canonicalize(raw)
        a=a/torch.sqrt(torch.dot(a,torch.mv(self.mass,a)))
        if previous is not None:
            q=previous.detach()
            for _ in range(2):a=a-q@(q.T@torch.mv(self.mass,a))
        norm=torch.dot(a,torch.mv(self.mass,a))
        if not torch.isfinite(norm) or norm<1e-12:raise FloatingPointError('Mass MGS rank collapse')
        return a/torch.sqrt(norm)

def physical_metrics(c,a,reference=None):

    a=np.asarray(a);ma=c.M@a;sa=c.S@a
    lam=np.sum(a*sa,axis=0)/np.sum(a*ma,axis=0)
    inv=sla.splu(c.M.tocsc());defect=inv.solve(sa)-a*lam
    velocity_norm=np.sqrt(np.sum(a*sa,axis=0))
    residual=np.sqrt(np.maximum(np.sum(defect*(c.S@defect),axis=0),0))/(lam*velocity_norm)
    gram=(a.T@sa)/np.outer(velocity_norm,velocity_norm)
    result=dict(eigenvalues=lam.tolist(),velocity_residuals=residual.tolist(),velocity_residual_mean=float(np.mean(residual)),
        velocity_gram_defect=float(np.linalg.norm(gram-np.eye(len(lam)))),
        gauge_gradient_norm=float(np.linalg.norm(c.G.T@ma)),gauge_harmonic_norm=float(np.linalg.norm(c.H.T@ma)),
        measured_wall_normal_residual=None,measured_wall_tangential_vorticity_residual=None,
        wall_reason='Independent fitted-boundary quadrature diagnostics must be evaluated by the geometry-specific evaluator')
    if reference is not None:
        rlam,ra=reference
        qa=a@np.linalg.inv(np.linalg.cholesky(a.T@sa)).T
        qr=ra@np.linalg.inv(np.linalg.cholesky(ra.T@(c.S@ra))).T
        overlap=qr.T@(c.S@qa)
        result['velocity_subspace_projector_frobenius']=float(np.sqrt(max(0,2*a.shape[1]-2*np.linalg.norm(overlap)**2)))
        result['eigenvalue_relative_errors']=(abs(lam-rlam)/rlam).tolist()
    return result
