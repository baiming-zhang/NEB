"""CPU physical diagnostics; network objectives use the native CUDA pipelines."""
import scipy.sparse.linalg as sla
from campaign_core import np

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
