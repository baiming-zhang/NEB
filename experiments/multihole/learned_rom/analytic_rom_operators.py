"""Native-CuPy full weak Hodge ROM operators from analytic streamfunction jets."""
import cupy as cp


def moments(grad, hess, weights):
    """grad[n,k,2], compact hess[n,k,(xx,xy,yy)]; positive physical weights."""
    g=cp.asarray(grad,dtype=cp.float64);h=cp.asarray(hess,dtype=cp.float64)
    w=cp.asarray(weights,dtype=cp.float64)
    assert g.ndim==3 and g.shape[2]==2 and h.shape==(*g.shape[:2],3)
    assert w.shape==(len(g),) and bool(cp.all(w>0).item())
    omega=-(h[:,:,0]+h[:,:,2])
    velocity=cp.stack((g[:,:,1],-g[:,:,0]),axis=2)
    velocity_gradient=cp.stack((cp.stack((h[:,:,1],h[:,:,2]),axis=2),
                               cp.stack((-h[:,:,0],-h[:,:,1]),axis=2)),axis=2)
    G=cp.einsum('nia,nja,n->ij',velocity,velocity,w,optimize=True)
    D=omega.T@(w[:,None]*omega)
    wedge=g[:,:,0,None]*g[:,None,:,1]-g[:,:,1,None]*g[:,None,:,0]
    Q=(wedge.reshape(len(g),-1).T@(w[:,None]*omega)).reshape(g.shape[1],g.shape[1],g.shape[1])
    direct=-cp.einsum('nia,njb,nkab,n->ijk',velocity,velocity,velocity_gradient,w,optimize=True)
    return G,D,Q,direct
