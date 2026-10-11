






import math
import numpy as np


def _backend(array, xp):
    if xp is not None:
        return xp
    if type(array).__module__.split('.')[0] == 'cupy':
        import cupy
        return cupy
    return np


def _host(array, xp):
    return np.asarray(array) if xp is np else xp.asnumpy(array)


def _scalar(value):
    return float(value)


def _inputs(Z, U, M, physical_volume, npoints, xp):
    xp = _backend(Z, xp)
    V = float(physical_volume)
    if not math.isfinite(V) or V <= 0:
        raise ValueError('A finite positive physical volume is required')
    if isinstance(npoints, bool) or int(npoints) != npoints or npoints <= 0:
        raise ValueError('npoints must be the complete positive physical point count')
    npoints = int(npoints)
    Z, U = xp.asarray(Z, dtype='f8'), xp.asarray(U, dtype='f8')
    if Z.ndim != 2 or U.ndim != 2 or Z.shape[1] != U.shape[1]:
        raise ValueError('Z and U must be matrices with the same complete mode count')
    if Z.shape[1] <= 0 or U.shape[0] != 3*npoints or M.shape != (Z.shape[0], Z.shape[0]):
        raise ValueError('Physical metric or complete point/mode dimensions changed')
    if not bool(xp.all(xp.isfinite(Z))) or not bool(xp.all(xp.isfinite(U))):
        raise FloatingPointError('Nonfinite complete projected-potential or curl field')
    return Z, U, V, npoints, xp


def _one_mode(z,u,mz,q,w,mq,npoints,xp,minimum_relative_mass):

    raw_mass = _scalar(z@mz)
    raw_curl = _scalar(u@u/npoints)
    if not math.isfinite(raw_mass) or raw_mass <= 0 or not math.isfinite(raw_curl) or raw_curl <= 0:
        raise FloatingPointError('Zero/nonpositive quotient vector is inadmissible')
    normal = xp.sqrt(z@mz)
    a, v, ma = z/normal, u/normal, mz/normal
    alphas=[]
    for _ in range(2):
        alpha = q.T@ma
        a = a-q@alpha
        v = v-w@alpha
        ma = ma-mq@alpha
        alphas.append(alpha)
    mass = a@ma
    stiff = (v@v)/npoints
    relmass = _scalar(mass)
    if not math.isfinite(relmass) or relmass <= minimum_relative_mass:
        raise FloatingPointError('Canonical quotient lost a full mode after metric GS')
    if not math.isfinite(_scalar(stiff)) or _scalar(stiff) <= 0:
        raise FloatingPointError('Zero/nonpositive empirical curl mode is inadmissible')
    gz = (-2*stiff/(mass*mass))*ma
    gu = (2/(npoints*mass))*v


    for _ in range(2):
        gz = gz-mq@(q.T@gz+w.T@gu)
    normal_bar = gz@z+gu@u
    GZ = gz/normal-mz*normal_bar/(normal**3)
    GU = gu/normal
    return dict(loss=stiff/mass, GZ=GZ, GU=GU, residual_z=a,
                residual_u=v, residual_mz=ma, alphas=alphas,
                relative_mass=mass, normal=normal)


def detached_prefix_loss(z, u, prefix_z, prefix_u, M, physical_volume, npoints,
                         *, xp=None, minimum_relative_mass=1e-12):





    xp = _backend(z, xp)
    z, u = xp.asarray(z, dtype='f8'), xp.asarray(u, dtype='f8')
    q, w = xp.asarray(prefix_z, dtype='f8'), xp.asarray(prefix_u, dtype='f8')
    V, npoints = float(physical_volume), int(npoints)
    return _one_mode(z,u,(M@z)/V,q,w,(M@q)/V,npoints,xp,minimum_relative_mass)


def projected_metric_objective(Z, U, M, physical_volume, npoints, *, xp=None,
                               minimum_relative_mass=1e-12,
                               minimum_normalized_gram_eigenvalue=1e-12,
                               maximum_canonical_gram_error=1e-8):










    Z, U, V, npoints, xp = _inputs(Z,U,M,physical_volume,npoints,xp)
    K = Z.shape[1]
    MZ = (M@Z)/V
    raw_gram = Z.T@MZ
    diagonal = xp.diag(raw_gram)
    curl_diagonal = xp.sum(U*U,axis=0)/npoints
    if not bool(xp.all(xp.isfinite(raw_gram))) or not bool(xp.all(diagonal>0)):
        raise FloatingPointError('Zero/nonpositive canonical lift mass is inadmissible')
    if not bool(xp.all(xp.isfinite(curl_diagonal))) or not bool(xp.all(curl_diagonal>0)):
        raise FloatingPointError('Pure-gradient/zero physical curl vector is inadmissible')
    roots = xp.sqrt(diagonal)
    normalized_raw = raw_gram/roots[:,None]/roots[None,:]
    symmetry = xp.max(xp.abs(normalized_raw-normalized_raw.T))
    if _scalar(symmetry)>maximum_canonical_gram_error:
        raise FloatingPointError('Physical consistent mass is not numerically symmetric')
    eigenvalues = xp.linalg.eigvalsh((normalized_raw+normalized_raw.T)*.5)
    if not bool(xp.all(xp.isfinite(eigenvalues))) or _scalar(eigenvalues[0])<=minimum_normalized_gram_eigenvalue:
        raise FloatingPointError('Complete canonical lift Gram does not retain all requested modes')
    order = xp.argsort(curl_diagonal/diagonal,kind='stable')
    Q = xp.empty_like(Z)
    MQ = xp.empty_like(Z)
    W = xp.empty_like(U)
    T = xp.zeros((K,K),dtype='f8')
    GZ, GU = xp.zeros_like(Z), xp.zeros_like(U)
    losses, masses = xp.empty(K,dtype='f8'), xp.empty(K,dtype='f8')
    for rank,index in enumerate(_host(order,xp)):
        index=int(index)
        z, u = Z[:,index], U[:,index]
        q, w = Q[:,:rank], W[:,:rank]
        one = _one_mode(z,u,MZ[:,index],q,w,MQ[:,:rank],npoints,xp,minimum_relative_mass)
        normal = one['normal']


        t = xp.zeros(K,dtype='f8');t[index]=1/normal
        for alpha in one['alphas']:
            t = t-T[:,:rank]@alpha
        residual_norm = xp.sqrt(one['relative_mass'])
        Q[:,rank]=one['residual_z']/residual_norm
        MQ[:,rank]=one['residual_mz']/residual_norm
        W[:,rank]=one['residual_u']/residual_norm
        T[:,rank]=t/residual_norm
        losses[rank]=one['loss'];masses[rank]=one['relative_mass']
        GZ[:,index]=one['GZ'];GU[:,index]=one['GU']


    actual_MQ=(M@Q)/V
    metric_image_error=xp.linalg.norm(actual_MQ-MQ)/xp.linalg.norm(actual_MQ)
    if _scalar(metric_image_error)>maximum_canonical_gram_error:
        raise FloatingPointError('Same-call metric images lost physical consistency')
    canonical_gram=Q.T@actual_MQ
    gram_error=xp.max(xp.abs(canonical_gram-xp.eye(K,dtype='f8')))
    if _scalar(gram_error)>maximum_canonical_gram_error:
        raise FloatingPointError('Full canonical Gram failed orthonormality')
    if not bool(xp.all(xp.isfinite(GZ))) or not bool(xp.all(xp.isfinite(GU))):
        raise FloatingPointError('Nonfinite complete per-mode quotient adjoint')
    return dict(losses=losses,loss_sum=xp.sum(losses),loss_mean=xp.mean(losses),order=order,T=T,
        Zbasis=Q,Ubasis=W,GZ=GZ,GU=GU,relative_masses=masses,
        raw_mass_gram=raw_gram,canonical_gram=canonical_gram,gram_max=gram_error,
        normalized_raw_gram_eigenvalues=eigenvalues,raw_mass_diagonal=diagonal,
        metric_image_relative=metric_image_error,
        qualified_full_rank=True,all_modes=K,all_physical_points=npoints,
        physical_volume=V,zero_vector_accepted=False,
        gradient_semantics='Each column differentiates only its own sorted-mode loss; normalized earlier prefixes detached')
