




import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import time
import subprocess
import traceback
from native_cupy_core import *
from cupyx import errstate

DEST = ROOT/'results/native_gauge_rbf_screen_v1'
PRIOR = ROOT/'results/native_gauge_residual_preflight_v1'
H = .15
NC = 2048
SEED = 9172048
CHUNK = 512

def sync():
    cp.cuda.Stream.null.synchronize()

def describe(x):
    return dict(min=float(cp.min(x)), median=float(cp.median(x)), max=float(cp.max(x)))

def radial(points, centers, e, de, le):

    delta = points[:, None, :] - centers[None, :, :]
    z = cp.sum(delta*delta, axis=2)/H**2
    active = z < 1
    t = cp.where(active, 1-z, 1.)
    b = cp.where(active, cp.exp(-1/t), 0.)
    bp = -b/t**2
    bpp = b/t**4 - 2*b/t**3
    lap = e[:, None]*(4*z*bpp + 6*bp)/H**2
    lap += 4*bp*cp.sum(de[:, None, :]*delta, axis=2)/H**2
    lap += le[:, None]*b
    return lap, active.sum(1), cp.sqrt(cp.min(z, axis=1))*H

def spectrum(r, u):

    mass = u.T@u
    with errstate(linalg='raise'):
        chol = cp.linalg.cholesky(mass)
        w = solve_triangular(chol, (r.T@r), lower=True)
        w = solve_triangular(chol, w.T, lower=True).T
        ev = cp.linalg.eigvalsh((w+w.T)/2)
    return dict(min=float(cp.sqrt(cp.maximum(ev[0], 0))),
                rms=float(cp.sqrt(cp.maximum(cp.mean(ev), 0))),
                max=float(cp.sqrt(cp.maximum(ev[-1], 0))),
                raw_channel_ratios=cp.asnumpy(cp.linalg.norm(r, axis=0)/cp.linalg.norm(u, axis=0)).tolist())

def load_fields(name):
    with np.load(PRIOR/(name+'_fields.npz')) as data:
        return {k: cp.asarray(data[k]) for k in ('points','U','div','G0','L0','GR')}

def assemble(data, centers, c, geo):
    n = len(data['points']); matrix = cp.empty((n, NC+126), dtype='f8')
    matrix[:, :126] = data['L0']
    counts=[]; distances=[]; env=[]
    for start in range(0, n, CHUNK):
        stop=min(n,start+CHUNK); x=data['points'][start:stop].astype('f8')
        e = envelope1(x, geo, c)[0].value[:, 0]
        de = data['G0'][3*start:3*stop, 0].reshape(-1,3)
        block,count,distance=radial(x,centers,e,de,data['L0'][start:stop,0])
        matrix[start:stop,126:]=block
        counts.append(count);distances.append(distance);env.append(e)
    return matrix, cp.concatenate(env), dict(active_centers=describe(cp.concatenate(counts)),
        uncovered_points=int(cp.sum(cp.concatenate(counts)==0)),
        nearest_center_distance=describe(cp.concatenate(distances)))

def analytic_check(c,geo,centers):

    from native_gauge_residual_preflight import Taylor4
    t=Taylor4(); center=centers[0]
    offsets=cp.asarray([[.01,.02,-.03],[.04,-.01,.02],[.08,.04,.01],[.14,0,0],[.151,0,0]])
    x=center[None,:]+offsets
    xyz,e,chi=t.geometry(x,geo,c)
    z=sum(t.mul(v-t.constant(cp.full(len(x),center[j])),v-t.constant(cp.full(len(x),center[j]))) for j,v in enumerate(xyz))/H**2
    active=z[:,0]<1
    safe=z.copy();safe[~active]=0
    b=t.exp(-t.power(t.constant(cp.ones(len(x)))-safe,-1));b[~active]=0
    exact=t.laplacian(t.mul(e,b))
    actual=radial(x,center[None,:],e[:,0],t.gradient(e),t.laplacian(e))[0][:,0]
    error=float(cp.max(cp.abs(actual-exact))/cp.maximum(1.,cp.max(cp.abs(exact))))
    return dict(relative_laplacian_error=error, passed=error<1e-11,
                method='independent normalized Taylor4 product algebra; includes inside and outside support, no finite differences')
