



import os, time, traceback, threading
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from canonical_poisson_native import *
from native_gauge_residual_preflight import Taylor4
from native_gauge_rbf_screen import spectrum
from native_cupy_verify import jvp

D = ROOT/'results/canonical_poisson_component_v1/attempt_001'
START = time.perf_counter()


def shift(x, geo, c):

    t = Taylor4(); xyz, e, chi = t.geometry(x, geo, c)
    cos = lambda a: t.sin(a+t.constant(cp.full(len(x), cp.pi/2)))
    f = t.mul(t.mul(e, e), t.mul(cos(5*cp.pi*xyz[0]),
              t.mul(t.sin(3*cp.pi*xyz[1]), cos(4*cp.pi*xyz[2]))))
    return t.gradient(f).reshape(-1, 1), t.laplacian(f)[:, None]


def raw_fields(model, x, geo):
    n = len(x); A = cp.empty((3*n, 32)); U = cp.empty_like(A); div = cp.empty((n, 32))
    for first in range(0, n, 256):
        xx = x[first:first+256]
        data = SimpleNamespace(features=features1(xx, geo, model.c), envelope=envelope1(xx, geo, model.c)[0])
        for mode in range(0, 32, 4):
            f, _ = model.forward(data, mode, mode+4)
            A[3*first:3*(first+len(xx)), mode:mode+4] = f.value.reshape(4, -1).T
            U[3*first:3*(first+len(xx)), mode:mode+4] = curl1(f).reshape(4, -1).T
            div[first:first+len(xx), mode:mode+4] = sum(f.grad[..., j, j] for j in range(3)).T
    return A, U, div


def kernel_check(s):

    t = Taylor4(); center = s.centers[s.n//2]
    x = center+cp.asarray([[.11,.04,-.08],[.41,.13,.19],[.8,.1,.03]])*s.radius
    xyz, e, ch = t.geometry(x, s.geo, s.c)
    r = t.power(sum(t.mul(a-t.constant(cp.full(len(x),center[j])), a-t.constant(cp.full(len(x),center[j]))) for j,a in enumerate(xyz)), .5)/s.radius
    one = t.constant(cp.ones(len(x)))
    k = t.mul(t.power(one-r, 8), 32*t.power(r,3)+25*t.mul(r,r)+8*r+one)
    phi = t.mul(e,k)/s.center_e[s.n//2]
    analytic_lap = t.laplacian(phi)
    analytic_grad = t.gradient(phi)

    lp = s.operator(x,'lap')[:, s.n//2].toarray().reshape(-1)
    gp = s.operator(x,'grad')[:, s.n//2].toarray().reshape(-1,3)
    return dict(laplacian_relative=float(cp.linalg.norm(lp-analytic_lap)/cp.linalg.norm(analytic_lap)),
                gradient_relative=float(cp.linalg.norm(gp-analytic_grad)/cp.linalg.norm(analytic_grad)),
                curl_of_gradient='analytic symmetric scalar Hessian; no finite differences',
                boundary='phi=e(x)*sum c_j k_j(x)/e(center_j): exact zero on both implicit walls')


def parameter_pairing(model, scalar, proj, points, ga, gd):


    rng = cp.random.RandomState(20260909)
    total_jvp = cp.asarray(0.); total_vjp = cp.asarray(0.)
    for mode in range(0,32,4):
        dw = [rng.normal(size=w[mode:mode+4].shape)*1e-3 for w in model.W]
        db = [rng.normal(size=b[mode:mode+4].shape)*1e-3 for b in model.b]
        for x, adj, kind in ((points,ga,'value'),(scalar.centers,gd,'div')):
            for start in range(0,len(x),256):
                xx = x[start:start+256]; n = len(xx)
                data = SimpleNamespace(features=features1(xx,scalar.geo,model.c),envelope=envelope1(xx,scalar.geo,model.c)[0])
                _, cache = model.forward(data,mode,mode+4,True)
                tang = jvp(model,data,dw,db,mode,mode+4)
                av = cp.zeros((4,n,3)); ag = cp.zeros((4,n,3,3))
                if kind == 'value':
                    av = adj[3*start:3*(start+n),mode:mode+4].T.reshape(4,n,3)
                else:
                    for j in range(3): ag[...,j,j] = adj[start:start+n,mode:mode+4].T
                total_jvp += cp.sum(tang.value*av)+cp.sum(tang.grad*ag)
                gw, gb = model.backward(data,mode,mode+4,cache,Jet1(av,ag))
                total_vjp += sum(cp.sum(a*b) for a,b in zip(gw+gb,dw+db))
    return dict(jvp=float(total_jvp),vjp=float(total_vjp),relative=float(abs(total_jvp-total_vjp)/cp.maximum(abs(total_jvp)+abs(total_vjp),1e-30)))


def run_level(resolution, model, geo, c, saved):
    dd = D/f'resolution_{resolution}'; dd.mkdir(parents=True,exist_ok=True)
    t0 = time.perf_counter(); scalar = None
    report = dict(resolution=resolution, status='started')
    write(dd/'report.json',report)
    try:
        scalar = ScalarPoisson(c,geo,resolution,budget_seconds=540.)
        report.update(centers=scalar.n,nonzeros=int(scalar.L.nnz),kernel_check=kernel_check(scalar))
        write(dd/'report.json',report)
        print(json.dumps(dict(stage='assembled',resolution=resolution,centers=scalar.n,seconds=time.perf_counter()-t0)),flush=True)
        x = cp.asarray(saved['points'],dtype='f8')
        proj = CanonicalProjection(scalar,x)

        rng = cp.random.RandomState(82912)
        da = rng.normal(size=(3*len(x),32)); ddv = rng.normal(size=(scalar.n,32)); y = rng.normal(size=da.shape)
        py, _, _ = proj.forward(da,ddv); ga,gd = proj.backward(y)
        lhs = cp.sum(y*py); rhs = cp.sum(ga*da)+cp.sum(gd*ddv)
        report['mixed_point_adjoint_relative'] = float(abs(lhs-rhs)/cp.maximum(abs(lhs)+abs(rhs),1e-30))
        report['harmonic_divergence_over_field'] = float(cp.linalg.norm(proj.hdiv)/cp.linalg.norm(proj.h))
        report['harmonic_removal_relative'] = float(cp.linalg.norm(proj.remove_harmonic(proj.h))/cp.linalg.norm(proj.h))
        write(dd/'report.json',report)
        ac,uc,divc = raw_fields(model,scalar.centers,geo)
        a,u,div = (cp.asarray(saved[k]) for k in ('A','U','div'))
        canonical,coeff,amp = proj.forward(a,divc)
        residual = div-proj.L_eval@coeff-proj.hdiv@amp
        report['independent_coulomb'] = spectrum(residual,u)

        _,lc = shift(scalar.centers,geo,c)
        sh = cp.asarray(saved['shift']); sl = cp.asarray(saved['shiftlap'])
        corrected_shift,shift_coeff,shift_amp = proj.forward(sh,lc)
        scale = cp.linalg.norm(a,axis=0)/cp.linalg.norm(sh)
        sign = cp.where((canonical.T@corrected_shift)[:,0]>=0,1.,-1.)
        moved = canonical+corrected_shift*(scale*sign)[None,:]
        rq = cp.sum(u*u,axis=0)/cp.sum(canonical*canonical,axis=0)
        rq_shift = cp.sum(u*u,axis=0)/cp.sum(moved*moved,axis=0)
        report['gradient_shift'] = dict(remaining_fraction=float(cp.linalg.norm(corrected_shift)/cp.linalg.norm(sh)),
            max_loss_relative_change=float(cp.max(cp.abs(rq_shift/rq-1))),
            all32_loss_relative_changes=cp.asnumpy(rq_shift/rq-1).tolist(),
            independent_shift_poisson_residual=float(cp.linalg.norm(sl-proj.L_eval@shift_coeff)/cp.linalg.norm(sl)))

        report['parameter_adjoint'] = parameter_pairing(model,scalar,proj,x,ga,gd)
        report['canonical_mass_rank'] = int(cp.linalg.matrix_rank(canonical.T@canonical))
        report['pde_and_shift_passed'] = report['independent_coulomb']['max'] <= .02 and report['gradient_shift']['max_loss_relative_change'] <= .01
        report['status'] = 'component_evaluated'
        np.savez(dd/'arrays.npz',points=cp.asnumpy(x),centers=cp.asnumpy(scalar.centers),
                 canonical=cp.asnumpy(canonical),divergence=cp.asnumpy(residual),coefficients=cp.asnumpy(coeff),
                 harmonic=cp.asnumpy(proj.h),harmonic_divergence=cp.asnumpy(proj.hdiv),
                 corrected_shift=cp.asnumpy(corrected_shift),rq=cp.asnumpy(rq),rq_shift=cp.asnumpy(rq_shift))
    except Exception as exc:
        report.update(status='component_failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        report.update(seconds=time.perf_counter()-t0,solves=scalar.solve_records if scalar is not None else [],
                      gpu_pool_reserved=cp.get_default_memory_pool().total_bytes())
        write(dd/'report.json',report)
        print(json.dumps(report),flush=True)
    return report
