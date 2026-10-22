




from _neb_paths import Path
import importlib.util
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'exports/span_constraints/span_penalty.py'
spec=importlib.util.spec_from_file_location('original_span_constraint',SOURCE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
span_augmented=original.span_augmented
span_constraint_functional=original.span_constraint_functional

def span_dual_only(cp,a,d,h,lam,eta,rho,scale):
    assert float(rho)>=0 and float(scale)>=0
    a,d,h,lam,eta,n,k,mass,s,sh,r,moment,b,inverse,info=original._terms(cp,a,d,h,lam,eta)
    linear_div=cp.mean(lam*r,axis=0)
    weak_term=scale*(eta*moment+0.5*rho*moment*moment)
    span_energy=cp.trace(inverse@b)
    cost=linear_div+weak_term+0.5*rho*span_energy/k
    assert bool(cp.all(cp.isfinite(cost))) and bool(cp.all(cp.isfinite(r))) and bool(cp.all(cp.isfinite(moment)))
    info.update(divergence_quadratic_cost=float(0.5*rho*span_energy),
        normalized_raw_multiplier_cost=float(cp.sum(linear_div)),weak_wall_cost=float(cp.sum(weak_term)),total_constraint_cost=float(cp.sum(cost)))
    return r,moment,info
