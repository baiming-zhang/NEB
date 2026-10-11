
from canonical_poisson_residual_fit import *

def raw_lift_jets(points,geo,c,engine=None):
    if engine is None:
        const,mul,power,exp,log,div,index=jconst,jmul,jpow,jexp,jlog,jdiv,INDEX
    else:
        const,mul,power,exp,log,div,index=engine.constant,engine.mul,engine.power,engine.exp,engine.log,engine.div,engine.index
    n=len(points);xyz=[]
    for j in range(3):
        a=const(points[:,j]);a[:,index[tuple(int(k==j) for k in range(3))]]=1;xyz.append(a)
    out=const(cp.ones(n))
    for a in xyz:out-=power((a-const(cp.full(n,.5)))/c.outer_half_extent,c.outer_superellipsoid_power)
    distances=[]
    for center in geo:
        delta=[a-const(cp.full(n,center[j])) for j,a in enumerate(xyz)]
        distances.append(power(sum(mul(d,d) for d in delta),.5)-const(cp.full(n,center[3])))
    shift=cp.min(cp.stack([d[:,0] for d in distances]),axis=0)
    inner=const(shift)-c.metaball_softmin_tau*log(sum(exp(-(d-const(shift))/c.metaball_softmin_tau) for d in distances))
    do=out*(c.outer_half_extent/c.outer_superellipsoid_power)
    denominator=do+inner
    return div(do,denominator),denominator

def lift_fields(x,geo,c):
    q,d=raw_lift_jets(x,geo,c)
    grad=cp.stack([q[:,INDEX[tuple(int(k==j) for k in range(3))]] for j in range(3)],axis=1)
    lap=sum(2*q[:,INDEX[tuple(2*int(k==j) for k in range(3))]] for j in range(3))
    return q[:,0],grad,lap,d[:,0]
