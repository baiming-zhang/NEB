
import json
from types import SimpleNamespace
from core import *
D=HERE.parent/'results/velocity_k1024_full2000_v1'
c=SimpleNamespace(**json.loads((D/'config.json').read_text()));cp.get_default_memory_pool().set_limit(size=4*2**30)
m=Model(c,8,c.seed);geo=geometry_from_token(c,1);pts=fluid_points(c,geo,128,c.seed+100001)
data=FieldData(pts,geo,c,'f8');f,h=forward(m,data,0,8,True)
U=velocity(f).reshape(8,-1).T;A=velocity_grad(f).reshape(8,-1).T
o=coefficient_peel(*grams(U,A));loss,gu,gd,Q,V,order=literal_peel(U,A)
rel=lambda a,b:float(cp.linalg.norm(a-b)/cp.maximum(cp.linalg.norm(b),1e-30))
result=dict(loss_relative=rel(o['losses'],loss),velocity_adjoint_relative=rel(U@o['Cu']/128,gu),gradient_adjoint_relative=rel(A@o['Cd']/128,gd),gram=float(o['gram_max']))
rng=np.random.Generator(np.random.PCG64(712));dw=[cp.asarray(rng.normal(size=w.shape)*.001) for w in m.W];db=[cp.asarray(rng.normal(size=b.shape)*.001) for b in m.b]
t=jvp(m,data,dw,db,0,8);du=velocity(t).reshape(8,-1).T;dd=velocity_grad(t).reshape(8,-1).T
lhs=cp.sum(gu*du)+cp.sum(gd*dd)
GW,Gb=backward(m,data,h,lift(gu.T.reshape(8,128,3),gd.T.reshape(8,128,3,3)))
rhs=sum(cp.sum(a*b) for a,b in zip(GW+Gb,dw+db));result['parameter_adjoint_relative']=float(abs(lhs-rhs)/cp.maximum(abs(lhs),1e-30))

direct=cp.asarray(0.,dtype='f8')
for r,index in enumerate(cp.asnumpy(order)):
    u=U[:,index].copy();a=A[:,index].copy();tu=du[:,index].copy();ta=dd[:,index].copy();q=Q[:,:r];v=V[:,:r]
    for _ in range(2):
        alpha=q.T@u/128;da=q.T@tu/128;u-=q@alpha;a-=v@alpha;tu-=q@da;ta-=v@da
    mass=cp.dot(u,u)/128;stiff=cp.dot(a,a)/128
    direct+=2*cp.dot(a,ta)/(128*mass)-2*stiff*cp.dot(u,tu)/(128*mass*mass)
result['frozen_prefix_tangent_relative']=float(abs(direct-lhs)/cp.maximum(abs(lhs),1e-30))
div=cp.trace(velocity_grad(f),axis1=-2,axis2=-1)
result['analytic_divergence_relative']=float(cp.linalg.norm(div)/cp.linalg.norm(velocity_grad(f)))

sym=cp.asarray(rng.normal(size=(8,128,3,3)));sym=(sym+sym.swapaxes(-1,-2))*.5
gauge=J2(cp.zeros_like(f.value),sym,cp.zeros_like(f.hess))
result['gauge_velocity_max']=float(cp.max(cp.abs(velocity(gauge))))
assert all(v<2e-8 for v in result.values()),result
result['passed']=True;result['test_modes']=8;result['production_modes']=1024;result['no_finite_differences']=True
write(D/'verification.json',result);print(json.dumps(result),flush=True)
