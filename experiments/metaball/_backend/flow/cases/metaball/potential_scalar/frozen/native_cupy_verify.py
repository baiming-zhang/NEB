
import time
import numpy as np
import cupy as cp
from native_cupy_core import *

def jvp(model,data,dW,db,start=0,stop=2):
    x=data.features;dx=Jet1(cp.zeros_like(x.value),cp.zeros_like(x.grad))
    for j in range(len(model.W)):
        y=linear1(x,model.W[j][start:stop],model.b[j][start:stop])
        z=linear1(dx,model.W[j][start:stop],cp.zeros_like(db[j]))
        d=linear1(x,dW[j],db[j]);dy=Jet1(z.value+d.value,z.grad+d.grad)
        if j<len(model.W)-1:
            omega=model.c.siren_first_omega0 if j==0 else model.c.siren_hidden_omega0
            y=scale1(y,omega);dy=scale1(dy,omega);s=cp.sin(y.value);c=cp.cos(y.value)
            dx=Jet1(c*dy.value,c[...,None]*dy.grad-s[...,None]*dy.value[...,None]*y.grad)
            x=old.sine_forward_jet1(y)
        else:x=y;dx=dy
    return product1(data.envelope,dx)

def heldout_gradient(data):
    e=data.envelope;phase=cp.pi*data.x.astype('f8');s=cp.sin(phase);c=cp.cos(phase)
    value=s.prod(1);grad=cp.stack([cp.pi*c[:,j]*s[:,(j+1)%3]*s[:,(j+2)%3] for j in range(3)],1)
    return (e.grad[:,0].astype('f8')*value[:,None]+e.value[:,0,None].astype('f8')*grad).reshape(-1,1)

def gauge_trial(model,geometry,order):
    start=time.perf_counter();fit=Data(model.c,geometry,6144,111000,order);val=Data(model.c,geometry,4096,112000,order)
    f,fc=model.forward(fit,0,1,True);v,vc=model.forward(val,0,1,True)
    af=f.value.reshape(-1,1).astype('f8');av=v.value.reshape(-1,1).astype('f8');u=curl1(v).reshape(-1,1).astype('f8')
    Bval=val.G/fit.scales
    def result(sf,sv):
        coeff=fit.solve(fit.B.T@(af+sf));a=av+sv-Bval@coeff;m=cp.sum(a*a);s=cp.sum(u*u)
        ga=-2*s*a/m**2;gu=2*u/m
        ga_fit=-fit.B@fit.solve(Bval.T@ga)
        wf,bf=model.backward(fit,0,1,fc,Jet1(ga_fit.reshape(1,fit.n,3).astype('f4'),cp.zeros((1,fit.n,3,3),dtype='f4')))
        wv,bv=model.backward(val,0,1,vc,Jet1(ga.reshape(1,val.n,3).astype('f4'),curl_adjoint(gu.reshape(1,val.n,3).astype('f4'))))
        grad=cp.concatenate([(x+y).reshape(-1) for x,y in zip(wf+bf,wv+bv)])
        return a,s/m,grad
    base,loss,gradient=result(0,0);rows=[]
    for kind,sf,sv in [('in_span',fit.G[:,0:1],val.G[:,0:1]),('held_out_smooth',heldout_gradient(fit),heldout_gradient(val))]:
        scale=cp.linalg.norm(af)/cp.linalg.norm(sf);a,l,g=result(scale*sf,scale*sv)
        rows.append(dict(order=order,kind=kind,independent_canonical_relative=float(cp.linalg.norm(a-base)/cp.linalg.norm(base)),
            loss_relative_change=float(abs(l/loss-1)),parameter_gradient_relative_change=float(cp.linalg.norm(g-gradient)/cp.linalg.norm(gradient))))
    return dict(rows=rows,seconds=time.perf_counter()-start,gram_condition=float(cp.linalg.cond(fit.gram)))

def vector_and_production_gauge(settings):
    c=config_namespace();model=Model(c,32,settings['seed']);data=Data(c,settings['geometry'],256,113027,5)
    x=cp.asarray(data.points,dtype='f8');components=[]
    for axis,k in [(1,2.),(2,3.),(0,5.)]:
        phase=jconst(k*x[:,axis]);phase[:,INDEX[tuple(int(j==axis) for j in range(3))]]=k
        components.append(jsin(phase))
    A=cp.stack(components,axis=1)[None]
    exactU=cp.stack((-3*cp.cos(3*x[:,2]),-5*cp.cos(5*x[:,0]),-2*cp.cos(2*x[:,1])),axis=1)
    exactW=cp.stack((4*cp.sin(2*x[:,1]),9*cp.sin(3*x[:,2]),25*cp.sin(5*x[:,0])),axis=1)
    exactLap=cp.stack((27*cp.cos(3*x[:,2]),125*cp.cos(5*x[:,0]),8*cp.cos(2*x[:,1])),axis=1)
    U=jet_velocity(A)[0];W=jet_vorticity(A)[0]
    Lap=sum(jet_velocity(A,tuple(2*int(k==j) for k in range(3)))[0] for j in range(3))
    div=sum(jet_velocity(A,tuple(int(k==j) for k in range(3)))[0,:,j] for j in range(3))
    manufactured=max(float(cp.max(cp.abs(U-exactU))),float(cp.max(cp.abs(W-exactW))),
                     float(cp.max(cp.abs(Lap-exactLap))),float(cp.max(cp.abs(div))))
    f,_=model.forward(data,0,2);raw=f.value.reshape(2,-1).T
    _,coeff=data.project(raw,True)
    scalars=scalar_gauge_jets(data.points,settings['geometry'],c,5)
    def gradient_field(co):
        potential=cp.einsum('nsc,sb->bnc',scalars,co)
        value=cp.stack([potential[...,INDEX[tuple(int(k==j) for k in range(3))]] for j in range(3)],axis=-1)
        gradient=cp.empty((*value.shape,3),dtype='f8')
        for i in range(3):
            for j in range(3):
                mi=tuple(int(k==i)+int(k==j) for k in range(3))
                gradient[...,i,j]=potential[...,INDEX[mi]]*(2 if i==j else 1)
        return Jet1(value,gradient)
    gauge=gradient_field(coeff)
    corrected=Jet1(f.value.astype('f8')-gauge.value,f.grad.astype('f8')-gauge.grad)
    original=curl1(f).astype('f8');lift=curl1(corrected)
    preservation=float(cp.linalg.norm(lift-original)/cp.linalg.norm(original))
    rng=cp.random.RandomState(847)
    dW=[rng.normal(size=w[:2].shape).astype('f4')*.01 for w in model.W]
    db=[rng.normal(size=b[:2].shape).astype('f4')*.01 for b in model.b]
    tangent=jvp(model,data,dW,db);_,dc=data.project(tangent.value.reshape(2,-1).T,True)
    dg=gradient_field(dc);dt=Jet1(tangent.value.astype('f8')-dg.value,tangent.grad.astype('f8')-dg.grad)
    t0=curl1(tangent).astype('f8');t1=curl1(dt)
    directional=float(cp.linalg.norm(t1-t0)/cp.linalg.norm(t0))
    result=dict(manufactured_vector_curl_vorticity_laplacian_divergence_max=manufactured,
        actual_production_gauge_curl_relative=preservation,actual_parameter_directional_curl_relative=directional,
        nonzero_gauge_correction_relative=float(cp.linalg.norm(gauge.value)/cp.linalg.norm(f.value)),
        passed=manufactured<1e-11 and preservation<1e-6 and directional<1e-6,
        previous_curl_gradient_zero='Algebraic symmetry only; identical coefficient subtraction did not test production curl')
    write(OUT/'vector_and_production_gauge.json',result);print(json.dumps(result),flush=True)
    if not result['passed']:raise AssertionError('Nontrivial vector/gauge check failed')
    return result

def causal_and_rom_check(settings):
    from native_cupy_evaluate import ROM,raw_first
    model=Model(config_namespace(),32,settings['seed']);data=Data(model.c,settings['geometry'],512,114927,5)
    A,U=raw_first(model,data);p=peel(A,U,data,True)
    rng=cp.random.RandomState(157);dA=rng.normal(size=A.shape);dU=rng.normal(size=U.shape)
    P=data.project(A);dP=data.project(dA);direction=cp.asarray(0.,dtype='f8')
    for rank,index in enumerate(cp.asnumpy(p['order'])):
        a=P[:,index].copy();u=U[:,index].copy();da=dP[:,index].copy();du=dU[:,index].copy()
        norm=cp.sqrt(cp.dot(a,a)/data.n);dn=cp.dot(a,da)/(data.n*norm)
        da=da/norm-a*dn/norm**2;du=du/norm-u*dn/norm**2;a=a/norm;u=u/norm
        q=p['Q'][:,:rank];v=p['U'][:,:rank]
        for _ in range(2):
            alpha=q.T@a/data.n;dalpha=q.T@da/data.n
            a-=q@alpha;u-=v@alpha;da-=q@dalpha;du-=v@dalpha
        m=cp.dot(a,a);st=cp.dot(u,u)
        direction+=2*cp.dot(u,du)/m-2*st*cp.dot(a,da)/m**2
    reverse=cp.sum(p['GA']*dA)+cp.sum(p['GU']*dU)
    relative=float(abs(direction-reverse)/cp.maximum(abs(direction)+abs(reverse),1e-30))


    u=U.reshape(data.n,3,32);w=rng.normal(size=u.shape)
    rom=ROM(u,w,settings['viscosity']);q=rng.normal(size=32);f,J=rom.nonlinear(q,True)
    energy=float(abs(cp.dot(q,f))/cp.maximum(cp.linalg.norm(q)*cp.linalg.norm(f),1e-30))
    v=rng.normal(size=32);vel=(rom.U@q).reshape(-1,3);vor=(rom.W@q).reshape(-1,3)
    dv=(rom.U@v).reshape(-1,3);dw=(rom.W@v).reshape(-1,3)
    direct=rom.U.T@(cp.cross(dv,vor)+cp.cross(vel,dw)).reshape(-1)/data.n
    jac=float(cp.linalg.norm(J@v-direct)/cp.linalg.norm(direct))
    result=dict(K=32,optimizer_steps=0,causal_normalize_two_pass_MGS_directional_adjoint=relative,
        full_mass_ROM_energy_tangency=energy,analytic_nonlinear_Jacobian_action=jac,
        passed=relative<1e-8 and energy<1e-10 and jac<1e-10)
    write(OUT/'causal_and_ROM_algebra.json',result);print(json.dumps(result),flush=True)
    if not result['passed']:raise AssertionError('Causal/ROM algebra failed')
    return result

def verify(settings):
    c=config_namespace();model=Model(c,32,settings['seed']);data=Data(c,settings['geometry'],256,111123,5)
    rng=cp.random.RandomState(717)
    dW=[rng.normal(size=w[:2].shape).astype('f4')*.01 for w in model.W]
    db=[rng.normal(size=b[:2].shape).astype('f4')*.01 for b in model.b]
    f,cache=model.forward(data,0,2,True);adj=Jet1(rng.normal(size=f.value.shape).astype('f4'),rng.normal(size=f.grad.shape).astype('f4'))
    dw,dbrev=model.backward(data,0,2,cache,adj);tangent=jvp(model,data,dW,db)
    forward=cp.sum(tangent.value.astype('f8')*adj.value)+cp.sum(tangent.grad.astype('f8')*adj.grad)
    reverse=sum(cp.sum(a.astype('f8')*b) for a,b in zip(dw+dbrev,dW+db))
    parameter_adjoint=float(abs(forward-reverse)/cp.maximum(abs(forward)+abs(reverse),1e-30))

    higher=higher_fields(model,data.points[:12],settings['geometry'],0,2)
    short=Data(c,settings['geometry'],12,0,order=2,points=data.points[:12])
    first,_=model.forward(short,0,2)
    g=cp.stack([higher[...,INDEX[tuple(int(k==j) for k in range(3))]] for j in range(3)],axis=-1)
    first_error=float(cp.linalg.norm(g-first.grad)/cp.linalg.norm(g))
    value_error=float(cp.linalg.norm(higher[...,0]-first.value)/cp.linalg.norm(higher[...,0]))

    x=cp.asarray(data.points[:12],dtype='f8');wave=cp.asarray([1.2,-.7,2.1]);phase=jconst(x@wave)
    for j in range(3):phase[:,INDEX[tuple(int(k==j) for k in range(3))]]=wave[j]
    sin=jsin(phase);exact=[]
    for m in MULTI:exact.append(cp.sin(x@wave+sum(m)*cp.pi/2)*cp.prod(wave**cp.asarray(m))/math.prod(math.factorial(i) for i in m))
    manufactured=float(cp.max(cp.abs(sin-cp.stack(exact,axis=1))))
    a=rng.normal(size=(3*data.n,2));b=rng.normal(size=a.shape)
    pa=data.project(a);pb=data.project(b)
    gauge_adjoint=float(abs(cp.sum(pa*b)-cp.sum(a*pb))/cp.maximum(cp.linalg.norm(a)*cp.linalg.norm(b),1e-30))


    xyz,e,chi=jet_geometry(x,cp.asarray(settings['geometry']),c)
    def gradient_curl(s):
        return max(float(cp.max(cp.abs(s[:,INDEX[tuple(int(k==a)+int(k==b) for k in range(3))]]-
            s[:,INDEX[tuple(int(k==b)+int(k==a) for k in range(3))]]))) for a in range(3) for b in range(3))
    curl_gradient=max(gradient_curl(e),gradient_curl(chi))
    trials=[gauge_trial(model,settings['geometry'],order) for order in (5,7)]
    checks=dict(parameter_directional_adjoint=parameter_adjoint<2e-5,jet1_jet3_first=first_error<2e-5,
        jet1_jet3_value=value_error<2e-5,manufactured_jet3=manufactured<1e-12,
        gauge_adjoint=gauge_adjoint<1e-10,curl_gradient=curl_gradient<1e-12)
    chosen=trials[0]['rows'];checks['heldout_loss']=chosen[1]['loss_relative_change']<.01
    checks['heldout_parameter_gradient']=chosen[1]['parameter_gradient_relative_change']<.02
    checks['in_span']=chosen[0]['independent_canonical_relative']<1e-8
    report=dict(passed=all(checks.values()),checks=checks,parameter_directional_adjoint=parameter_adjoint,
        jet1_jet3_first=first_error,jet1_jet3_value=value_error,manufactured_jet3=manufactured,
        gauge_adjoint=gauge_adjoint,curl_gradient=curl_gradient,gauge_capacity=trials,
        backend='native CuPy; no finite differences, automatic differentiation, or optimizer updates',
        torch_imported='torch' in sys.modules)
    write(OUT/'analytic_verification.json',report);print(json.dumps(report),flush=True)
    if not report['passed'] or report['torch_imported']:raise AssertionError('Native analytic verification failed')
    return report
