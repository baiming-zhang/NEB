






import os, gc, time, csv, shutil
from original_potential_k32_core import *
from native_cupy_evaluate import eigh, save_npz, saved_initial, walls, analytic_arrays
import velocity_trace_jet2 as jet2
from piola_reference_fields import Reference32
from piola_radial_geometry import RadialMap

REFERENCE=ROOT/'results/fitted_metaball_reference_v2/h0.025_surface80_K1024/modes.npz'
PIOLA=ROOT/'results/piola_geometry_v1/attempt_001'
VOLUME=.7587013244629
K=32

def release():
    gc.collect();cp.get_default_memory_pool().free_all_blocks()

def progress(d,phase,**kw):
    import psutil
    row=dict(phase=phase,process=os.getpid(),updated_unix=time.time(),run=str(d),epoch=2000,
             new_K2048_allowed=False,host_bytes=psutil.Process().memory_info().rss,**kw)
    write(d/'progress.json',row);write(ROOT/'status.json',row)
    print(json.dumps(row),flush=True)
    if row['host_bytes']>8*2**30:raise MemoryError('Evaluation host usage exceeds 8GiB')

def neural_fields(model,points,geo,T=None):
    n=len(points);u=cp.empty((n,3,K));w=cp.empty_like(u)
    for i in range(0,n,512):
        data=jet2.FieldData(points[i:i+512],geo,model.c,'f8')
        a,b=jet2.fields(model,[data])
        u[i:i+data.n]=a.reshape(data.n,3,K);w[i:i+data.n]=b.reshape(data.n,3,K)
    return (u,w) if T is None else (u@T,w@T)

def ref_fields(ref,mapping,points):
    us=[];ps=[];errors=[]
    for i in range(0,len(points),2048):
        x=cp.asarray(points[i:i+2048],dtype='f8');source,J,det,check=mapping.map(x,inverse=True)
        u,ids=ref.velocity(source)
        if bool(cp.any(ids<0)):raise RuntimeError('Reference map has unsupported points; no extension permitted')
        p=ref.potential(source,ids)/cp.sqrt(ref.lam)[None,None,:]
        uu=cp.einsum('nij,njk->nik',J,u)/det[:,None,None]
        pp=cp.einsum('nij,njk->nik',cp.linalg.inv(J).transpose(0,2,1),p)
        us.append(uu);ps.append(pp)
        errors.append(float(cp.linalg.norm(ref.lifted_curl_check(ids,J)-uu)/cp.linalg.norm(uu)))
    return cp.concatenate(us),cp.concatenate(ps),max(errors)

def tensor(u,w):

    n=len(u);C=cp.zeros((K,K,K))
    for a in range(3):
        b=(a+1)%3;c=(a+2)%3
        for k in range(K):
            C[:,:,k]+=(u[:,a,:].T@(u[:,b,:]*w[:,c,k,None])-u[:,a,:].T@(u[:,c,:]*w[:,b,k,None]))/n
    defect=float(cp.linalg.norm(C+C.transpose(1,0,2))/cp.maximum(cp.linalg.norm(C),1e-30))
    return .5*(C-C.transpose(1,0,2)),defect

class FullROM:
    def __init__(self,u,w,nu=.00025):
        n=len(u);uf=u.reshape(-1,K);wf=w.reshape(-1,K)
        self.M=uf.T@uf/n;self.S=wf.T@wf/n
        self.lam,self.V=eigh(self.S,self.M)
        if not bool(cp.all(cp.isfinite(self.lam))) or float(self.lam[0])<=0:
            raise FloatingPointError('Full physical operator is not positive definite')
        self.C,self.antisym_error=tensor(u@self.V,w@self.V);self.nu=nu
        self.mass_error=float(cp.max(cp.abs(self.V.T@self.M@self.V-cp.eye(K))))
    def f(self,z):return cp.einsum('ijk,j,k->i',self.C,z,z)
    def jac(self,z):return cp.einsum('ijk,k->ij',self.C,z)+cp.einsum('ikj,k->ij',self.C,z)
    def step(self,z,dt):
        decay=cp.exp(-.5*self.nu*self.lam*dt);a=z*decay;y=a+dt*self.f(a)
        for it in range(20):
            mid=.5*(a+y);r=y-a-dt*self.f(mid)
            if float(cp.linalg.norm(r))<=1e-12+1e-11*float(cp.linalg.norm(a)):break
            y-=cp.linalg.solve(cp.eye(K)-.5*dt*self.jac(mid),r)
        else:raise RuntimeError('Implicit midpoint Newton20 failed')
        result=y*decay
        if not bool(cp.all(cp.isfinite(result))):raise FloatingPointError('Nonfinite autonomous ROM')
        if float(cp.dot(result,result))>float(cp.dot(z,z))*(1+2e-9)+1e-20:
            raise FloatingPointError('Unforced dissipative ROM energy increased beyond solve tolerance')
        return result,it+1
    def run(self,q0,dt):
        z=cp.linalg.solve(self.V,q0);zs=[z.copy()];iterations=[]
        for frame in range(1,30):
            for _ in range(round(.01/dt)):
                z,it=self.step(z,dt);iterations.append(it)
            zs.append(z.copy())
        z=cp.stack(zs)
        return z@self.V.T,.5*VOLUME*cp.sum(z*z,axis=1),max(iterations)
    def save(self,path):
        save_npz(path,mass=self.M,stiffness=self.S,eigenvalues=self.lam,transform=self.V,nonlinear_tensor=self.C,
                 tensor_basis=np.asarray('mass-orthonormal generalized Ritz'),volume=np.asarray(VOLUME))

def project(u,source):
    a=u.reshape(-1,K);return cp.linalg.solve(a.T@a,a.T@source.reshape(-1))

def relative(a,b):return float(cp.linalg.norm(a-b)/cp.maximum(cp.linalg.norm(b),1e-30))

def assemble_run(d,tag,u,w,fit,test,source_fit,source_test,weak_p=None):
    rom=FullROM(u,w);q0=project(fit,source_fit);q,e,it=rom.run(q0,.001);qh,eh,ith=rom.run(q0,.0005)
    rom.save(d/f'{tag}_operators.npz')
    save_npz(d/f'{tag}_trajectory.npz',coefficients=q,half_dt_coefficients=qh,energy=e,half_dt_energy=eh,
             times=np.arange(30)*.01,initial_coefficients=q0)


    n=len(u)//2;ur=u[:n];wr=w[:n]
    if weak_p is not None:
        pf=weak_p[:n].reshape(-1,K);uf=ur.reshape(-1,K)
        dr=cp.linalg.solve(pf.T@pf,uf.T@uf);wr=weak_p[:n]@dr
    refined=FullROM(ur,wr);qr,er,_=refined.run(q0,.001)
    save_npz(d/f'{tag}_quadrature_trajectory.npz',coefficients=qr,energy=er,eigenvalues=refined.lam,
             mass=refined.M,stiffness=refined.S,nonlinear_tensor=refined.C,transform=refined.V)
    rows=[]
    for f in range(30):
        value=test@q[f];half=test@qh[f];coarse=test@qr[f]
        rows.append(dict(frame=f,time=f*.01,energy=float(e[f]),heldout_energy=float(.5*VOLUME*cp.mean(cp.sum(value*value,1))),
                         dt_halving_relative=relative(half,value),quadrature_relative=relative(coarse,value)))
    initial=test@q0
    result=dict(K=32,initial_relative_l2=relative(initial,source_test),initial_mse=float(cp.mean((initial-source_test)**2)),
                initial_component_mean=cp.asnumpy(cp.mean(initial,0)).tolist(),rows=rows,
                mass_rank=int(cp.linalg.matrix_rank(rom.M)),mass_condition=float(cp.linalg.cond(rom.M)),
                eigenvalues=cp.asnumpy(rom.lam).tolist(),mass_whitening_error=rom.mass_error,
                tensor_antisymmetry_before_roundoff_projection=rom.antisym_error,
                nonlinear_energy_pairing=float(cp.dot(cp.ones(K),rom.f(cp.ones(K)))),
                max_newton_iterations=max(it,ith),quadrature_points=len(u),nested_check_points=n,
                quadrature_mass_relative=relative(refined.M,rom.M),quadrature_stiffness_relative=relative(refined.S,rom.S),
                quadrature_eigenvalues_relative=cp.asnumpy(cp.abs(refined.lam/rom.lam-1)).tolist(),
                initialization='independent full-mass projection of original saved frame000 on identical16384 points',
                integration='same Strang exact full-matrix viscosity + implicit midpoint rotational nonlinearity; no forcing or later GT injection')
    write(d/f'{tag}_metrics.json',result)
    return result,q,rom

def gauge_and_basis(d,model,c,geo,fitpoints):
    ctx=Context(c,geo,'f8');batch=CanonicalBatch(ctx,fitpoints,'f8');state=compute(model,ctx,batch,False)
    T=state['objective']['T'];coeff=state['coeff'];amp=batch.h.T@(state['a']-batch.G@coeff)/batch.hmass
    save_npz(d/'canonical_basis.npz',transform=T,scalar_coefficients=coeff,harmonic_amplitudes=amp,
             points=fitpoints,losses=state['objective']['losses'],order=state['objective']['order'])
    result=dict(**metrics(state,ctx),potential_rank=int(cp.linalg.matrix_rank(state['p'].T@state['p'])),
                velocity_rank=int(cp.linalg.matrix_rank(state['u'].T@state['u'])),
                canonical_note='fixed fit103244 scalar inverse, harmonic coefficient from assessment16384 held fixed on all other points; approximate gauge')
    del batch,state;release()
    x=fluid_points(c,geo,512,1919727);data=NeuralPoints(x,geo,c,'f8');raw=network_fields(model,data,'aud')
    G=ctx.scalar.operator(cp.asarray(x,dtype='f8'),'grad');L=ctx.scalar.operator(cp.asarray(x,dtype='f8'),'lap')
    _,dg,lg,_=lift_fields(cp.asarray(x,dtype='f8'),ctx.geo,c)
    h=dg.reshape(-1,1)-G@ctx.hcoeff;hd=lg.reshape(-1,1)-L@ctx.hcoeff
    p=(raw['a']-G@coeff-h@amp)@T;div=(raw['d']-L@coeff-hd@amp)@T
    result['independent_canonical_divergence_per_potential']=cp.asnumpy(cp.linalg.norm(div,axis=0)/cp.maximum(cp.linalg.norm(p,axis=0),1e-30)).tolist()
    save_npz(d/'canonical_diagnostic_fields.npz',points=x,potential=p,divergence=div)

    from canonical_poisson_component import shift
    small=CanonicalBatch(ctx,fluid_points(c,geo,256,908001))
    base=compute(model,ctx,small,True);sh,_=shift(small.x,ctx.geo,c);laps=[]
    for first in range(0,ctx.fit.n,2048):laps.append(shift(ctx.fit.x[first:first+2048].astype('f8'),ctx.geo,c)[1])
    delta=small.forward(sh,ctx.solve(cp.concatenate(laps)))
    scale=cp.linalg.norm(base['a'],axis=0)/cp.linalg.norm(sh);sign=cp.where((base['p'].T@delta)[:,0]>=0,1.,-1.)
    shifted=dict(objective=potential_objective(base['p']+delta*(scale*sign)[None,:],base['u'],small.n))
    gradients,_,_=objective_gradients(model,ctx,small,shifted)
    a=base['grads'][0]+base['grads'][1];b=gradients[0]+gradients[1]
    result['gauge_stress']=dict(relative_loss_change=float(cp.sum(shifted['objective']['losses'])/cp.sum(base['objective']['losses'])-1),
        parameter_gradient_relative_change=float(cp.sqrt(sum(cp.sum((y-z)**2) for y,z in zip(a,b))/sum(cp.sum(y*y) for y in a))),
        remaining_gradient_fraction=float(cp.linalg.norm(delta)/cp.linalg.norm(sh)),training_gate=False,
        scalar_rhs_precision='FP64 for final assessment; actual FP32 training graph separately verified before training')
    write(d/'canonical_metrics.json',result)
    return T,result

def diagnostics(d,model,c,geo,T,rom,q):
    points=fluid_points(c,geo,512,1919727);u,w,cu,div,_=analytic_arrays(model,points,geo,True)
    tv=T@rom.V;u=u@tv;cu=cu@tv
    strong=cp.linalg.norm((cu-u*rom.lam).reshape(-1,K),axis=0)/cp.maximum(cp.linalg.norm((u*rom.lam).reshape(-1,K),axis=0),1e-30)
    boundary=np.load(d/'reference_boundary_fields.npz');bp=boundary['points'];normals=cp.asarray(boundary['normals'])
    area=cp.asarray(boundary['area_weights']);outer=boundary['outer'].astype(bool)
    bu,bw=neural_fields(model,bp,geo,tv)
    un=cp.einsum('ni,nik->nk',normals,bu);wt=cp.cross(normals[:,None,:],bw.transpose(0,2,1))
    rows=[]
    for name,sl in [('outer',outer),('obstacle',~outer)]:
        rows.append(dict(wall=name,normal_velocity_relative=cp.asnumpy(cp.linalg.norm(un[sl],axis=0)/cp.maximum(cp.linalg.norm(bu[sl].reshape(-1,K),axis=0),1e-30)).tolist(),
                         tangential_vorticity_relative=cp.asnumpy(cp.linalg.norm(wt[sl].transpose(0,2,1).reshape(-1,K),axis=0)/cp.maximum(cp.linalg.norm(bw[sl].reshape(-1,K),axis=0),1e-30)).tolist()))
    save_npz(d/'boundary_fields.npz',points=bp,normals=normals,velocity=bu,vorticity=bw,normal_velocity=un)
    boundary_power=[];z=cp.linalg.solve(rom.V,q.T).T
    for frame in range(30):
        value=bu@z[frame];omega=bw@z[frame];normal=cp.sum(value*normals,axis=1)
        boundary_power.append(dict(frame=frame,time=frame*.01,volume_flux=float(cp.sum(area*normal)),
            advective_energy_flux=float(cp.sum(.5*area*cp.sum(value*value,axis=1)*normal)),
            strong_viscous_boundary_power=float(-.00025*cp.sum(area*cp.sum(value*cp.cross(normals,omega),axis=1)))))
    result=dict(strong_eigenrelation_relative=cp.asnumpy(strong).tolist(),velocity_divergence_max=float(cp.max(cp.abs(div@tv))),walls=rows,boundary_power=boundary_power,
        boundary_quadrature='all mapped mesh-face centroids; cofactor surface-area weights; sampled surface quadrature, not exact curved integration',
        viscous_boundary='homogeneous Hodge natural condition imposed weakly by full curl-curl form; nonzero pointwise tangential vorticity is reported, not hidden',
        potential_normal_wall='Reused pretraining analytic envelope/gradient trace identity: tangential potential zero; all32 physical fields rechecked here')
    write(d/'neural_analytic_metrics.json',result);return result
