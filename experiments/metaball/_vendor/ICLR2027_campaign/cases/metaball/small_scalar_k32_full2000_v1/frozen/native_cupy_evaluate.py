




import csv
from native_cupy_core import *

def aslist(x):return cp.asnumpy(x).tolist()
def save_npz(path,**arrays):
    tmp=Path(path).with_suffix('.tmp.npz');np.savez(tmp,**{k:cp.asnumpy(v) if isinstance(v,cp.ndarray) else v for k,v in arrays.items()});tmp.replace(path)
def eigh(A,M=None):
    if M is None:return cp.linalg.eigh((A+A.T)/2)
    C=cp.linalg.cholesky((M+M.T)/2)
    X=solve_triangular(C,A,lower=True);B=solve_triangular(C,X.T,lower=True).T
    lam,V=cp.linalg.eigh((B+B.T)/2)
    return lam,solve_triangular(C.T,V,lower=False)

def raw_first(model,data):
    A=cp.empty((3*data.n,32),dtype='f8');U=cp.empty_like(A)
    for start in range(0,32,4):
        f,_=model.forward(data,start,start+4)
        A[:,start:start+4]=f.value.reshape(4,-1).T;U[:,start:start+4]=curl1(f).reshape(4,-1).T
    return A,U

def basis_state(model,s):
    data=Data(model.c,s['geometry'],s['assessment_samples'],s['assessment_seed'],s['gauge_order'])
    raw,U=raw_first(model,data);A,coeff=data.project(raw,True);M=A.T@A/data.n;S=U.T@U/data.n
    lam,T=eigh(S,M);Tv=T/cp.sqrt(lam)[None,:];me=cp.linalg.eigvalsh(M)
    metrics=dict(mass_rank=int(cp.sum(me>me[-1]*1e-10)),raw_mass_eigenvalues=aslist(me),eigenvalues=aslist(lam),
        potential_gram_max=float(cp.max(cp.abs(T.T@M@T-cp.eye(32)))),
        weak_gauge_relative=float(cp.linalg.norm(data.B.T@A)/cp.linalg.norm(A)))
    points=data.points.copy();del data
    refinement=Data(model.c,s['geometry'],len(points),0,s['gauge_refinement_order'],points)
    Ar=refinement.project(raw)
    metrics['gauge_refinement_relative']=aslist(cp.linalg.norm(Ar-A,axis=0)/cp.linalg.norm(Ar,axis=0))
    del refinement,Ar,A
    independent=Data(model.c,s['geometry'],s['assessment_refinement_samples'],s['assessment_seed']+1000000,s['gauge_order'])
    ai,ui=raw_first(model,independent);ai=independent.project(ai)
    Li,_=eigh(ui.T@ui,ai.T@ai)
    metrics['independent_quadrature_relative']=aslist(cp.abs(Li/lam-1))
    save_npz(OUT/'basis_state.npz',points=points,potential_transform=T,velocity_transform=Tv,gauge_coefficients=coeff,
             eigenvalues=lam,raw_mass=M,raw_curl_mass=S)
    return metrics,points,T,Tv,coeff,lam

def analytic_arrays(model,points,geo,third=False):
    n=len(points);U=cp.empty((n,3,32),dtype='f8');W=cp.empty_like(U)
    CU=cp.empty_like(U) if third else None;DIV=cp.empty((n,32),dtype='f8');DA=cp.empty_like(DIV)
    for start in range(0,32,4):
        for offset in range(0,n,256):
            f=higher_fields(model,points[offset:offset+256],geo,start,start+4)
            u=jet_velocity(f);w=jet_vorticity(f)
            U[offset:offset+256,:,start:start+4]=u.transpose(1,2,0)
            W[offset:offset+256,:,start:start+4]=w.transpose(1,2,0)
            divergence=sum(jet_velocity(f,tuple(int(k==j) for k in range(3)))[...,j] for j in range(3))
            da=sum(derivative(f,j,tuple(int(k==j) for k in range(3))) for j in range(3))
            DIV[offset:offset+256,start:start+4]=divergence.T;DA[offset:offset+256,start:start+4]=da.T
            if third:
                curlw=[]
                for j in range(3):
                    a=(j+1)%3;b=(j+2)%3
                    wa=jet_vorticity(f,tuple(int(k==a) for k in range(3)))
                    wb=jet_vorticity(f,tuple(int(k==b) for k in range(3)))
                    curlw.append(wa[...,b]-wb[...,a])
                CU[offset:offset+256,:,start:start+4]=cp.stack(curlw,-1).transpose(1,2,0)
        print(json.dumps(dict(stage='analytic_Jet3',completed_modes=start+4,total_modes=32,points=n)),flush=True)
    return U,W,CU,DIV,DA

def walls(c,geo):
    rng=np.random.Generator(np.random.PCG64(912771));d=cp.asarray(rng.normal(size=(512,3)));d/=cp.linalg.norm(d,axis=1)[:,None]
    out=.5+c.outer_half_extent*d/cp.sum(d**c.outer_superellipsoid_power,1)[:,None]**(1/c.outer_superellipsoid_power)
    center=cp.asarray(geo)[1,:3];lo=cp.zeros(512);hi=cp.ones(512)
    for _ in range(55):
        mid=(lo+hi)/2;_,level=levels(center+mid[:,None]*d,cp.asarray(geo),c)
        inside=level<0;lo=cp.where(inside,mid,lo);hi=cp.where(inside,hi,mid)
    inside=center+((lo+hi)/2)[:,None]*d
    no=((out-.5)/c.outer_half_extent)**(c.outer_superellipsoid_power-1);no/=cp.linalg.norm(no,axis=1)[:,None]
    delta=inside[:,None]-cp.asarray(geo)[None,:,:3];length=cp.linalg.norm(delta,axis=2);dist=length-cp.asarray(geo)[None,:,3]
    w=cp.exp(-(dist-dist.min(1)[:,None])/c.metaball_softmin_tau);w/=w.sum(1)[:,None]
    ni=-cp.sum(w[:,:,None]*delta/length[:,:,None],1);ni/=cp.linalg.norm(ni,axis=1)[:,None]
    return cp.asnumpy(cp.concatenate((out,inside))),cp.concatenate((no,ni))

def neural_assessment(model,s,points,T,Tv,coeff,lam):
    U,W,_,_,_=analytic_arrays(model,points,s['geometry']);U=U@Tv;W=W@Tv
    diag=fluid_points(model.c,s['geometry'],512,s['assessment_seed']+727)
    u,w,cu,div,da=analytic_arrays(model,diag,s['geometry'],True)
    u=u@Tv;cu=cu@Tv
    residual=cp.linalg.norm((cu-u*lam).reshape(-1,32),axis=0)/cp.linalg.norm((u*lam).reshape(-1,32),axis=0)
    scalar=scalar_gauge_jets(diag,s['geometry'],model.c,s['gauge_order'])
    lap=sum(2*scalar[:,:,INDEX[tuple(2*int(k==j) for k in range(3))]] for j in range(3))
    divcanon=(da-lap@coeff)@T
    curlpotential=u*cp.sqrt(lam)[None,None,:]
    coulomb=cp.linalg.norm(divcanon,axis=0)/cp.linalg.norm(curlpotential.reshape(-1,32),axis=0)
    bp,normal=walls(model.c,s['geometry']);bu,bw,_,_,_=analytic_arrays(model,bp,s['geometry']);bu=bu@Tv;bw=bw@Tv
    un=cp.einsum('nj,njk->nk',normal,bu);wn=cp.einsum('nj,njk->nk',normal,bw);wt=bw-normal[:,:,None]*wn[:,None,:]
    records=[]
    for name,sl in [('outer',slice(0,512)),('obstacle',slice(512,None))]:
        records.append(dict(wall=name,normal_velocity_relative=aslist(cp.linalg.norm(un[sl],axis=0)/cp.linalg.norm(bu[sl].reshape(-1,32),axis=0)),
            tangential_vorticity_relative=aslist(cp.linalg.norm(wt[sl].reshape(-1,32),axis=0)/cp.linalg.norm(bw[sl].reshape(-1,32),axis=0)),
            tangential_speed_rms=aslist(cp.sqrt(cp.mean(cp.sum(bu[sl]**2,axis=1)-un[sl]**2,axis=0)))))
    save_npz(OUT/'boundary_fields.npz',points=bp,normals=normal,velocity=bu,vorticity=bw)
    metrics=dict(velocity_eigenrelation_relative=aslist(residual),independent_coulomb_relative=aslist(coulomb),
        velocity_divergence_max=float(cp.max(cp.abs(div@Tv))),walls=records,
        boundary_rule='512 frozen PCG64 radial directions per exact implicit surface; directional RMS, not uniform area quadrature',
        derivatives='native CuPy normalized Taylor algebra through order3; no spatial FD or mesh differentiation')
    return U,W,metrics

def saved_initial(points):
    from cupyx.scipy.ndimage import map_coordinates
    v=cp.asarray(np.load(DNS/'velocity/frame_000.npy'));n=v.shape[0]
    indices=(cp.asarray(points)[:,::-1]*n-.5).T
    return cp.stack([map_coordinates(v[...,j],indices,order=1,mode='constant',cval=0) for j in range(3)],axis=1)

def historical_reference(s,points):

    sys.path.insert(0,str(ROOT/'vendor'))
    from skfem import MeshTet
    source=ROOT/'results/fitted_metaball_reference_v2/h0.075_surface80_K2048/modes.npz'
    z=np.load(source);mesh=MeshTet(z['p'],z['t']);K=s['reference_refinement_K']

    p=cp.asarray(mesh.p.T);t=cp.asarray(mesh.t.T);vertices=p[t]
    J=(vertices[:,1:]-vertices[:,:1]).transpose(0,2,1);inv=cp.linalg.inv(J)
    barygrad=cp.concatenate((-inv.sum(1)[:,None,:],inv),axis=1)
    lam=cp.asarray(z['eigenvalues'][:K]);co=cp.zeros((mesh.edges.shape[1],K),dtype='f8')
    co[cp.asarray(z['free_edges'])]=cp.asarray(z['potential'][:,:K])/cp.sqrt(lam)[None,:]
    local=co[cp.asarray(mesh.t2e.T)]
    pairs=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
    orientations=cp.stack([1-2*(t[:,a]>t[:,b]) for a,b in pairs],axis=1)
    local*=orientations[:,:,None]
    curls=cp.stack([2*cp.cross(barygrad[:,a],barygrad[:,b]) for a,b in pairs],axis=1)
    cellU=cp.einsum('nea,nek->nak',curls,local)
    xyz=cp.asarray(points,dtype='f8');U=cp.empty((len(points),3,K),dtype='f8');W=cp.empty_like(U);missed=0
    for offset in range(0,len(points),128):
        q=xyz[offset:offset+128];delta=q[None]-vertices[:,0,None]
        bary=cp.einsum('eij,enj->eni',inv,delta)
        inside=cp.all(bary>=-1e-10,axis=2)&(bary.sum(2)<=1+1e-10)
        found=inside.any(0);cells=cp.argmax(inside,axis=0)
        fallback=cp.argmin(cp.sum((q[None]-vertices.mean(1)[:,None])**2,axis=2),axis=0)
        cells=cp.where(found,cells,fallback);missed+=int(cp.sum(~found))
        bc=cp.einsum('nij,nj->ni',inv[cells],q-vertices[cells,0]);bc=cp.concatenate((1-bc.sum(1)[:,None],bc),axis=1)
        fields=cp.stack([bc[:,a,None]*barygrad[cells,b]-bc[:,b,None]*barygrad[cells,a] for a,b in pairs],axis=1)
        potential=cp.einsum('nea,nek->nak',fields,local[cells])
        U[offset:offset+128]=cellU[cells];W[offset:offset+128]=potential*lam
    lineage=dict(source_reference_sha256=sha(source),saved_GT_initial_sha256=sha(DNS/'velocity/frame_000.npy'),
        strong_Hodge_GT=False,continuum_reference_converged=False,
        missed_smooth_points_fraction=missed/len(points),
        method='Preserved fitted eigenmodes, analytic element evaluation in CuPy, weak-adjoint vorticity lambda*A; physical saved initial field projected using PCG64 quadrature',
        limitations='Mesh/smooth-domain mismatch uses explicit nearest-cell extension; piecewise curl is not a strong smooth Hodge field. This diagnostic reference does not certify the required GT gate.')
    write(OUT/'reference_lineage.json',lineage)
    return U,W,lineage

class ROM:
    def __init__(self,U,W,viscosity):
        self.n=len(U);self.K=U.shape[2];u=U.reshape(-1,self.K);w=W.reshape(-1,self.K)
        self.M=u.T@u/self.n;self.L=w.T@w/self.n;self.lam,self.V=eigh(self.L,self.M)
        self.U=u@self.V;self.W=w@self.V;self.nu=viscosity
    def nonlinear(self,z,jac=False):
        u=(self.U@z).reshape(-1,3);w=(self.W@z).reshape(-1,3)
        f=self.U.T@cp.cross(u,w).reshape(-1)/self.n
        if not jac:return f
        du=self.U.reshape(-1,3,self.K).transpose(0,2,1);dw=self.W.reshape(-1,3,self.K).transpose(0,2,1)
        d=cp.cross(du,w[:,None,:])+cp.cross(u[:,None,:],dw)
        return f,self.U.T@d.transpose(0,2,1).reshape(-1,self.K)/self.n
    def step(self,z,dt,implicit):
        decay=cp.exp(-.5*self.nu*self.lam*dt);z=z*decay
        if implicit:
            right=z+.5*dt*self.nonlinear(z);y=z.copy()
            for _ in range(12):
                f,J=self.nonlinear(y,True);r=y-.5*dt*f-right
                if float(cp.linalg.norm(r))<1e-10+1e-9*float(cp.linalg.norm(y)):break
                y-=cp.linalg.solve(cp.eye(self.K)-.5*dt*J,r)
            else:raise RuntimeError('Frozen Newton12 did not converge')
        else:
            a=self.nonlinear(z);b=self.nonlinear(z+.5*dt*a);c=self.nonlinear(z+.5*dt*b);d=self.nonlinear(z+dt*c)
            y=z+dt*(a+2*b+2*c+d)/6
        return y*decay
    def run(self,source,s,tag,dt,implicit=False):
        folder=OUT/tag;folder.mkdir(exist_ok=True);q0=self.U.T@source.reshape(-1)/self.n
        q=q0.copy();start=0;checkpoint=folder/'checkpoint.npz'
        if checkpoint.exists():
            z=np.load(checkpoint);q=cp.asarray(z['q']);start=int(z['next_frame'])
        for frame in range(start,s['frames']):
            if frame:
                for _ in range(round(s['frame_dt']/dt)):q=self.step(q,dt,implicit)
            if not bool(cp.all(cp.isfinite(q))):raise FloatingPointError(f'{tag}: nonfinite frame{frame}')
            np.save(folder/f'frame_{frame:03d}.npy',cp.asnumpy((self.U@q).reshape(-1,3)))
            save_npz(checkpoint,q=q,next_frame=np.asarray(frame+1))
            print(json.dumps(dict(stage=tag,frame=frame,time=frame*s['frame_dt'],energy=float(cp.dot(q,q)/2))),flush=True)
        return folder,dict(initial_energy=float(cp.dot(q0,q0)/2),final_energy=float(cp.dot(q,q)/2),K=self.K,dt=dt)

def flow(U,W,points,s):
    source=saved_initial(points);refU,refW,lineage=historical_reference(s,points);refs={};rmetrics={}
    for K,dt in [(s['reference_K'],s['dt']),(s['reference_refinement_K'],s['dt']),(s['reference_refinement_K'],s['dt']/2)]:
        tag=f'reference_K{K}_dt{dt:g}';status('native_cupy_reference_rollout',dict(K=K,dt=dt))
        rom=ROM(refU[:,:,:K],refW[:,:,:K],s['viscosity']);refs[(K,dt)],rmetrics[tag]=rom.run(source,s,tag,dt)
        del rom;cp.get_default_memory_pool().free_all_blocks()
    hi=(s['reference_refinement_K'],s['dt']);lo=(s['reference_K'],s['dt']);fine=(s['reference_refinement_K'],s['dt']/2)
    initial=cp.asarray(np.load(refs[hi]/'frame_000.npy'))
    source_projection=float(cp.linalg.norm(initial-source)/cp.linalg.norm(source))
    del refU,refW,source
    rom=ROM(U,W,s['viscosity']);paths={};details={}
    for dt in (s['dt'],s['dt']/2):paths[dt],details[str(dt)]=rom.run(initial,s,f'neural_dt{dt:g}',dt,True)
    save_npz(OUT/'neural_operators.npz',mass=rom.M,stiffness=rom.L,internal_transform=rom.V,eigenvalues=rom.lam)
    rows=[]
    for frame in range(s['frames']):
        name=f'frame_{frame:03d}.npy';gt=cp.asarray(np.load(refs[hi]/name));normal=cp.linalg.norm(gt)
        u=cp.asarray(np.load(paths[s['dt']]/name));uf=cp.asarray(np.load(paths[s['dt']/2]/name))
        rows.append(dict(frame=frame,time=frame*s['frame_dt'],relative_velocity_error=float(cp.linalg.norm(u-gt)/normal),
            dt_halving_relative=float(cp.linalg.norm(uf-u)/normal),
            reference_capacity_relative=float(cp.linalg.norm(cp.asarray(np.load(refs[lo]/name))-gt)/normal),
            reference_dt_halving_relative=float(cp.linalg.norm(cp.asarray(np.load(refs[fine]/name))-gt)/normal),
            neural_energy=float(cp.sum(u*u)/(2*len(points))),reference_energy=float(cp.sum(gt*gt)/(2*len(points)))))
    report=dict(rows=rows,lineage=lineage,source_projection_relative=source_projection,neural_details=details,reference_details=rmetrics,
        initialization='Each basis reconstructs the same physical high-reference initial field once; no later GT injection; different random coefficient vectors are never used as physical matching')
    write(OUT/'flow_metrics.json',report);return report

def summarize(s,basis,flow):
    g=s['gates'];checks=dict(reload=basis['reload_max_abs']==0,rank=basis['mass_rank']==32,
        gram=basis['potential_gram_max']<g['gram'],gauge_refinement=max(basis['gauge_refinement_relative'])<g['gauge_refinement'],
        independent_quadrature=max(basis['independent_quadrature_relative'])<g['independent_quadrature'],
        coulomb=max(basis['independent_coulomb_relative'])<g['coulomb'],
        normal_wall=max(max(w['normal_velocity_relative']) for w in basis['walls'])<g['normal_wall'],
        tangential_vorticity_wall=max(max(w['tangential_vorticity_relative']) for w in basis['walls'])<g['tangential_vorticity_wall'],
        velocity_eigenrelation=max(basis['velocity_eigenrelation_relative'])<g['velocity_eigenrelation'],
        dt_halving=max(r['dt_halving_relative'] for r in flow['rows'])<g['dt_halving'],
        reference_capacity=max(r['reference_capacity_relative'] for r in flow['rows'])<g['reference_capacity'],
        reference_strong_Hodge=flow['lineage']['strong_Hodge_GT'],reference_continuum_converged=flow['lineage']['continuum_reference_converged'],
        full_frames=len(flow['rows'])==30,
        energy=max(r['neural_energy'] for r in flow['rows'])<=(1+g['energy_growth'])*flow['rows'][0]['neural_energy'])
    with (OUT/'framewise_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flow['rows'][0]));w.writeheader();w.writerows(flow['rows'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=[json.loads(x) for x in (OUT/'training.jsonl').read_text().splitlines()]
    blue='#4C72B0';red='#C44E52';fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    axes[0].plot([r['epoch'] for r in history],[r['mean_rayleigh'] for r in history],color=blue);axes[0].set(xlabel='Epoch',ylabel='Mean Rayleigh quotient',title='Native CuPy K32 / 2000 epochs')
    t=[r['time'] for r in flow['rows']]
    axes[1].plot(t,[r['relative_velocity_error'] for r in flow['rows']],color=red,label='Neural / weak reference')
    axes[1].plot(t,[r['reference_capacity_relative'] for r in flow['rows']],color=blue,label='Reference capacity difference')
    axes[1].set(xlabel='Time',ylabel='Relative velocity L2',title='Autonomous saved-source reconstruction');axes[1].legend(fontsize=7)
    axes[2].semilogy(range(1,33),basis['velocity_eigenrelation_relative'],color=red,label='Strong velocity eigenrelation')
    axes[2].semilogy(range(1,33),basis['independent_coulomb_relative'],color=blue,label='Coulomb divergence')
    axes[2].set(xlabel='Mode',ylabel='Relative defect',title='All-mode analytic diagnostics');axes[2].legend(fontsize=7)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'smoke_summary.png',dpi=220);fig.savefig(OUT/'smoke_summary.pdf');plt.close(fig)
    result=dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v],new_K2048_allowed=all(checks.values()),
        backend='native_cupy',epochs=2000,K=32,torch_imported='torch' in sys.modules,
        completed_numerical_stages=['training','checkpoint reload','fresh transforms','all32 analytic derivatives','saved-source reconstruction','autonomous30frames','dt-halving','English PNG/PDF'],
        limitation='Historical weak FE reference is a diagnostic adaptation, not certified strong-Hodge GT; required compatibility and continuum gates remain explicit.')
    write(OUT/'summary.json',result)
    (OUT/'SUMMARY.md').write_text('# Native CuPy Metaball K32 smoke\n\n'+('PASS' if result['passed'] else 'NOT ACCEPTED')+
        ' after 2000 epochs and the full numerical evaluation chain.\n\nFailed gates: '+', '.join(result['failed'])+'.\n\n'+result['limitation']+'\n',encoding='utf-8')
    status('native_cupy_smoke_complete',dict(passed=result['passed'],failed=result['failed'],epochs=2000))

def evaluate(s):
    model=Model(config_namespace(),32,s['seed']);model.load(OUT/'checkpoint.npz',sha(OUT/'config.json'))
    if model.epoch!=2000:raise ValueError('All 2000 epochs required')
    with np.load(OUT/'checkpoint.npz') as z:
        error=max(float(cp.max(cp.abs(a-cp.asarray(z[f'{name}_{j}'])))) for name in ['W','b'] for j,a in enumerate(getattr(model,name)))
    if (OUT/'basis_metrics.json').exists() and (OUT/'neural_quadrature.npz').exists():
        basis=json.loads((OUT/'basis_metrics.json').read_text());q=np.load(OUT/'neural_quadrature.npz');U=cp.asarray(q['velocity']);W=cp.asarray(q['vorticity']);points=q['points']
    else:
        status('native_cupy_basis_evaluation',dict(stage='fresh independent scalar gauge and all32 transforms'))
        basis,points,T,Tv,coeff,lam=basis_state(model,s)
        U,W,analytic=neural_assessment(model,s,points,T,Tv,coeff,lam);basis.update(analytic,reload_max_abs=error)
        write(OUT/'basis_metrics.json',basis);save_npz(OUT/'neural_quadrature.npz',velocity=U,vorticity=W,points=points)
    result=json.loads((OUT/'flow_metrics.json').read_text()) if (OUT/'flow_metrics.json').exists() else flow(U,W,points,s)
    summarize(s,basis,result)
