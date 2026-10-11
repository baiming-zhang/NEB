






import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
import time
import traceback
from cupyx import errstate
from native_cupy_core import *
import native_cupy_core as core
from native_cupy_evaluate import eigh, saved_initial, save_npz
from native_cupy_verify import jvp

BASE=OUT
DEST=ROOT/'results/native_gauge_residual_preflight_v1'

class Taylor4:
    def __init__(self):
        self.multi=[a for k in range(5) for a in itertools.product(range(5),repeat=3) if sum(a)==k]
        self.index={a:i for i,a in enumerate(self.multi)}
        self.pairs=[[(self.index[b],self.index[tuple(a[j]-b[j] for j in range(3))])
            for b in self.multi if all(b[j]<=a[j] for j in range(3))] for a in self.multi]
    def constant(self,a):
        a=cp.asarray(a,dtype='f8');v=cp.zeros((*a.shape,35),dtype='f8');v[...,0]=a;return v
    def mul(self,a,b):return cp.stack([sum(a[...,i]*b[...,j] for i,j in pairs) for pairs in self.pairs],axis=-1)
    def unary(self,a,derivatives):
        d=a.copy();d[...,0]=0;power=d;out=self.constant(derivatives[0])
        for j in range(1,5):
            out+=derivatives[j][...,None]*power/math.factorial(j)
            if j<4:power=self.mul(power,d)
        return out
    def sin(self,a):return self.unary(a,[cp.sin(a[...,0]+j*cp.pi/2) for j in range(5)])
    def exp(self,a):return self.unary(a,[cp.exp(a[...,0])]*5)
    def log(self,a):
        x=a[...,0];return self.unary(a,[cp.log(x),1/x,-1/x**2,2/x**3,-6/x**4])
    def power(self,a,p):
        x=a[...,0];d=[x**p];factor=1.
        for j in range(1,5):factor*=p-j+1;d.append(factor*x**(p-j))
        return self.unary(a,d)
    def div(self,a,b):return self.mul(a,self.power(b,-1))
    def geometry(self,points,geo,c):
        xyz=[];n=len(points)
        for j in range(3):
            x=self.constant(points[:,j]);x[:,self.index[tuple(int(k==j) for k in range(3))]]=1;xyz.append(x)
        out=self.constant(cp.ones(n))
        for x in xyz:out-=self.power((x-self.constant(cp.full(n,.5)))/c.outer_half_extent,8)
        distances=[]
        for center in geo:
            delta=[x-self.constant(cp.full(n,center[j])) for j,x in enumerate(xyz)]
            distances.append(self.power(sum(self.mul(d,d) for d in delta),.5)-self.constant(cp.full(n,center[3])))
        shift=cp.min(cp.stack([d[:,0] for d in distances]),axis=0)
        inner=self.constant(shift)-c.metaball_softmin_tau*self.log(sum(self.exp(-(d-self.constant(shift))/c.metaball_softmin_tau) for d in distances))
        fo=self.div(out,out+self.constant(cp.full(n,c.outer_envelope_transition)))
        fi=self.div(inner,inner+self.constant(cp.full(n,c.obstacle_envelope_transition)))
        return xyz,self.mul(fo,fi),self.div(fo,fo+fi)
    def scalars(self,xyz,e,chi,order=5):
        values=[];n=len(e)
        for x in xyz:
            y=2*x-self.constant(cp.ones(n));p=[self.constant(cp.ones(n)),y]
            for m in range(1,order-1):p.append(((2*m+1)*self.mul(y,p[-1])-m*p[-2])/(m+1))
            values.append(cp.stack(p,axis=1))
        mi=cp.asarray(list(itertools.product(range(order),repeat=3)))
        product=self.mul(values[0][:,mi[:,0]],self.mul(values[1][:,mi[:,1]],values[2][:,mi[:,2]]))
        return cp.concatenate((self.mul(e[:,None,:],product),chi[:,None,:]),axis=1)
    def gradient(self,a):return cp.stack([a[...,self.index[tuple(int(k==j) for k in range(3))]] for j in range(3)],axis=-1)
    def laplacian(self,a):return sum(2*a[...,self.index[tuple(2*int(k==j) for k in range(3))]] for j in range(3))

def specification():
    s=json.loads((BASE/'config.json').read_text())
    return dict(version='native_gauge_residual_preflight_v1',checkpoint_sha256=sha(BASE/'checkpoint.npz'),
        basis_state_sha256=sha(BASE/'basis_state.npz'),geometry=s['geometry'],
        fit_points='first8192 original frozen assessment points from basis_state.npz',
        independent_points='first8192 of the old independent-refinement PCG64 seed1919000 lineage',
        fit_count=8192,independent_count=8192,independent_seed=s['assessment_seed']+1000000,
        baseline_order=5,enrichment='one block of32 residual scalar functions e_g*r_i, anchored to saved weights and saved order5 gauge coefficients',
        training_steps=0,new_GT_frames=0,frequency_change=False,
        thresholds=dict(taylor4_max=1e-10,scalar_jet3_relative_agreement=2e-12,
            projector_adjoint=1e-9,parameter_directional_adjoint=2e-5,
            independent_coulomb_relative=.02,heldout_loss_relative=.01,
            continuum_gauge_error_over_approximate_norm=.09950371902099892,
            maximum_Gram_condition=1e12),
        heldout='one moderate-frequency strong-zero gradient: grad(e_g^2*cos(5pi*x)*sin(3pi*y)*cos(4pi*z)); not an order sweep',
        error_control='conditional continuum L2 residual/inner-flux bound, evaluated as MC plug-in diagnostics, not a certified norm bound',
        direct_projection='fit on original frozen16384 points, test on the independent8192 points; original saved DNS frame0, no reference512 initialization')

def jet_check(t,c,geo):
    x=cp.asarray(fluid_points(c,geo,32,917017),dtype='f8');phase=t.constant(x@cp.asarray([1.1,-.8,2.3]))
    wave=cp.asarray([1.1,-.8,2.3])
    for j in range(3):phase[:,t.index[tuple(int(k==j) for k in range(3))]]=wave[j]
    s=t.sin(phase);exact=cp.stack([cp.sin(x@wave+sum(a)*cp.pi/2)*cp.prod(wave**cp.asarray(a))/math.prod(math.factorial(v) for v in a) for a in t.multi],axis=1)
    error=float(cp.max(cp.abs(s-exact)))
    xyz,e,chi=t.geometry(x,cp.asarray(geo),c);scalar=t.scalars(xyz,e,chi)
    old_scalar=scalar_gauge_jets(x,geo,c,5)
    absolute=float(cp.max(cp.abs(scalar[:,:,:20]-old_scalar)))
    agreement=absolute/max(1.,float(cp.max(cp.abs(old_scalar))))
    r=dict(manufactured_fourth_derivatives_max=error,independent_scalar_Jet3_agreement_max=agreement,
        independent_scalar_Jet3_absolute_difference=absolute,
        passed=error<1e-10 and agreement<2e-12)
    write(DEST/'Taylor4_verification.json',r)
    if not r['passed']:raise AssertionError(r)

def fields(model,t,points,geo,anchor,tag):
    cache=DEST/(tag+'.npz')
    if cache.exists():
        with np.load(cache) as z:
            if not np.array_equal(z['points'],points):raise ValueError('Cached field points differ')
            return {k:(float(z[k]) if k=='seconds' else cp.asarray(z[k])) for k in z.files if k!='points'}
    n=len(points);A=cp.empty((3*n,32),dtype='f8');U=cp.empty_like(A)
    rawdiv=cp.empty((n,32),dtype='f8');G0=cp.empty((3*n,126),dtype='f8');L0=cp.empty((n,126),dtype='f8')
    GR=cp.empty((3*n,32),dtype='f8');LR=cp.empty((n,32),dtype='f8');chi=cp.empty(n);dchi=cp.empty((n,3))
    shift=cp.empty((3*n,1),dtype='f8');shiftlap=cp.empty((n,1),dtype='f8')
    c=model.c;begin=time.perf_counter()
    for start in range(0,n,256):
        stop=min(n,start+256);x=cp.asarray(points[start:stop],dtype='f8')
        xyz,e,ch=t.geometry(x,cp.asarray(geo),c);scalars=t.scalars(xyz,e,ch)
        phi=cp.einsum('nsc,sk->nkc',scalars,anchor)
        r=cp.empty((len(x),32,10),dtype='f8')
        for first in range(0,32,4):
            f=higher_fields(model,points[start:stop],geo,first,first+4)
            A[3*start:3*stop,first:first+4]=f[...,0].reshape(4,-1).T
            U[3*start:3*stop,first:first+4]=jet_velocity(f).reshape(4,-1).T
            for k,a in enumerate(MULTI[:10]):
                div=0
                for j in range(3):
                    aj=list(a);aj[j]+=1
                    div+=(a[j]+1)*f[...,j,INDEX[tuple(aj)]]
                target=0
                for j in range(3):
                    aj=list(a);aj[j]+=2
                    target+=(a[j]+1)*(a[j]+2)*phi[:,first:first+4,t.index[tuple(aj)]]
                r[:,first:first+4,k]=div.T-target
                if k==0:rawdiv[start:stop,first:first+4]=div.T
        gradR=cp.stack([r[:,:,INDEX[tuple(int(k==j) for k in range(3))]] for j in range(3)],axis=-1)
        lapR=sum(2*r[:,:,INDEX[tuple(2*int(k==j) for k in range(3))]] for j in range(3))
        de=t.gradient(e);le=t.laplacian(e)
        gradpsi=e[:,0,None,None]*gradR+de[:,None,:]*r[:,:,0,None]
        lappsi=e[:,0,None]*lapR+2*cp.sum(de[:,None,:]*gradR,axis=2)+le[:,None]*r[:,:,0]
        GR[3*start:3*stop]=gradpsi.transpose(0,2,1).reshape(-1,32);LR[start:stop]=lappsi
        G0[3*start:3*stop]=t.gradient(scalars).transpose(0,2,1).reshape(-1,126);L0[start:stop]=t.laplacian(scalars)
        chi[start:stop]=ch[:,0];dchi[start:stop]=t.gradient(ch)
        cosineX=t.sin(xyz[0]*(5*cp.pi)+t.constant(cp.full(len(x),cp.pi/2)))
        sineY=t.sin(xyz[1]*(3*cp.pi));cosineZ=t.sin(xyz[2]*(4*cp.pi)+t.constant(cp.full(len(x),cp.pi/2)))
        hs=t.mul(t.mul(e,e),t.mul(cosineX,t.mul(sineY,cosineZ)))
        shift[3*start:3*stop,0]=t.gradient(hs).reshape(-1);shiftlap[start:stop,0]=t.laplacian(hs)
        if (start//256)%8==0:print(json.dumps(dict(stage=tag,points=stop,total=n)),flush=True)
    cp.cuda.Stream.null.synchronize()
    result=dict(A=A,U=U,div=rawdiv,G0=G0,L0=L0,GR=GR,LR=LR,chi=chi,dchi=dchi,shift=shift,shiftlap=shiftlap,seconds=time.perf_counter()-begin)
    save_npz(cache,points=points,**result)
    return result

class Projector:
    def __init__(self,G):
        self.G=G;self.scales=cp.linalg.norm(G,axis=0);self.B=G/self.scales
        self.gram=self.B.T@self.B;self.eigenvalues=cp.linalg.eigvalsh(self.gram)
        self.condition=float(self.eigenvalues[-1]/self.eigenvalues[0]);self.rank=int(cp.sum(self.eigenvalues>self.eigenvalues[-1]*1e-12))
        with errstate(linalg='raise'):self.factor=cp.linalg.cholesky(self.gram)
    def solve(self,x):return solve_triangular(self.factor.T,solve_triangular(self.factor,x,lower=True),lower=False)
    def coeff(self,A):return self.solve(self.B.T@A)/self.scales[:,None]
    def project(self,A):return A-self.G@self.coeff(A)

def scores(fit,val,projector,Gval,Lval,geo,volume):
    co=projector.coeff(fit['A']);a=val['A']-Gval@co;r=val['div']-Lval@co;n=len(val['chi'])
    M=a.T@a/n;S=val['U'].T@val['U']/n;lam,T=eigh(S,M)
    transformed_r=r@T;transformed_a=(a@T).reshape(n,3,32);transformed_u=val['U']@T
    divnorm=cp.linalg.norm(transformed_r,axis=0);unorm=cp.linalg.norm(transformed_u,axis=0)
    normA=cp.linalg.norm(transformed_a.reshape(-1,32),axis=0)*cp.sqrt(volume/n)
    flux=volume*cp.mean(cp.sum(transformed_a*val['dchi'][:,:,None],axis=1)+transformed_r*val['chi'][:,None],axis=0)
    Cp=.98/(cp.pi*cp.sqrt(3.));solid_lower=4*cp.pi*max(float(s[3]) for s in geo)**3/3
    bound=Cp*(divnorm*cp.sqrt(volume/n)+cp.abs(flux)/cp.sqrt(solid_lower))
    rawQ=cp.sum(val['U']**2,axis=0)/cp.sum(a*a,axis=0)
    out=dict(independent_Coulomb_relative=as_host(divnorm/unorm),Ritz_values=as_host(lam),raw_channel_quotients=as_host(rawQ),
        raw_canonical_RMS=as_host(cp.sqrt(cp.sum(a*a,axis=0)/n)),Gram_condition=projector.condition,
        gauge_error_bound_estimate_over_A_norm=as_host(bound/normA),inner_flux_estimate=as_host(flux),
        bound_label='conditional continuum bound with MC plug-in residual/flux norms; not a certified continuum upper bound')
    scales=cp.linalg.norm(fit['A'],axis=0)/cp.linalg.norm(fit['shift'])
    cf=projector.coeff(fit['A']+fit['shift']*scales[None,:])
    shifted=val['A']+val['shift']*scales[None,:]-Gval@cf
    out['heldout_shift_canonical_relative']=as_host(cp.linalg.norm(shifted-a,axis=0)/cp.linalg.norm(a,axis=0))
    out['heldout_shift_quotient_relative']=as_host(cp.abs(cp.sum(a*a,axis=0)/cp.sum(shifted*shifted,axis=0)-1))
    return out,co,a

def as_host(x):return cp.asnumpy(x).tolist()

def parameter_check(model,data,projector):


    begin=time.perf_counter();data.project=lambda a:projector.project(a.astype('f8'))
    n=data.n;A=cp.empty((3*n,32),dtype='f8');U=cp.empty_like(A)
    for i in range(0,32,4):
        f,_=model.forward(data,i,i+4);A[:,i:i+4]=f.value.reshape(4,-1).T;U[:,i:i+4]=curl1(f).reshape(4,-1).T
    p=peel(A,U,data,True);rng=cp.random.RandomState(2717);fw=cp.asarray(0.,dtype='f8');rv=cp.asarray(0.,dtype='f8')
    for i in range(0,32,4):
        _,cache=model.forward(data,i,i+4,True)
        dw=[rng.normal(size=w[i:i+4].shape).astype('f4')*.01 for w in model.W]
        db=[rng.normal(size=b[i:i+4].shape).astype('f4')*.01 for b in model.b]
        tangent=jvp(model,data,dw,db,i,i+4)
        da=tangent.value.reshape(4,-1).T.astype('f8');du=curl1(tangent).reshape(4,-1).T.astype('f8')
        fw+=cp.sum(p['GA'][:,i:i+4]*da)+cp.sum(p['GU'][:,i:i+4]*du)
        adj=Jet1(p['GA'][:,i:i+4].T.reshape(4,n,3).astype('f4'),curl_adjoint(p['GU'][:,i:i+4].T.reshape(4,n,3).astype('f4')))
        gw,gb=model.backward(data,i,i+4,cache,adj)
        rv+=sum(cp.sum(a.astype('f8')*b) for a,b in zip(gw+gb,dw+db))
    x=rng.normal(size=(3*n,2));y=rng.normal(size=x.shape)
    adjoint=float(abs(cp.sum(projector.project(x)*y)-cp.sum(x*projector.project(y)))/(cp.linalg.norm(x)*cp.linalg.norm(y)))
    parameter=float(abs(fw-rv)/cp.maximum(abs(fw)+abs(rv),1e-30))
    return dict(saved_checkpoint_all32_parameter_directional_adjoint=parameter,projector_adjoint=adjoint,
        anchors_frozen=True,moving_anchor_derivative_tested=False,optimizer_steps=0,
        all32_forward_direction_and_backward_seconds=time.perf_counter()-begin)

def direct_projection(model,points_val,Urawval):
    z=np.load(BASE/'basis_state.npz');Tv=cp.asarray(z['velocity_transform'])
    q=np.load(BASE/'neural_quadrature.npz');points=q['points'];Ufit=cp.asarray(q['velocity']).reshape(-1,32)
    source_fit=saved_initial(points);source_val=saved_initial(points_val)
    M=Ufit.T@Ufit/len(points);rhs=Ufit.T@source_fit.reshape(-1)/len(points);coeff=cp.linalg.solve(M,rhs)
    fit=Ufit@coeff;Uval=Urawval@Tv;prediction=Uval@coeff
    return dict(fit_points=len(points),independent_points=len(points_val),
        direct_original_DNS_fit_relative=float(cp.linalg.norm(fit-source_fit.reshape(-1))/cp.linalg.norm(source_fit)),
        direct_original_DNS_independent_relative=float(cp.linalg.norm(prediction-source_val.reshape(-1))/cp.linalg.norm(source_val)),
        fit_retained_energy_fraction=float(cp.sum(fit*fit)/cp.sum(source_fit*source_fit)),
        independent_prediction_energy_fraction=float(cp.sum(prediction*prediction)/cp.sum(source_val*source_val)),
        normal_equation_residual=float(cp.linalg.norm(M@coeff-rhs)/cp.linalg.norm(rhs)),
        initialization='Original saved DNS frame_000 directly, not reference512-projected frame0',
        source_sha256=sha(DNS/'velocity/frame_000.npy'),basis_checkpoint_sha256=sha(BASE/'checkpoint.npz'),
        interpolation='existing cell-centered zyx trilinear interpolation in CuPy; this interpolates saved source data, not the learned field',
        span_invariance='A gauge-only correction cannot change this K32 physical velocity span or improve its best projection error.')
