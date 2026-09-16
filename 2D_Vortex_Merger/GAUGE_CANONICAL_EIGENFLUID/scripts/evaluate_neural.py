"""Potential Ritz diagnostics and full-matrix velocity Galerkin evaluation.

Potential eigenvalues are not substituted for the true viscous operator of
an imperfect neural basis. The full mass and curl stiffness are assembled,
then diagonalized only as a change of ROM coordinates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import cupy as cp
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
from verify_analytic import ROOT, CASE, CMAP, RED, BLUE, load_gt, metrics, savefig, write_json, analytic_fields, waves
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0,str(CASE/'src'))
from eigenfluid.gen.vortex_merger2d.hardcoded_backend import HardcodedModel, cfg_namespace

def evaluate_fields(model,cfg,points):
    values=np.empty((len(points),cfg.modes,4),dtype=np.float64)
    for start in range(0,len(points),2048):
        end=min(start+2048,len(points))
        for mode,f,_ in model.forward_stream_jet2(cfg,cp.asarray(points[start:end],dtype=cp.float32),0.,0.,False):
            j=int(mode.mode_index)
            values[start:end,j,0]=cp.asnumpy(f.value[:,0])
            values[start:end,j,1]=cp.asnumpy(f.grad[:,0,1])
            values[start:end,j,2]=cp.asnumpy(-f.grad[:,0,0])
            values[start:end,j,3]=cp.asnumpy(-(f.hess[:,0,0]+f.hess[:,0,2]))
    return tuple(values[:,:,i].copy() for i in range(4))

def ritz(p,u,v,w,weight):
    p=p-p.mean(axis=0)
    mass=weight*p.T@p; stiff=weight*(u.T@u+v.T@v)
    lam,t=la.eigh(stiff,mass)
    p=p@t; u=u@t/np.sqrt(lam); v=v@t/np.sqrt(lam); w=w@t/np.sqrt(lam)
    return p,u,v,w,lam

def build_rom(u,v,w,weight):
    mass=weight*(u.T@u+v.T@v); stiff=weight*w.T@w
    visc,t=la.eigh(stiff,mass)
    u=u@t; v=v@t; w=w@t
    gu,gv,gw=map(lambda a:cp.asarray(a,dtype=cp.float64),(u,v,w))
    k=u.shape[1]; tensor=np.empty((k,k,k))
    for z in range(k):
        tensor[:,:,z]=cp.asnumpy(weight*(gu.T@(gv*gw[:,z,None])-gv.T@(gu*gw[:,z,None])))
    del gu,gv,gw
    return u,v,w,visc,tensor

def integrate(u,v,w,visc,tensor,gt,dt=.01):
    weight=(2*np.pi/128)**2
    c=weight*(u.T@gt['velocity_u'][0].ravel()+v.T@gt['velocity_v'][0].ravel())
    def rhs(a): return np.einsum('ijk,j,k->i',tensor,a,a,optimize=False)
    rng=np.random.default_rng(42); probe=rng.normal(size=len(c))
    energy=float(abs(probe@rhs(probe))/(np.linalg.norm(probe)**3+1e-30))
    assert energy<1e-9,energy
    outputs=[]; current=0.
    for target in gt['time']:
        steps=max(0,int(np.ceil((target-current)/dt-1e-10)))
        if steps:
            h=(target-current)/steps; linear=-gt['nu']*visc
            e=np.exp(h*linear); e2=np.exp(.5*h*linear)
            roots=np.exp(1j*np.pi*(np.arange(1,33)-.5)/32)
            lr=h*linear[:,None]+roots
            q=h*np.real(np.mean((np.exp(lr/2)-1)/lr,axis=-1))
            f1=h*np.real(np.mean((-4-lr+np.exp(lr)*(4-3*lr+lr*lr))/lr**3,axis=-1))
            f2=h*np.real(np.mean((2+lr+np.exp(lr)*(-2+lr))/lr**3,axis=-1))
            f3=h*np.real(np.mean((-4-3*lr-lr*lr+np.exp(lr)*(4-lr))/lr**3,axis=-1))
            for _ in range(steps):
                nv=rhs(c); a=e2*c+q*nv; na=rhs(a)
                b=e2*c+q*na; nb=rhs(b); d=e2*a+q*(2*nb-nv)
                c=e*c+f1*nv+2*f2*(na+nb)+f3*rhs(d)
        current=target; outputs.append(c.copy())
    coefficients=np.array(outputs)
    U=(coefficients@u.T).reshape(-1,128,128)
    V=(coefficients@v.T).reshape(-1,128,128)
    W=(coefficients@w.T).reshape(-1,128,128)
    assert np.isfinite(W).all()
    return U,V,W,coefficients,energy

def diagnostics(p,u,v,w,lam):
    n=128; weight=(2*np.pi/n)**2
    freq=np.fft.fftfreq(n)*n; kx,ky=np.meshgrid(freq,freq); k2=kx*kx+ky*ky
    def fft(a): return np.fft.fft2(a.reshape(n,n,-1),axes=(0,1))
    def inv(a): return np.fft.ifft2(a,axes=(0,1)).real.reshape(n*n,-1)
    fu,fv=fft(u),fft(v)
    lu,lv=inv(k2[...,None]*fu),inv(k2[...,None]*fv)
    residual=np.sqrt(np.sum((lu-u*lam)**2+(lv-v*lam)**2,axis=0))/lam/np.sqrt(np.sum(u*u+v*v,axis=0))
    div=inv(1j*kx[...,None]*fu+1j*ky[...,None]*fv)
    # Multiplicities: compare projectors for every complete exact shell below
    # the truncation boundary, never compare a chosen individual eigenvector.
    ref,_,_,_,exact=analytic_fields(len(lam))
    subspaces=[]
    for ev in np.unique(exact):
        if ev==exact[-1]: continue
        idx=np.flatnonzero(exact==ev)
        cross=weight*ref[:,idx].T@p[:,idx]
        error=np.sqrt(max(0.,2*len(idx)-2*np.sum(cross*cross)))
        subspaces.append(dict(eigenvalue=int(ev),multiplicity=len(idx),projector_frobenius=float(error)))
    return dict(potential_eigenvalues=lam.tolist(),exact_eigenvalues=exact.tolist(),
        eigenvalue_relative_error_mean=float(np.mean(abs(lam-exact)/exact)),
        eigenvalue_relative_error_max=float(np.max(abs(lam-exact)/exact)),
        velocity_gram_frobenius=float(np.linalg.norm(weight*(u.T@u+v.T@v)-np.eye(len(lam)))),
        potential_gram_frobenius=float(np.linalg.norm(weight*p.T@p-np.eye(len(lam)))),
        velocity_eigen_residual_mean=float(residual.mean()),velocity_eigen_residual_max=float(residual.max()),
        velocity_eigen_residuals=residual.tolist(),divergence_fft_rms=float(np.sqrt(np.mean(div*div))),
        canonical_mean_max=float(np.max(abs(p.mean(axis=0)))),shell_subspaces=subspaces)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--analytic-control',action='store_true'); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    gt=load_gt(); started=time.perf_counter(); weight=(2*np.pi/128)**2
    if args.analytic_control:
        p,u,v,w,lam=analytic_fields(64); metadata={'source':'analytic Fourier control'}
    else:
        metadata=json.loads(args.checkpoint.with_suffix('.json').read_text())
        cfg=cfg_namespace(metadata['config'])
        assert cfg.geometry_mode=='periodic_box'
        assert cfg.gauge_canonicalization=='mean_zero_before_sort_normalize_mgs'
        with np.load(args.checkpoint) as z: arrays={key:z[key] for key in z.files}
        model=HardcodedModel(cfg,arrays)
        x,y=np.meshgrid(gt['x'],gt['y'],indexing='xy')
        p,u,v,w=evaluate_fields(model,cfg,np.column_stack((x.ravel(),y.ravel())))
        # Check the continuous seam directly using analytic Jet2 derivatives.
        edge=np.linspace(0,2*np.pi,128,endpoint=False)
        points=np.concatenate([np.column_stack((np.zeros(128),edge)),np.column_stack((np.full(128,2*np.pi),edge)),
                               np.column_stack((edge,np.zeros(128))),np.column_stack((edge,np.full(128,2*np.pi)))])
        seam=evaluate_fields(model,cfg,points)
        metadata['periodic_seam_max_abs']={name:float(max(np.max(abs(f[:128]-f[128:256])),np.max(abs(f[256:384]-f[384:]))))
                                                  for name,f in zip(('psi','u','v','omega'),seam)}
        original=p.copy(); shifted=p+np.random.default_rng(3).normal(size=cfg.modes)
        metadata['gauge_shift_canonical_max_abs']=float(np.max(abs((shifted-shifted.mean(axis=0))-(p-p.mean(axis=0)))))
        p,u,v,w,lam=ritz(p,u,v,w,weight)
        metadata['checkpoint_sha256']=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report=diagnostics(p,u,v,w,lam)
    print('Potential spectrum diagnosed; assembling full Galerkin operator',flush=True)
    ru,rv,rw,visc,tensor=build_rom(u,v,w,weight)
    U,V,W,coeff,energy=integrate(ru,rv,rw,visc,tensor,gt)
    result,vm,wm=metrics(gt,U,V,W)
    result.update(name='GAUGE-CANONICAL EIGENFLUID' if not args.analytic_control else 'Analytic Fourier control',
        epoch=metadata.get('epoch'),seconds=time.perf_counter()-started,modes=len(lam),
        dynamics='Full velocity mass and curl stiffness; rotational Galerkin; ETDRK4 dt<=0.01',
        initialization='Velocity L2 projection at t=0 only; no later GT injection',
        energy_skew_defect=energy,eigen_diagnostics=report,
        gauge_shift_canonical_max_abs=metadata.get('gauge_shift_canonical_max_abs'),
        periodic_seam_max_abs=metadata.get('periodic_seam_max_abs'),
        checkpoint_sha256=metadata.get('checkpoint_sha256'))
    if args.analytic_control:
        with np.load(ROOT/'outputs/analytic/rollout_k64/fields.npz') as ref:
            delta=max(np.max(abs(U-ref['u'])),np.max(abs(V-ref['v'])))
        result['independent_fft_rollout_max_abs_difference']=float(delta)
        assert delta<1e-9,delta
    np.savez_compressed(out/'fields.npz',time=gt['time'],u=U,v=V,omega=W,coefficients=coeff)
    np.savez_compressed(out/'basis.npz',psi=p,u=u,v=v,omega=w,potential_eigenvalues=lam,viscous_eigenvalues=visc)
    np.savetxt(out/'mse.csv',np.column_stack((gt['time'],vm,wm)),delimiter=',',header='time,velocity_mse,vorticity_mse',comments='')
    write_json(out/'summary.json',result)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    ax[0].plot(report['exact_eigenvalues'],color=BLUE,label='Analytic spectrum'); ax[0].plot(lam,color=RED,label='Potential Ritz')
    ax[0].set(xlabel='Mode rank',ylabel='Eigenvalue'); ax[0].legend()
    analytic=np.loadtxt(ROOT/'outputs/analytic/rollout_k64/mse.csv',delimiter=',',skiprows=1)
    ax[1].semilogy(gt['time'],analytic[:,1],color=BLUE,label='Analytic K=64'); ax[1].semilogy(gt['time'],vm,color=RED,label=result['name'])
    ax[1].set(xlabel='Time',ylabel='Velocity MSE'); ax[1].legend(fontsize=8)
    savefig(fig,out/'spectrum_and_rollout')
    fig,axes=plt.subplots(3,4,figsize=(12,8.2),layout='constrained')
    with np.load(ROOT/'outputs/analytic/rollout_k64/fields.npz') as z: analytic_w=z['omega']
    for row,(name,data) in enumerate([('DNS',gt['vorticity']),('Canonical Laplacian K=64',analytic_w),('GAUGE-CANONICAL EIGENFLUID',W)]):
        for col,i in enumerate((0,20,40,60)):
            im=axes[row,col].imshow(data[i],origin='lower',extent=[0,2*np.pi,0,2*np.pi],cmap=CMAP,norm=TwoSlopeNorm(vmin=-.12,vcenter=0,vmax=1.))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(['t = 0',r'$t=2\pi$',r'$t=4\pi$',r'$t=6\pi$'][col])
            if col==0: axes[row,col].set_ylabel(name,fontsize=9)
    fig.colorbar(im,ax=axes,shrink=.75,label='Vorticity'); savefig(fig,out/'vortex_merger_comparison')
    print(json.dumps({key:value for key,value in result.items() if key!='eigen_diagnostics'},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
