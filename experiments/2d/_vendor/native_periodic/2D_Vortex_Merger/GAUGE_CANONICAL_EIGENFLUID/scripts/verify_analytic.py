"""Gauge-class checks and actual Fourier-Galerkin vortex-merger rollouts.

The periodic specialization is P psi = psi - mean(psi). A positive Fourier
basis already satisfies it. Gauge shifting a potential is NOT another
positive Laplacian eigenfunction, although it represents the same velocity.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT.parent
WORKSPACE = CASE.parents[1]
GT = CASE / 'outputs/gt_128_re350_l0p90_t6pi/vortex_merger_gt.h5'
THEORY = WORKSPACE / 'ICLR2027_Potential_Eigenfluids/ICLR2027_Potential_Eigenfluids.tex'
OUT = ROOT / 'outputs/analytic'
RED, BLUE = '#E64B35', '#4DBBD5'
CMAP = LinearSegmentedColormap.from_list('nature_rb', ['#3C5488', '#F7F7F7', RED])
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def savefig(fig, stem):
    fig.savefig(str(stem)+'.png', dpi=190, bbox_inches='tight')
    fig.savefig(str(stem)+'.pdf', bbox_inches='tight')
    plt.close(fig)

def load_gt():
    with h5py.File(GT, 'r') as f:
        data = {key: f[key][:] for key in ('time','x','y','vorticity','streamfunction','velocity_u','velocity_v')}
        data['nu'] = float(f.attrs['kinematic_viscosity'])
    return data

def waves(k):
    """Real sine/cosine pairs; deterministic tie break inside multiplicities."""
    assert k % 2 == 0
    extent = int(np.ceil(np.sqrt(k)))
    pairs = [(x,y) for x in range(extent+1) for y in range(-extent,extent+1)
             if x > 0 or (x == 0 and y > 0)]
    return np.array(sorted(pairs, key=lambda p:(p[0]**2+p[1]**2,p[0],p[1]))[:k//2])

def analytic_fields(k, n=128):
    axis=np.arange(n)*2*np.pi/n
    x,y=np.meshgrid(axis,axis,indexing='xy')
    p=waves(k)
    phase=x.ravel()[:,None]*p[:,0]+y.ravel()[:,None]*p[:,1]
    scale=np.sqrt(2)/(2*np.pi)
    phi=np.stack((np.cos(phase),np.sin(phase)),axis=-1).reshape(n*n,k)*scale
    deriv=np.stack((-np.sin(phase),np.cos(phase)),axis=-1).reshape(n*n,k)*scale
    kx=np.repeat(p[:,0],2); ky=np.repeat(p[:,1],2)
    lam=kx*kx+ky*ky
    return phi, deriv*ky/np.sqrt(lam), -deriv*kx/np.sqrt(lam), phi*np.sqrt(lam), lam

def gauge_checks():
    OUT.mkdir(parents=True,exist_ok=True)
    n=128; length=2*np.pi; weight=(length/n)**2
    a=np.arange(n)*length/n; x,y=np.meshgrid(a,a)
    phi=np.sqrt(2)/length*np.cos(x+2*y)
    grad2=2/length**2*5*np.sin(x+2*y)**2
    alpha=np.array([0.,.1,1.,10.,100.,1000.])
    raw=[]; fixed=[]; field_error=[]
    for c in alpha:
        shifted=phi+c/length
        canonical=shifted-shifted.mean()
        raw.append(float(grad2.sum()/np.square(shifted).sum()))
        fixed.append(float(grad2.sum()/np.square(canonical).sum()))
        field_error.append(float(np.max(np.abs(canonical-phi))))
    np.testing.assert_allclose(raw,5/(1+alpha**2),rtol=1e-12)
    np.testing.assert_allclose(fixed,5,rtol=1e-12)
    p,u,v,w,lam=analytic_fields(64)
    constants=np.random.default_rng(37).normal(size=64)*10/length
    shifted=p+constants
    canonical=shifted-shifted.mean(axis=0)
    gram=weight*canonical.T@canonical
    ug=weight*(u.T@u+v.T@v)
    np.testing.assert_allclose(canonical,p,atol=1e-12)
    np.testing.assert_allclose(gram,np.eye(64),atol=1e-12)
    np.testing.assert_allclose(ug,np.eye(64),atol=1e-12)
    # Independent FFT differentiation, including C* C and lifted velocity.
    freq=np.fft.fftfreq(n)*n; kx,ky=np.meshgrid(freq,freq); k2=kx*kx+ky*ky
    pf=np.fft.fft2(canonical.reshape(n,n,64),axes=(0,1))
    lap=np.fft.ifft2(k2[...,None]*pf,axes=(0,1)).real.reshape(n*n,64)
    pu=np.fft.ifft2(1j*ky[...,None]*pf,axes=(0,1)).real.reshape(n*n,64)/np.sqrt(lam)
    pv=np.fft.ifft2(-1j*kx[...,None]*pf,axes=(0,1)).real.reshape(n*n,64)/np.sqrt(lam)
    np.testing.assert_allclose(pu,u,atol=1e-11)
    np.testing.assert_allclose(pv,v,atol=1e-11)
    residual=np.linalg.norm(lap-canonical*lam,axis=0)/(lam*np.linalg.norm(canonical,axis=0))
    # Scalar Dirichlet negative control: subtraction of the mean violates
    # the fixed zero boundary trace for sin(x)sin(y) on [0,pi]^2.
    dirichlet_mean=4/np.pi**2
    moments=[float(np.sum(np.array([1.,.25])*np.array([1.,5.])**r)) for r in range(4)]
    hierarchy=np.array(moments[1:])/moments[:-1]
    assert np.all(np.diff(hierarchy)>0)
    summary=dict(analytic_mode='sqrt(2)/(2*pi) * cos(x+2*y)',eigenvalue=5.,
        alpha=alpha.tolist(),raw_rayleigh=raw,canonical_rayleigh=fixed,
        max_canonical_field_difference=max(field_error),
        k64_potential_gram_frobenius=float(np.linalg.norm(gram-np.eye(64))),
        k64_velocity_gram_frobenius=float(np.linalg.norm(ug-np.eye(64))),
        k64_fft_eigen_residual_max=float(residual.max()),
        k64_gauge_shift_velocity_max_abs=float(max(np.max(abs(pu-u)),np.max(abs(pv-v)))),
        k64_raw_shifted_gram_defect=float(np.linalg.norm(weight*shifted.T@shifted-np.eye(64))),
        dirichlet_control=dict(projector='identity on H_0^1',constant_shift_admissible=False,
            incorrectly_centered_boundary_value=-dirichlet_mean),
        moment_hierarchy=hierarchy.tolist(),
        harmonic_velocity_sector=dict(dimension=2,basis='constant x/y velocities',gt_coefficients='measured separately in rollout'),
        interpretation='A positive Fourier basis is already canonical. Gauge fixing protects potential norms and the objective; it does not change the represented velocity.')
    write_json(OUT/'gauge_checks.json',summary)
    np.savez_compressed(OUT/'analytic_k64_basis.npz',psi=p,u=u,v=v,omega=w,eigenvalues=lam,wave_pairs=waves(64))
    with (OUT/'gauge_shift.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['alpha','raw_rayleigh','canonical_rayleigh','max_field_error'])
        writer.writerows(zip(alpha,raw,fixed,field_error))
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    aa=np.logspace(-2,3,150)
    ax[0].loglog(aa,5/(1+aa**2),color=RED,label='Raw potential quotient')
    ax[0].loglog(aa,np.full_like(aa,5),color=BLUE,label='Gauge-canonical quotient')
    ax[0].set(xlabel=r'Gauge amplitude $\alpha$',ylabel='Rayleigh quotient',title='Same velocity, different raw objective')
    ax[0].legend(fontsize=9)
    t=np.linspace(0,10,200); nu=.1
    ax[1].plot(t,np.exp(-nu*5*t),color=BLUE,label='Exact / canonical')
    ax[1].plot(t,np.exp(-nu*5/101*t),color=RED,label=r'Using raw quotient, $\alpha=10$')
    ax[1].set(xlabel='Time',ylabel='Velocity-mode amplitude',title='Single-mode viscous-decay check')
    ax[1].legend(fontsize=9)
    savefig(fig,OUT/'gauge_necessity')
    return summary

def dns_module():
    spec=importlib.util.spec_from_file_location('gauge_reference_dns',CASE/'scripts/generate_gt.py')
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod

def metrics(gt,u,v,omega):
    velocity=.5*np.mean((u-gt['velocity_u'])**2+(v-gt['velocity_v'])**2,axis=(1,2))
    vort=np.mean((omega-gt['vorticity'])**2,axis=(1,2))
    return dict(velocity_mse_initial=float(velocity[0]),velocity_mse_final=float(velocity[-1]),
        velocity_mse_mean=float(velocity.mean()),vorticity_mse_initial=float(vort[0]),
        vorticity_mse_final=float(vort[-1]),vorticity_mse_mean=float(vort.mean())),velocity,vort

def rollout(k,gt,dt=.01,n=64,output_name=None):
    started=time.perf_counter(); pairs=waves(k)
    solver=dns_module().VorticityDNS(n,2*np.pi,gt['nu'],dt)
    keep=np.zeros((n,n),bool)
    for x,y in pairs:
        keep[y%n,x%n]=keep[-y%n,-x%n]=True
    if np.any(keep & ~solver.dealias):
        raise ValueError('Increase collocation grid for this spectral cutoff')
    solver.dealias=keep
    state=np.zeros((n,n),complex); source=np.fft.fft2(gt['vorticity'][0])
    for x,y in np.concatenate((pairs,-pairs)):
        state[y%n,x%n]=source[y%128,x%128]*(n/128)**2
    # Mean flow is the separate harmonic velocity sector on the torus.
    mean_u=float(gt['velocity_u'][0].mean()); mean_v=float(gt['velocity_v'][0].mean())
    assert max(abs(mean_u),abs(mean_v))<1e-12
    fields=[]; current=0.
    for i,target in enumerate(gt['time']):
        steps=max(0,int(np.ceil((target-current)/dt-1e-10)))
        if steps:
            step=(target-current)/steps
            solver.dt=step; solver._build_etdrk4_coefficients(-gt['nu']*solver.k2)
            for _ in range(steps): state=solver.step(state)
        current=target
        out=np.zeros((128,128),complex)
        for x,y in np.concatenate((pairs,-pairs)):
            out[y%128,x%128]=state[y%n,x%n]*(128/n)**2
        f=np.fft.fftfreq(128)*128; kx,ky=np.meshgrid(f,f); k2=kx*kx+ky*ky
        inv=np.divide(1.,k2,out=np.zeros_like(k2),where=k2>0)
        psi=out*inv
        fields.append((np.fft.ifft2(out).real,np.fft.ifft2(1j*ky*psi).real+mean_u,
                       np.fft.ifft2(-1j*kx*psi).real+mean_v))
        if i%20==0: print(f'Analytic K={k}: {i+1}/61 frames',flush=True)
    omega,u,v=np.asarray(fields).transpose(1,0,2,3)
    result,vm,wm=metrics(gt,u,v,omega)
    result.update(modes=k,seconds=time.perf_counter()-started,collocation_grid=n,
                  time_step_max=dt,mean_velocity=[mean_u,mean_v],
                  basis='analytic mean-zero Fourier; sine/cosine pairs',
                  dynamics='dealiased Fourier-Galerkin ETDRK4; initialized only at t=0',
                  gauge_projector='zero Fourier coefficient fixed to zero',
                  standard_positive_fourier_is_already_canonical=True)
    dest=OUT/(output_name or f'rollout_k{k}'); dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'fields.npz',time=gt['time'],omega=omega,u=u,v=v,wave_pairs=pairs)
    write_json(dest/'summary.json',result)
    np.savetxt(dest/'mse.csv',np.column_stack((gt['time'],vm,wm)),delimiter=',',
               header='time,velocity_mse,vorticity_mse',comments='')
    return result,omega,vm

def main():
    start=time.perf_counter(); gauge=gauge_checks(); gt=load_gt()
    results={}; fields={}; errors={}
    for k in (64,1024):
        results[str(k)],fields[k],errors[k]=rollout(k,gt)
    fig,axes=plt.subplots(3,4,figsize=(12,8.2),layout='constrained')
    for row,(name,data) in enumerate([('DNS',gt['vorticity']),('Canonical Laplacian K=64',fields[64]),('Canonical Laplacian K=1024',fields[1024])]):
        for col,i in enumerate([0,20,40,60]):
            im=axes[row,col].imshow(data[i],origin='lower',extent=[0,2*np.pi,0,2*np.pi],cmap=CMAP,norm=TwoSlopeNorm(vmin=-.12,vcenter=0,vmax=1.))
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(['t = 0',r'$t=2\pi$',r'$t=4\pi$',r'$t=6\pi$'][col])
            if col==0: axes[row,col].set_ylabel(name)
    fig.colorbar(im,ax=axes,shrink=.75,label='Vorticity')
    savefig(fig,OUT/'vortex_merger_analytic_comparison')
    fig,ax=plt.subplots(figsize=(6,3.8),layout='constrained')
    for k,color in [(64,RED),(1024,BLUE)]: ax.semilogy(gt['time'],errors[k],color=color,label=f'Canonical Laplacian K={k}')
    ax.set(xlabel='Time',ylabel='Velocity MSE',title='Vortex merger: autonomous reduced rollout'); ax.legend()
    savefig(fig,OUT/'rollout_velocity_mse')
    provenance=dict(theory=str(THEORY),theory_sha256=hashlib.sha256(THEORY.read_bytes()).hexdigest(),
        gt=str(GT),gt_sha256=hashlib.sha256(GT.read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-start,gauge=gauge,rollouts=results)
    write_json(OUT/'summary.json',provenance)
    print(json.dumps(provenance,indent=2),flush=True)

if __name__=='__main__': main()
