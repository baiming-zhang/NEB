"""Matched physical rollout and native-objective diagnostics for both arms.

The ROM is assembled DIRECTLY from the learned velocity/vorticity fields.
No potential centering, potential Ritz rotation, clipping of eigenvalues,
or use of the training Rayleigh values enters the physical dynamics.
Centering is used in a separate diagnostic of the learned positive span.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
from verify_analytic import ROOT,load_gt,metrics,write_json,analytic_fields
from evaluate_neural import evaluate_fields,build_rom,integrate,HardcodedModel,cfg_namespace

OUT=ROOT/'outputs/gauge_ablation'

def native_diagnostics(p,u,v,w,enabled,weight):
    area=(2*np.pi)**2
    used=p-p.mean(axis=0) if enabled else p.copy()
    mass=weight*used.T@used
    stiff=weight*(u.T@u+v.T@v)
    lam,t=la.eigh(stiff,mass)
    phi=used@t
    gauge_fraction=area*np.square(phi.mean(axis=0))
    # A native raw near-constant mode can have a vanishing potential
    # Rayleigh value even if its tiny nonconstant velocity is high frequency.
    nu=u@t; nv=v@t; nw=w@t
    velocity_mass=weight*np.sum(nu*nu+nv*nv,axis=0)
    curl_energy=weight*np.sum(nw*nw,axis=0)
    velocity_rq=curl_energy/velocity_mass
    exact=analytic_fields(p.shape[1])[4].astype(float)
    centered=p-p.mean(axis=0)
    plam,pt=la.eigh(stiff,weight*centered.T@centered)
    ph=centered@pt
    ref=analytic_fields(p.shape[1])[0]
    subspaces=[]
    for ev in np.unique(exact)[:-1]:
        idx=np.flatnonzero(exact==ev)
        cross=weight*ref[:,idx].T@ph[:,idx]
        subspaces.append(dict(eigenvalue=int(ev),multiplicity=len(idx),
            projector_error=float(np.sqrt(max(0.,2*len(idx)-2*np.sum(cross*cross))))))
    # Poincare on the zero-mean periodic space: R >= 1. Do not apply
    # the same lower bound to the unprojected scalar space with its zero mode.
    return dict(native_potential_ritz_eigenvalues=lam.tolist(),
        native_min_rayleigh=float(lam[0]),native_mean_rayleigh=float(lam.mean()),
        native_ritz_gauge_energy_fractions=gauge_fraction.tolist(),
        native_max_gauge_energy_fraction=float(gauge_fraction.max()),
        native_modes_below_positive_spectral_floor=int(np.sum(lam<.5)),
        native_velocity_rayleigh=velocity_rq.tolist(),
        native_positive_target_eigenvalue_relative_error=float(np.mean(abs(lam-exact)/exact)),
        centered_span_eigenvalues=plam.tolist(),
        centered_span_eigenvalue_relative_error_mean=float(np.mean(abs(plam-exact)/exact)),
        centered_span_eigenvalue_relative_error_max=float(np.max(abs(plam-exact)/exact)),
        centered_span_shell_subspaces=subspaces,
        note='Native diagnostics use the training arm projector. Centered-span values are offline diagnostics only; no potential projection enters ROM assembly.')

def evaluate(checkpoint,out,save_fields=True,dt=.01):
    started=time.perf_counter(); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(checkpoint.with_suffix('.json').read_text())
    cfg=cfg_namespace(meta['config']); enabled=bool(getattr(cfg,'gauge_fixing_enabled',True))
    assert cfg.geometry_mode=='periodic_box'
    assert cfg.gauge_canonicalization==('mean_zero_before_sort_normalize_mgs' if enabled else 'none')
    with np.load(checkpoint) as z: arrays={k:z[k] for k in z.files}
    model=HardcodedModel(cfg,arrays); gt=load_gt()
    x,y=np.meshgrid(gt['x'],gt['y'],indexing='xy')
    p,u,v,w=evaluate_fields(model,cfg,np.column_stack((x.ravel(),y.ravel())))
    weight=(2*np.pi/128)**2
    diag=native_diagnostics(p,u,v,w,enabled,weight)
    print(f"{('canonical' if enabled else 'raw')} epoch={meta['epoch']}: native min R={diag['native_min_rayleigh']:.8g}, max gauge fraction={diag['native_max_gauge_energy_fraction']:.8g}",flush=True)
    # Rescale individual physical columns to improve conditioning; this is
    # an invertible velocity coordinate change applied identically to both.
    norm=np.sqrt(weight*np.sum(u*u+v*v,axis=0))
    if np.any(norm<=0): raise ValueError('An exact null-lift mode requires explicit rank handling')
    u=u/norm; v=v/norm; w=w/norm
    mass=weight*(u.T@u+v.T@v)
    mv=np.linalg.eigvalsh(mass)
    rank_tol=np.finfo(float).eps*max(2*len(u),len(mv))*mv[-1]
    resolved=int(np.sum(mv>rank_tol))
    if resolved!=cfg.modes:
        raise ValueError(f'Physical mass rank={resolved}, expected {cfg.modes}; inspect before evaluating')
    ru,rv,rw,visc,tensor=build_rom(u,v,w,weight)
    U,V,W,coeff,skew=integrate(ru,rv,rw,visc,tensor,gt,dt=dt)
    result,vm,wm=metrics(gt,U,V,W)
    # Projection-only error per frame isolates representational capacity
    # from rollout dynamics. It is a diagnostic, never used to reset rollout.
    fits=weight*(gt['velocity_u'].reshape(61,-1)@ru+gt['velocity_v'].reshape(61,-1)@rv)
    fit_u=(fits@ru.T).reshape(61,128,128); fit_v=(fits@rv.T).reshape(61,128,128)
    projection=.5*np.mean((fit_u-gt['velocity_u'])**2+(fit_v-gt['velocity_v'])**2,axis=(1,2))
    result.update(arm='canonical' if enabled else 'raw',gauge_fixing_enabled=enabled,
        epoch=int(meta['epoch']),modes=int(cfg.modes),seconds=time.perf_counter()-started,
        checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        physical_mass_resolved_rank=resolved,physical_mass_condition=float(mv[-1]/mv[0]),
        physical_mass_rank_tolerance=float(rank_tol),energy_skew_defect=skew,
        projection_only_mean_velocity_mse=float(projection.mean()),projection_only_final_velocity_mse=float(projection[-1]),
        dynamics='identical full velocity mass/curl-stiffness rotational Galerkin; ETDRK4',
        max_dt=dt,potential_canonicalization_in_rom=False,native_diagnostics=diag)
    if enabled and meta['epoch']==2000:
        prior=ROOT/'outputs/neural_epoch2000_evaluation/fields.npz'
        with np.load(prior) as f:
            delta=max(np.max(abs(U-f['u'])),np.max(abs(V-f['v'])))
        result['earlier_canonical_coordinate_check_max_abs']=float(delta)
        assert delta<1e-8,delta
    write_json(out/'summary.json',result)
    np.savetxt(out/'mse.csv',np.column_stack((gt['time'],vm,wm,projection)),delimiter=',',
        header='time,velocity_mse,vorticity_mse,projection_only_velocity_mse',comments='')
    if save_fields:
        np.savez_compressed(out/'fields.npz',time=gt['time'],u=U,v=V,omega=W,coefficients=coeff)
        np.savez_compressed(out/'physical_basis.npz',u=ru,v=rv,omega=rw,viscous_eigenvalues=visc)
    print(json.dumps({k:v for k,v in result.items() if k!='native_diagnostics'},indent=2),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--dt',type=float,default=.01)
    args=parser.parse_args()
    evaluate(args.checkpoint,args.output,not args.compact,args.dt)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
