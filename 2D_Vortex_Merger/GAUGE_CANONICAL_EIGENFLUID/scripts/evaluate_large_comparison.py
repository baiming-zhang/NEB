"""Scalable, matrix-free Galerkin rollout and analytic Jet3 eigenrelations."""
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
from verify_analytic import ROOT,CASE,load_gt,metrics,write_json,waves
sys.path.insert(0,str(CASE/'src'))
from eigenfluid.gen.vortex_merger2d.hardcoded_backend import HardcodedModel,cfg_namespace

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def fields(checkpoint,out):
    signature=sha(checkpoint)
    meta=json.loads(checkpoint.with_suffix('.json').read_text())
    cfg=cfg_namespace(meta['config'])
    cache=out/'jets.npy'; manifest=out/'jets.json'
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text())['sha256']==signature:
        return cfg,meta,np.load(cache,mmap_mode='r')
    with np.load(checkpoint) as a: arrays={k:a[k] for k in a.files}
    model=HardcodedModel(cfg,arrays)
    axis=np.arange(128)*2*np.pi/128; x,y=np.meshgrid(axis,axis,indexing='xy')
    pts=np.column_stack((x.ravel(),y.ravel())).astype(np.float32)
    values=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float64,shape=(6,len(pts),cfg.modes))
    for start in range(0,len(pts),1024):
        stop=min(start+1024,len(pts))
        block=cp.empty((6,stop-start,cfg.modes),dtype=cp.float32)
        for mode,f,_ in model.forward_stream(cfg,cp.asarray(pts[start:stop]),0.,0.,False):
            j=int(mode.mode_index)
            block[0,:,j]=f.value[:,0]
            block[1,:,j]=f.grad[:,0,1]
            block[2,:,j]=-f.grad[:,0,0]
            block[3,:,j]=-(f.hess[:,0,0]+f.hess[:,0,2])
            block[4,:,j]=-(f.third[:,0,1]+f.third[:,0,3])
            block[5,:,j]=f.third[:,0,0]+f.third[:,0,2]
        values[:,start:stop,:]=cp.asnumpy(block)
        values.flush()
        print(f'Jet3 fields {stop}/{len(pts)}, K={cfg.modes}',flush=True)
    write_json(manifest,dict(sha256=signature,channels=['psi','u','v','omega','minus_lap_u','minus_lap_v'],grid=128))
    return cfg,meta,values

def symmetric(a): return .5*(a+a.T)

def eig(stiff,mass):
    return la.eigh(symmetric(stiff),symmetric(mass),check_finite=True)

def relation(g,lam,t,weight):
    k=len(lam); output=[]
    for start in range(0,k,64):
        stop=min(start+64,k); rot=cp.asarray(t[:,start:stop]); ev=cp.asarray(lam[start:stop])
        p,u,v,w,lu,lv=[a@rot for a in g]
        pot_mass=weight*cp.sum(p*p,axis=0)
        vel_mass=weight*cp.sum(u*u+v*v,axis=0)
        rp=cp.sqrt(weight*cp.sum((w-p*ev)**2,axis=0))/(cp.abs(ev)*cp.sqrt(pot_mass))
        ru=cp.sqrt(weight*cp.sum((lu-u*ev)**2+(lv-v*ev)**2,axis=0))/(cp.abs(ev)*cp.sqrt(vel_mass))
        velrq=weight*cp.sum(w*w,axis=0)/vel_mass
        gauge=(2*np.pi)**2*cp.mean(p,axis=0)**2/pot_mass
        output.append(cp.asnumpy(cp.stack([rp,ru,velrq,gauge],axis=1)))
    result=np.concatenate(output)
    return dict(potential_relation_residual=result[:,0].tolist(),velocity_relation_residual=result[:,1].tolist(),
        potential_relation_residual_mean=float(result[:,0].mean()),potential_relation_residual_max=float(result[:,0].max()),
        velocity_relation_residual_mean=float(result[:,1].mean()),velocity_relation_residual_max=float(result[:,1].max()),
        velocity_rayleigh=result[:,2].tolist(),gauge_fraction=result[:,3].tolist(),
        modes_potential_residual_below_001=int(np.sum(result[:,0]<.01)),
        modes_velocity_residual_below_001=int(np.sum(result[:,1]<.01)))

def diagnose(g,enabled,weight,out):
    p,u,v,w,lu,lv=g
    centered=p-cp.mean(p,axis=0)
    used=centered if enabled else p
    stiff=cp.asnumpy(weight*(u.T@u+v.T@v))
    native_lam,native_t=eig(stiff,cp.asnumpy(weight*(used.T@used)))
    native=relation([used,u,v,w,lu,lv],native_lam,native_t,weight)
    if enabled:
        pos_lam,pos_t=native_lam,native_t; positive=native
    else:
        pos_lam,pos_t=eig(stiff,cp.asnumpy(weight*(centered.T@centered)))
        positive=relation([centered,u,v,w,lu,lv],pos_lam,pos_t,weight)
    exact=np.repeat(np.sum(waves(len(pos_lam))**2,axis=1),2)
    result=dict(native_eigenvalues=native_lam.tolist(),native=native,
        positive_span_eigenvalues=pos_lam.tolist(),positive_span=positive,exact_eigenvalues=exact.tolist(),
        positive_eigenvalue_relative_error_mean=float(np.mean(abs(pos_lam-exact)/exact)),
        positive_eigenvalue_relative_error_max=float(np.max(abs(pos_lam-exact)/exact)),
        equations=dict(potential='||-Delta psi - lambda psi|| / (|lambda| ||psi||)',
                       velocity='||-Delta u - lambda u|| / (|lambda| ||u||)'),
        derivatives='continuous analytic Jet3, not FFT or finite differences',
        positive_span_note='Both spans are centered for the offline positive-spectrum diagnostic only; no potential projection enters physical ROM.')
    write_json(out/'eigenrelation.json',result)
    table=np.column_stack((np.arange(1,len(pos_lam)+1),exact,native_lam,
        native['potential_relation_residual'],native['velocity_relation_residual'],native['gauge_fraction'],
        pos_lam,positive['potential_relation_residual'],positive['velocity_relation_residual']))
    np.savetxt(out/'eigenrelation_all_modes.csv',table,delimiter=',',comments='',
        header='rank,exact_lambda,native_lambda,native_psi_residual,native_u_residual,native_gauge_fraction,positive_lambda,positive_psi_residual,positive_u_residual')
    return result

def matrixfree_rollout(gu,gv,gw,visc,gt,dt=.01):
    weight=(2*np.pi/128)**2
    c=weight*(gu.T@cp.asarray(gt['velocity_u'][0].ravel())+gv.T@cp.asarray(gt['velocity_v'][0].ravel()))
    def rhs(a):
        u=gu@a; v=gv@a; w=gw@a
        return gu.T@(weight*w*v)-gv.T@(weight*w*u)
    rng=cp.asarray(np.random.default_rng(42).normal(size=len(c)))
    skew=float((cp.abs(rng@rhs(rng))/(cp.linalg.norm(rng)**3)).get())
    assert skew<1e-9,skew
    states=[]; current=0.
    for index,target in enumerate(gt['time']):
        steps=max(0,int(np.ceil((target-current)/dt-1e-10)))
        if steps:
            h=(target-current)/steps; linear=-gt['nu']*visc
            e=cp.exp(h*linear); e2=cp.exp(.5*h*linear)
            roots=cp.exp(1j*cp.pi*(cp.arange(1,33)-.5)/32)
            lr=h*linear[:,None]+roots
            q=h*cp.real(cp.mean((cp.exp(lr/2)-1)/lr,axis=-1))
            f1=h*cp.real(cp.mean((-4-lr+cp.exp(lr)*(4-3*lr+lr*lr))/lr**3,axis=-1))
            f2=h*cp.real(cp.mean((2+lr+cp.exp(lr)*(-2+lr))/lr**3,axis=-1))
            f3=h*cp.real(cp.mean((-4-3*lr-lr*lr+cp.exp(lr)*(4-lr))/lr**3,axis=-1))
            for _ in range(steps):
                nv=rhs(c); a=e2*c+q*nv; na=rhs(a)
                b=e2*c+q*na; nb=rhs(b); z=e2*a+q*(2*nb-nv)
                c=e*c+f1*nv+2*f2*(na+nb)+f3*rhs(z)
        current=target; states.append(cp.asnumpy(c))
        if index%10==0: print(f'Autonomous rollout {index+1}/61 frames',flush=True)
    states=cp.asarray(np.array(states))
    U=cp.asnumpy(states@gu.T).reshape(61,128,128)
    V=cp.asnumpy(states@gv.T).reshape(61,128,128)
    W=cp.asnumpy(states@gw.T).reshape(61,128,128)
    assert np.isfinite(W).all()
    return U,V,W,cp.asnumpy(states),skew

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--relations-only',action='store_true')
    parser.add_argument('--check-against',type=Path)
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); cfg,meta,raw=fields(args.checkpoint,out)
    enabled=bool(getattr(cfg,'gauge_fixing_enabled',True))
    g=[cp.asarray(a) for a in raw]
    weight=(2*np.pi/128)**2
    relation_report=json.loads((out/'eigenrelation.json').read_text()) if (out/'eigenrelation.json').exists() else {}
    if relation_report.get('checkpoint_sha256')!=sha(args.checkpoint):
        relation_report=diagnose(g,enabled,weight,out)
        relation_report['checkpoint_sha256']=sha(args.checkpoint)
        write_json(out/'eigenrelation.json',relation_report)
    if args.relations_only:
        print('Eigenrelation diagnostics complete',flush=True); return
    gu,gv,gw=g[1:4]
    norm=cp.sqrt(weight*cp.sum(gu*gu+gv*gv,axis=0)); gu=gu/norm; gv=gv/norm; gw=gw/norm
    mass=cp.asnumpy(weight*(gu.T@gu+gv.T@gv)); stiffness=cp.asnumpy(weight*(gw.T@gw))
    mv=la.eigvalsh(symmetric(mass)); threshold=np.finfo(float).eps*len(gu)*2*mv[-1]
    rank=int(np.sum(mv>threshold))
    if rank!=cfg.modes: raise ValueError(f'Physical mass resolves {rank}/{cfg.modes}, threshold={threshold}')
    visc,t=eig(stiffness,mass); rot=cp.asarray(t)
    gu=gu@rot; gv=gv@rot; gw=gw@rot
    del g,rot
    cp.get_default_memory_pool().free_all_blocks()
    gt=load_gt(); U,V,W,states,skew=matrixfree_rollout(gu,gv,gw,cp.asarray(visc),gt,args.dt)
    result,vm,wm=metrics(gt,U,V,W)
    result.update(arm='canonical' if enabled else 'raw',modes=cfg.modes,epoch=int(meta['epoch']),
        physical_mass_rank=rank,physical_mass_condition=float(mv[-1]/mv[0]),
        dt=args.dt,energy_skew_defect=skew,seconds=time.perf_counter()-start,
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),
        dynamics='full velocity mass/curl stiffness; matrix-free rotational Galerkin; float64 ETDRK4',
        initialization='t=0 velocity L2 projection only; no later GT injection',
        potential_projection_used_in_rom=False,
        positive_eigenvalue_relative_error_mean=relation_report['positive_eigenvalue_relative_error_mean'])
    if args.check_against:
        with np.load(args.check_against) as f: error=max(np.max(abs(U-f['u'])),np.max(abs(V-f['v'])))
        assert error<1e-8,error
        result['independent_evaluator_max_abs_velocity_difference']=float(error)
    np.savez_compressed(out/'fields.npz',u=U,v=V,omega=W,time=gt['time'],coefficients=states)
    np.savetxt(out/'mse.csv',np.column_stack((gt['time'],vm,wm)),delimiter=',',comments='',header='time,velocity_mse,vorticity_mse')
    write_json(out/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
