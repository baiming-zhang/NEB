"""Native analytic P/PH full-quadrature weak-Hodge ROM.

Generate all analytic jets and complete global mass/curl moments, then advance
independent paired initial fields with original full-mass IMEX midpoint.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('operators','dynamics'),required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--harmonic-checkpoint',type=Path)
    parser.add_argument('--causal-order',type=Path)
    parser.add_argument('--total-K',type=int)
    parser.add_argument('--quadrature',type=Path)
    parser.add_argument('--fields',type=Path)
    parser.add_argument('--field-ensemble',choices=('quintic','compact-vortex'),default='quintic')
    parser.add_argument('--vortex-config',type=Path)
    parser.add_argument('--operators',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dt',type=float,default=.001)
    parser.add_argument('--final-time',type=float,default=1.)
    parser.add_argument('--snapshot-dt',type=float,default=.02)
    parser.add_argument('--viscosity',type=float,default=1/750)
    a=parser.parse_args(argv)
    if a.mode=='operators' and any(x is None for x in (a.checkpoint,a.quadrature,a.fields)):parser.error('operators requires --checkpoint --quadrature --fields')
    if a.mode=='dynamics' and a.operators is None:parser.error('dynamics requires --operators')
    if min(a.dt,a.final_time,a.snapshot_dt)<=0 or a.viscosity<0:parser.error('Invalid temporal/viscosity parameters')
    import numpy as np,cupy as cp
    from ..runtime import guard
    from .analytic_rom_full_quadrature import full_linear_moments,FullQuadratureAnalyticROM
    a.output.mkdir(parents=True,exist_ok=False);stop=guard(cp,a.output);started=time.perf_counter()
    try:
        if a.mode=='operators':
            base=Path(__file__).resolve().parents[2]/'2d/_vendor';top=base/'ICLR2027_campaign/cases/topology3';learned=top/'learned_stable_v2'
            os.environ['NEB_2D_VENDOR_ROOT']=str(base)
            sys.path[:0]=[str(learned/'positive/src'),str(learned/'positive/scripts'),str(learned/'harmonic/scripts'),str(learned)]
            from neb2d.basisnet.metaball2d import hardcoded_backend as be
            from harmonic_jet2 import learned_harmonic_fields
            from train_harmonic import load_checkpoint
            meta=json.loads(a.checkpoint.with_suffix('.json').read_text());assert meta['epoch']==2000
            from train_hardcoded import validate_config,FP32_CONFIG_KEYS
            original=dict(meta['config']);validate_config(original)
            config=dict(original)
            for key in (*FP32_CONFIG_KEYS,'mlp_compute_dtype'):config[key]='float64'
            be.F32=cp.float64;be._compute_dtype=lambda cfg:cp.float64
            cfg=be.cfg_namespace(config)
            with np.load(a.checkpoint) as z:model=be.HardcodedModel(cfg,{key:z[key] for key in z.files})
            hmodel=None;h=0
            if a.harmonic_checkpoint:
                hm,ha=load_checkpoint(a.harmonic_checkpoint);assert hm['epoch']==2000
                hconfig=dict(hm['config'])
                for key in (*FP32_CONFIG_KEYS,'mlp_compute_dtype'):hconfig[key]='float64'
                import harmonic_jet2
                original_xy=harmonic_jet2.xy_jets2
                harmonic_jet2.xy_jets2=lambda points,dtype:original_xy(points,cp.float64)
                hc=be.cfg_namespace(hconfig);assert hc.subcase==cfg.subcase
                hmodel=be.HardcodedModel(hc,ha);h=hc.modes
            K=a.total_K or cfg.modes+h;positive=K-h
            assert 0<positive<=cfg.modes and 0<=h<K
            order=np.arange(cfg.modes)
            if positive<cfg.modes:
                if not a.causal_order:raise ValueError('Actual final causal order required for P(K-h)H(h)')
                order=np.asarray(json.loads(a.causal_order.read_text())['details'][0]['causal_order'])
                assert sorted(order.tolist())==list(range(cfg.modes))
            selected=order[:positive]
            with np.load(a.quadrature) as z:points=z['points'];weights=z['weights']
            assert points.shape==(len(weights),2) and np.isfinite(points).all() and np.isfinite(weights).all() and (weights>0).all()
            g=np.lib.format.open_memmap(a.output/'gradients.npy',mode='w+',dtype='<f8',shape=(len(points),K,2))
            omega=np.lib.format.open_memmap(a.output/'omega.npy',mode='w+',dtype='<f8',shape=(len(points),K))
            fields=[];velocities=None;cells=None
            if a.field_ensemble=='compact-vortex':
                from .compact_fields import load
                velocities,cells=load(a.fields,cfg.subcase,a.quadrature,a.vortex_config)
            else:
                path=top/'random_streamfunction_20260928/preview_random_streamfunction.py'
                spec=importlib.util.spec_from_file_location('neb_rom_quintic',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
                fields=[]
                for sample in range(20):
                    field=mod.RandomField(cfg.subcase,sample)
                    with np.load(a.fields/f'sample{sample:02d}.npz') as z:
                        assert np.array_equal(field.seed,z['seed_hierarchy']) and np.array_equal(field.raw,z['raw_lattice']) and np.array_equal(field.constants,z['trace_constants'])
                        field.scale=float(z['global_scale'])
                    fields.append(field)
            rhs=cp.zeros((K,20),dtype='f8')
            for first in range(0,len(points),1024):
                x=points[first:first+1024];G=cp.empty((len(x),cfg.modes,2),dtype='f8');H=cp.empty((len(x),cfg.modes,3),dtype='f8')
                for mode,f,_ in model.forward_stream_jet2(cfg,x,0.,0.,False):
                    i=int(mode.mode_index);G[:,i]=f.grad[:,0];H[:,i]=f.hess[:,0]
                G=G[:,selected];H=H[:,selected]
                if hmodel is not None:
                    hg=cp.empty((len(x),h,2),dtype='f8');hh=cp.empty((len(x),h,3),dtype='f8')
                    for j,(_,f,_) in enumerate(learned_harmonic_fields(hmodel,hc,x,False)):hg[:,j]=f.grad[:,0];hh[:,j]=f.hess[:,0]
                    G=cp.concatenate((G,hg),axis=1);H=cp.concatenate((H,hh),axis=1)
                assert bool(cp.isfinite(G).all()) and bool(cp.isfinite(H).all())
                O=-(H[:,:,0]+H[:,:,2]);g[first:first+len(x)]=cp.asnumpy(G);omega[first:first+len(x)]=cp.asnumpy(O)
                if velocities is None:target=cp.asarray(np.stack([f.evaluate(x)[:,1:3] for f in fields],axis=2))
                else:
                    from .compact_fields import targets
                    target=cp.asarray(targets(velocities,cells,first,len(x)))
                w=cp.asarray(weights[first:first+len(x)])
                rhs+=G[:,:,1].T@(w[:,None]*target[:,0])-G[:,:,0].T@(w[:,None]*target[:,1])
            g.flush();omega.flush();M,D=full_linear_moments(g,omega,weights)
            c0=cp.linalg.solve(M,rhs);assert bool(cp.isfinite(c0).all())
            np.savez(a.output/'mass.npz',mass=cp.asnumpy(M),curl_stiffness=cp.asnumpy(D),initial_coefficients=cp.asnumpy(c0),rhs=cp.asnumpy(rhs),weights=weights,points=points)
            (a.output/'operators.json').write_text(json.dumps(dict(K=K,positive=positive,harmonic=h,selected_positive_modes=selected.tolist(),case=cfg.subcase,epoch=2000,points=len(points),samples=20,field_ensemble=a.field_ensemble,representation='raw analytic streamfunction gradient/curl; full global mass',checkpoint=str(a.checkpoint),harmonic_checkpoint=str(a.harmonic_checkpoint) if h else None,actual_seconds=time.perf_counter()-started),indent=2))
        else:
            meta=json.loads((a.operators/'operators.json').read_text())
            with np.load(a.operators/'mass.npz') as z:M=z['mass'];D=z['curl_stiffness'];c0=z['initial_coefficients'];w=z['weights']
            g=np.load(a.operators/'gradients.npy',mmap_mode='r');o=np.load(a.operators/'omega.npy',mmap_mode='r')
            rom=FullQuadratureAnalyticROM(M,D,g,o,w,nu=a.viscosity)
            t0=time.perf_counter();frames=rom.trajectory(c0,a.dt,a.final_time,a.snapshot_dt);cp.cuda.Stream.null.synchronize();seconds=time.perf_counter()-t0
            np.savez(a.output/'trajectory.npz',t=np.arange(len(frames))*a.snapshot_dt,coefficients=cp.asnumpy(frames))
            (a.output/'trajectory.json').write_text(json.dumps(dict(K=meta['K'],case=meta['case'],samples=20,dt=a.dt,T=a.final_time,snapshot_dt=a.snapshot_dt,viscosity=a.viscosity,actual_native_integration_seconds=seconds,actual_wall_seconds=time.perf_counter()-started,method='original analytic weak-Hodge fullmass IMEX midpoint; exact all-point tensor factorization'),indent=2))
    finally:stop.set()
if __name__=='__main__':main()
