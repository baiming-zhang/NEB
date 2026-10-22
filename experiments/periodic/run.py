"""Periodic paper case: DNS, fresh initialization, training and full-mass rollout.

CLI help is CPU-only. Execution of train/evaluate requires native CuPy.
"""
from pathlib import Path
import argparse, ast, importlib.util, json, os, runpy, sys
BASE=Path(__file__).resolve().parents[1]/'2d/_backend'
CASE=BASE/'periodic_core/vortex_merger'
TRAIN=BASE/'flow/cases/vortex_merger/neural_training'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['dns','laplacian','prepare','train','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--gt',type=Path)
    p.add_argument('--method',choices=['Potential','Velocity','NKB'],default='Potential')
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();os.environ['NEB_2D_VENDOR_ROOT']=str(BASE)
    sys.path[:0]=[str(TRAIN),str(CASE/'src'),str(CASE/'canonical_gauge/scripts')]
    if a.stage in ('prepare','train'):
        if a.stage=='prepare':
            if not a.config:p.error('prepare requires --config')
            path=TRAIN/'prepare_gauge_canonical.py';args=['--config',str(a.config),'--output-dir',str(a.output)]
        else:
            if not a.checkpoint:p.error('train requires --checkpoint (fresh epoch-zero initialization)')
            path=TRAIN/'train_gauge_canonical.py';args=['--resume',str(a.checkpoint),'--run-dir',str(a.output)]
        sys.argv=[str(path),*args];runpy.run_path(str(path),run_name='__main__');return
    import numpy as np
    if a.stage in ('dns','laplacian'):
        dns=load(CASE/'scripts/generate_gt.py','neb_periodic_dns')
        cfg=dns.MergerConfig(grid_size=256,vortex_separation=1.35,final_time=18*np.pi,time_step=.005,output_frames=241)
        _,initial,_,_=dns.initial_vorticity(cfg)
        nu=.0010742913669168762;solver=dns.VorticityDNS(256,2*np.pi,nu,.005)
        if a.stage=='laplacian':
            pairs=np.loadtxt(Path(__file__).with_name('configs')/'fourier_pairs_K1024.csv',delimiter=',',skiprows=1,dtype=int)
            assert pairs.shape==(512,2)
            mask=np.zeros((256,256),bool)
            for kx,ky in pairs:mask[ky%256,kx%256]=True;mask[-ky%256,-kx%256]=True
            solver.dealias=mask
        state=solver.filtered_hat(initial);now=0.;rows=[];times=np.linspace(0,18*np.pi,241)
        a.output.mkdir(parents=True,exist_ok=False)
        for t in times:
            steps=max(0,int(np.ceil((t-now)/.005-1e-10)))
            if steps:
                h=(t-now)/steps;solver.dt=h;solver._build_etdrk4_coefficients(-nu*solver.k2)
                for _ in range(steps):state=solver.step(state)
            now=t;w,_,u,v=solver.fields_from_hat(state);rows.append([u[::2,::2],v[::2,::2],w[::2,::2]])
        rows=np.asarray(rows);np.savez_compressed(a.output/('dns.npz' if a.stage=='dns' else 'laplacian1024.npz'),u=rows[:,0],v=rows[:,1],omega=rows[:,2],time=times)
        (a.output/'summary.json').write_text(json.dumps(dict(separation=1.35,frames=241,K=1024,viscosity=nu,quadrature_N=256,observation_N=128,horizon=18*np.pi,dt_max=.005),indent=2));return
    if not a.checkpoint or not a.gt:p.error('evaluate requires --checkpoint and --gt directory containing dns.npz/summary.json')
    module=load(Path(__file__).with_name('paper_rollout.py'),'neb_paper_merger')
    module.SOURCE=a.gt.resolve();module.OUT=a.output.resolve();module.WORK=BASE
    module.explicit_checkpoint=a.checkpoint.resolve()
    module.main(a.method,a.seed,256)
if __name__=='__main__':main()
