"""Physical quadratic-chart LEB reconstruction on the fourth held-out domain."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(32*2**20),b''):h.update(block)
    return h.hexdigest()

def run(args):
    import psutil
    from threadpoolctl import threadpool_limits
    from skfem import MeshTet
    root=Path(args.output).resolve()
    config=read(args.config)
    with np.load(args.samples,allow_pickle=False) as z:
        geometry=z['geometry'].copy()
        for key in ('fit_points','assessment_points','fit_target','assessment_target'):
            if z[key].shape!=(16384,3) or not np.isfinite(z[key]).all():
                raise ValueError('All original finite 16384 fit and assessment points required')
    source=Path(args.samples).resolve()
    target=dict(case=4,K=256,h=.025,surface_resolution=80,geometry=geometry.tolist(),source_samples=str(source))
    binding=dict(config_sha256=sha(args.config),samples_sha256=sha(source),legacy_config=config,target_config=target)
    if args.stage=='mesh':
        root.mkdir(parents=True,exist_ok=False)
        (root/'inputs.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    elif read(root/'inputs.json')!=binding:
        raise ValueError('Stages must retain identical complete samples and configuration')
    started=time.perf_counter();done=threading.Event();failure=[]
    def guard():
        vm=psutil.virtual_memory();rss=psutil.Process().memory_info().rss
        if rss>=args.RAM_limit or vm.available<=9*2**30 or time.perf_counter()-started>=args.timeout_seconds:
            raise RuntimeError('CPU memory reserve or finite stage deadline exceeded')
    def watch():
        while not done.wait(1):
            try:guard()
            except BaseException as e:
                failure.append(str(e));os._exit(124)
    def configure(module,output):
        output.mkdir(parents=True,exist_ok=True)
        def write(name,value):
            guard();p=output/name;p.parent.mkdir(parents=True,exist_ok=True)
            temporary=p.with_name(p.name+'.tmp')
            with temporary.open('w',encoding='utf-8') as f:
                json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
            os.replace(temporary,p)
        def progress(*phase,**values):
            guard();print(json.dumps(dict(phase=phase[0] if phase else args.stage,actual_seconds=time.perf_counter()-started,**values)),flush=True)
        module.OUT=output;module.HERE=root;module.read=read;module.sha=sha;module.write=write
        module.progress=progress;module.guard=guard;module.DONE=done;module.START=started
        module.target_config=target;module.legacy_config=config
        return module
    def module(name,output):
        return configure(importlib.import_module('experiments.metaball._geometry4.'+name),output)
    guard();threading.Thread(target=watch,daemon=True).start()
    try:
        with threadpool_limits(limits=2):
            if args.stage=='mesh':
                m=module('mesh',root/'mesh');m.cfg=lambda:SimpleNamespace(**config)
                mesh=m.mesh_metaball(geometry,.025,80,m.OUT)
                np.savez(m.OUT/'mesh_arrays.npz',p=mesh.p,t=mesh.t)
                m.write('config.json',target)
            elif args.stage=='quality':
                m=module('quality',root/'geometry4_optimized');m.OLD=root/'mesh';m.main()
            elif args.stage in ('chord','chart'):
                pre=args.stage=='chord'
                m=module('chart_builder',root/('geometry4_chord_chart' if pre else 'geometry4_local_repaired_chart'))
                m.MESH=root/('geometry4_optimized' if pre else 'geometry4_local_repair')/'mesh_arrays.npz'
                m.PROJECT_VERTICES=True;m.EXTEND_INTERIOR=False;m.STRAIGHT_INTERIOR_EDGES=True;m.PRE_REPAIR=pre
                m.main()
            elif args.stage=='repair':
                m=module('repair',root/'geometry4_local_repair');m.SOURCE=root/'geometry4_optimized';m.CHART=root/'geometry4_chord_chart'
                method=read(m.CHART/'chart_method.json')
                if method['nonpositive_corners']:m.main()
                else:
                    with np.load(m.SOURCE/'mesh_arrays.npz') as z:np.savez(m.OUT/'mesh_arrays.npz',p=z['p'],t=z['t'])
                    m.write('mesh_complete.json',dict(interior_repair_required=False,boundary_unchanged=True))
            elif args.stage=='manifold':
                m=module('manifold',root/'geometry4_boundary_flow');m.CHART=root/'geometry4_local_repaired_chart';m.main()
            elif args.stage=='boundary':
                m=module('boundary',root/'geometry4_boundary_flow');m.SOURCE=root/'geometry4_local_repaired_chart';m.main()
            else:
                curved=module('curved',root/'numerical');chart=curved.Chart(root)
                if args.stage=='assemble':chart.assemble(curved.progress)
                else:
                    m=module('solve',curved.OUT);m.CHART=root/'geometry4_local_repaired_chart'
                    if args.stage=='solve':
                        if not args.mkl_library or not Path(args.mkl_library).is_file():raise ValueError('Explicit oneMKL LP64 runtime library required')
                        os.environ.update(NEB_MKL_LIBRARY=str(Path(args.mkl_library).resolve()),MKL_INTERFACE_LAYER='LP64',MKL_NUM_THREADS='2',MKL_DYNAMIC='FALSE',MKL_PARDISO_OOC_MAX_CORE_SIZE='3072',MKL_PARDISO_OOC_MAX_SWAP_SIZE='0')
                        m.spectrum(chart)
                    else:m.evaluate(chart)
                    m.write(f'{args.stage}_stage_complete.json',dict(status='complete',stage=args.stage,seconds=time.perf_counter()-started,source_sha=chart.source_sha))
            guard()
            if args.stage=='assemble':curved.write('assemble_stage_complete.json',dict(status='complete',stage=args.stage,seconds=time.perf_counter()-started,source_sha=chart.source_sha))
            endpoint=dict(stage=args.stage,actual_seconds=time.perf_counter()-started,config_sha256=binding['config_sha256'],samples_sha256=binding['samples_sha256'])
            (root/f'{args.stage}_complete.json').write_text(json.dumps(endpoint,indent=2),encoding='utf-8')
            print(json.dumps(endpoint))
    finally:done.set()

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stage',choices=['mesh','quality','chord','repair','chart','manifold','boundary','assemble','solve','evaluate'],required=True)
    p.add_argument('--config',type=Path,default=HERE/'configs/vrq_ccq_k32.json')
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mkl-library',type=Path);p.add_argument('--RAM-limit',type=int,default=62_500_000_000);p.add_argument('--timeout-seconds',type=float,default=7200)
    run(p.parse_args())

if __name__=='__main__':main()
