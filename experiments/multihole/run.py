"""Native multihole initialization/training and exact random-field generator."""
from pathlib import Path
import argparse, importlib.util, json, os, runpy, sys
BASE=Path(__file__).resolve().parents[1]/'2d/_backend'
TOP=BASE/'flow/cases/topology3'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','harmonic-train','fields'])
    p.add_argument('--config',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['one_circle','circle_square','circle_square_triangle'],default='one_circle')
    a=p.parse_args();os.environ['NEB_2D_VENDOR_ROOT']=str(BASE)
    if a.stage!='fields':
        root=TOP/'learned_basis'
        if a.stage=='prepare':
            if not a.config:p.error('prepare requires --config')
            script=root/'positive/scripts/prepare_hardcoded_run.py';args=['--config',str(a.config),'--output-dir',str(a.output)]
        elif a.stage=='train':
            if not a.checkpoint:p.error('train requires --checkpoint epoch-zero file')
            script=root/'positive/scripts/train_hardcoded.py';args=['--resume',str(a.checkpoint),'--run-dir',str(a.output)]
        else:
            if not a.config:p.error('harmonic-train requires --config')
            script=root/'harmonic/scripts/train_harmonic.py';args=['--config',str(a.config),'--run-dir',str(a.output)]
        sys.path.insert(0,str(script.parent));sys.argv=[str(script),*args];runpy.run_path(str(script),run_name='__main__');return
    import numpy as np
    path=TOP/'random_streamfunctions/preview_random_streamfunction.py'
    spec=importlib.util.spec_from_file_location('neb_random_fields',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    a.output.mkdir(parents=True,exist_ok=False)
    xx,yy=np.meshgrid(np.linspace(1,9,321),np.linspace(-4,4,321));points=np.column_stack((xx.ravel(),yy.ravel()))
    points=points[module.geo.fluid_mask(points,a.case)]
    for sample in range(20):
        field=module.RandomField(a.case,sample)
        values=field.evaluate(points);boundary=field.evaluate(np.vstack(field.loops))
        peak=max(float(np.linalg.norm(values[:,1:3],axis=1).max()),float(np.linalg.norm(boundary[:,1:3],axis=1).max()))
        field.scale=.5/peak
        np.savez_compressed(a.output/f'sample{sample:02d}.npz',raw_lattice=field.raw,raw_x=field.x,raw_y=field.y,trace_constants=field.constants,global_scale=field.scale,seed_hierarchy=field.seed)
    (a.output/'distribution.json').write_text(json.dumps(dict(case=a.case,n=20,master_seed=20260928,normalization='0.5 maximum on fixed 321 grid fluid points and original boundary loops',training_targets=False,reference='original quintic analytic RandomField; CPU evaluation'),indent=2))
if __name__=='__main__':main()
