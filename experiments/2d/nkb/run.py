"""Original NKB 2000-update geometry-only training and coefficient dynamics."""
from pathlib import Path
import argparse,json,os,runpy,sys
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','operators','dynamics'])
    p.add_argument('--case',choices=['periodic','one_circle','circle_square','circle_square_triangle'],default='periodic')
    p.add_argument('--K',type=int,default=1024);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--operators',type=Path);p.add_argument('--dt',type=float,default=.0025)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--fields',type=Path);p.add_argument('--quadrature',type=Path)
    a=p.parse_args();os.environ['NEB_NKB_OUTPUT_ROOT']=str(a.output.resolve());sys.path.insert(0,str(Path(__file__).resolve().parent))
    if a.stage=='train':
        sys.argv=['train.py','--case',a.case,'--K',str(a.K),'--seed',str(a.seed)];runpy.run_path(str(Path(__file__).with_name('train.py')),run_name='__main__')
        out=a.output/f'{a.case}_k{a.K}_seed{a.seed}'
        complete=json.loads((out/'training_complete.json').read_text());assert complete['epoch']==2000
        (out/'checkpoint_e2000.json').write_text(json.dumps(dict(epoch=2000,config=json.loads((out/'config.json').read_text())),indent=2));return
    import numpy as np
    if a.stage=='prepare':
        from core import Model
        model=Model(np,4 if a.case=='periodic' else 2,2,a.K,a.seed)
        out=a.output/f'{a.case}_k{a.K}_seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
        model.save(out/'initial_checkpoint.npz');model.save(out/'checkpoint.npz');return
    if a.stage=='operators':
        if not a.checkpoint or not a.fields or not a.quadrature:p.error('operators requires --checkpoint --fields --quadrature')
        from operators import build
        build(a);return
    if not a.operators:p.error('dynamics requires --operators directory mass.npz and packed_Q.npy')
    import cupy as cp
    from rom import ROM
    from common import guard
    a.output.mkdir(parents=True,exist_ok=False);stop=guard(cp,a.output)
    try:
        with np.load(a.operators/'mass.npz') as z:
            model=ROM(z['mass'],z['curl_stiffness'],np.load(a.operators/'packed_Q.npy',mmap_mode='r'));initial=z['initial_coefficients']
        states=[]
        for i in range(initial.shape[1]):
            trajectory,_=model.trajectory(initial[:,i],a.dt,lambda *args,**kwargs:None);states.append(cp.asnumpy(trajectory))
        np.savez(a.output/'coefficients.npz',coefficients=np.stack(states,axis=1),time=np.linspace(0,1,51))
    finally:stop.set()
if __name__=='__main__':main()
