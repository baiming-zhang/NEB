"""Original per-geometry frame000 and disjoint16384 fit/assessment samples."""
from pathlib import Path
import argparse,json,sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--case',type=int,choices=[1,2,3,4],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from train import V,resource
    sys.path.insert(0,str(V));sys.path.insert(0,str(V/'heldout'))
    from reference import geometry_grid,Spectral
    from samples import freeze_samples
    record=json.loads((Path(__file__).parent/f'configs/heldout_case{a.case}.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    grid=geometry_grid(record['geometry']);solver=Spectral(grid);cp=solver.cp;resource(cp,out)
    state=solver.initialize(record['phases']);velocity=cp.asnumpy(cp.where(solver.mask[None],solver.real(state),cp.float32(0)))
    freeze_samples(grid['fluid_mask'],velocity,record['geometry'],a.case,out/'samples.npz')
    for k,v in grid.items():np.save(out/(k+'.npy'),v,allow_pickle=False)
    np.save(out/'velocity.npy',velocity,allow_pickle=False);(out/'geometry.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
