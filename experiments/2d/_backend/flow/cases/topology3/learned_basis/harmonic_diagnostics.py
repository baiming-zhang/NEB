"""Strong curl and homotopic circulation checks; rank alone is not harmonicity."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import cupy as cp
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'positive/src'),str(ROOT/'harmonic/scripts')]
from train_harmonic import initialize_arrays
from harmonic_common import learned_harmonic_fields as jet1_fields
from harmonic_jet2 import learned_harmonic_fields as jet2_fields
from neb2d.basisnet.metaball2d import hardcoded_backend as be,geometry as geo

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 config=json.loads(args.config.read_text());cfg=be.cfg_namespace(config)
 if args.checkpoint:
  with np.load(args.checkpoint) as z:arrays={key:z[key] for key in z.files}
 else:arrays=initialize_arrays(cfg)
 model=be.HardcodedModel(cfg,arrays);points=be.fluid_points(cfg,8192,0,0,cfg.seed+1800000)
 grad_energy=np.zeros(cfg.modes);curl_energy=np.zeros(cfg.modes);parity=0.
 for start in range(0,len(points),1024):
  chunk=points[start:start+1024];one=jet1_fields(model,cfg,chunk,False);two=jet2_fields(model,cfg,chunk,False)
  for j,(a,b) in enumerate(zip(one,two)):
   parity=max(parity,float(cp.max(cp.abs(a[1].grad-b[1].grad)).item()),float(cp.max(cp.abs(a[1].value-b[1].value)).item()))
   field=b[1];grad_energy[j]+=float(cp.sum(field.grad**2).item())
   curl_energy[j]+=float(cp.sum((field.hess[...,0]+field.hess[...,2])**2).item())
 periods={};valid={}
 for factor in (1.04,1.12,1.25):
  mat=[];all_valid=True
  for spec in geo.get_obstacles(cfg.subcase):
   boundary=geo.obstacle_boundary(spec,1024);center=np.array([spec.x,spec.y]);loop=center+factor*(boundary-center)
   mid=(loop+np.roll(loop,-1,axis=0))/2;dl=np.roll(loop,-1,axis=0)-loop
   all_valid=all_valid and bool(geo.fluid_mask(mid,cfg.subcase,0).all())
   columns=[]
   for _,field,_ in jet1_fields(model,cfg,mid,False):
    gradient=cp.asnumpy(field.grad[:,0]);u=np.column_stack((gradient[:,1],-gradient[:,0]));columns.append(float(np.sum(u*dl)))
   mat.append(columns)
  periods[str(factor)]=mat;valid[str(factor)]=all_valid
 ref=np.array(periods['1.04']);variation=max(float(np.linalg.norm(np.array(m)-ref)/max(np.linalg.norm(ref),1e-30)) for m in periods.values())
 out=dict(case=cfg.subcase,checkpoint=str(args.checkpoint),jet1_jet2_max_difference=parity,energy_density=(grad_energy/len(points)).tolist(),strong_curl_rms=np.sqrt(curl_energy/len(points)).tolist(),curl_over_velocity_inverse_length=np.sqrt(curl_energy/grad_energy).tolist(),period_matrices=periods,loops_in_fluid=valid,relative_homotopic_period_variation=variation,period_rank=int(np.linalg.matrix_rank(ref)),scope='Finite neural fields; report residual and loop dependence rather than infer exact harmonicity from period rank.')
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
 assert parity<2e-5 and all(valid.values()),'Implementation/loop geometry gate failed'
if __name__=='__main__':main()
