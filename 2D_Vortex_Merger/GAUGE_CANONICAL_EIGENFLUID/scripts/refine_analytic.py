"""Resolve the remaining time-step/collocation error in the K=1024 result."""
import numpy as np
from verify_analytic import OUT,load_gt,rollout,write_json

gt=load_gt()
report,_,_=rollout(1024,gt,dt=.005,n=128,output_name='refinement_k1024_n128_dt005')
with np.load(OUT/'rollout_k1024/fields.npz') as a,np.load(OUT/'refinement_k1024_n128_dt005/fields.npz') as b:
    refinement={key:float(np.max(abs(a[key]-b[key]))) for key in ('u','v','omega')}
assert max(refinement.values())<1e-8,refinement
write_json(OUT/'refinement_check.json',dict(coarse=dict(grid=64,dt=.01),fine=dict(grid=128,dt=.005),
    max_abs_difference=refinement,interpretation='Remaining DNS discrepancy is spectral truncation, not the tested time/collocation resolution.'))
print(refinement)
