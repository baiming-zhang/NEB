"""CPU public compact-vortex asset reader and exact quadrature cell-order checks."""
from pathlib import Path
import tempfile,json,time
import numpy as np
from .compact_fields import load,targets
from ..reference.run import vortex_parameters

def checks():
    start=time.perf_counter();config=json.loads((Path(__file__).parents[1]/'reference/configs/one_circle_random20.json').read_text())
    with tempfile.TemporaryDirectory(prefix='neb-compact-reader-') as temp:
        root=Path(temp);xy=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]);tri=np.array([[0,1,2],[1,3,2]])
        cells=np.array([1,0,1,0]);bc=np.array([[.2,.3,.5]]*4);points=np.einsum('ni,nij->nj',bc,xy[tri[cells]])
        np.savez(root/'q.npz',points=points,cell_index=cells,barycentric=bc)
        for sample in range(20):
            d=root/f'sample{sample:02d}';d.mkdir();p=vortex_parameters(config,sample)
            (d/'initial_specification.json').write_text(json.dumps(dict(seed=p['seed'],vortex_profile='compact Wendland C4',harmonic_to_vortical_energy_ratio=p['harmonic_energy_ratio'],target_peak_speed=p['target_peak_speed'],CPU_synthetic_state=True)))
            np.savez(d/'initial_state.npz',xy=xy,tri=tri,cell_velocity=np.array([[sample,2*sample],[3*sample,4*sample]],dtype=float))
        from .compare import read_reference
        for key in ('t','times'):
            np.savez(root/f'ref_{key}.npz',xy=xy,psi=np.ones((2,4)),**{key:np.array([0.,.02])})
            assert np.array_equal(read_reference(root/f'ref_{key}.npz')[2],[0.,.02])
        v,ids=load(root,'one_circle',root/'q.npz');T=targets(v,ids,0,4)
        assert T.shape==(4,2,20) and np.array_equal(T[:,:,7],np.array([[21,28],[7,14],[21,28],[7,14]]))
        d=root/'sample19';s=json.loads((d/'initial_specification.json').read_text());s['seed']+=1;(d/'initial_specification.json').write_text(json.dumps(s))
        try:load(root,'one_circle',root/'q.npz')
        except AssertionError:pass
        else:raise AssertionError('Wrong seed accepted')
    return dict(CPU_only=True,CUDA_initialized=False,public_same_mesh_quadrature_reader=True,all20_exact_original_seed_parameters=True,wrong_seed_rejected=True,original_P1_times_and_P3_t_keys_verified=True,synthetic_P1_state_only=True,actual_seconds=time.perf_counter()-start)
if __name__=='__main__':print(json.dumps(checks()))
