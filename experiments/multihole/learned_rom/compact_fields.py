"""Original compact-vortex P1 initial velocities on the same physical quadrature."""
from pathlib import Path
import json
import numpy as np

def load(root,case,quadrature,config=None):
    from ..reference.run import vortex_parameters
    source=Path(__file__).parents[1]/'reference/configs'/f'{case}_random20.json'
    config=json.loads(Path(config or source).read_text());assert config['subcase']==case
    with np.load(quadrature) as z:points=z['points'];cells=z['cell_index'];bc=z['barycentric']
    assert cells.shape==(len(points),) and bc.shape==(len(points),3) and cells.dtype.kind in 'iu'
    velocities=[];canonical=None
    for sample in range(20):
        directory=Path(root)/f'sample{sample:02d}'
        spec=json.loads((directory/'initial_specification.json').read_text())
        expected=vortex_parameters(config,sample)
        assert spec['seed']==expected['seed'] and spec['vortex_profile']=='compact Wendland C4'
        assert spec['harmonic_to_vortical_energy_ratio']==expected['harmonic_energy_ratio'] and spec['target_peak_speed']==expected['target_peak_speed']
        with np.load(directory/'initial_state.npz') as z:xy=z['xy'];tri=z['tri'];v=z['cell_velocity']
        assert tri.shape[1]==3 and v.shape==(len(tri),2) and np.isfinite(v).all()
        if canonical is None:canonical=(xy,tri)
        else:assert np.array_equal(xy,canonical[0]) and np.array_equal(tri,canonical[1])
        assert cells.min()>=0 and cells.max()<len(tri)
        reconstructed=np.einsum('ni,nij->nj',bc,xy[tri[cells]])
        assert np.allclose(points,reconstructed,rtol=0,atol=2e-12)
        velocities.append(v)
    return velocities,cells

def targets(velocities,cells,first,count):
    selected=cells[first:first+count]
    return np.stack([v[selected] for v in velocities],axis=2)
