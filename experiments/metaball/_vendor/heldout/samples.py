from pathlib import Path
import numpy as np
def point_coordinates(indices, resolution=128):
    z, y, x = np.unravel_index(indices, (resolution,) * 3)
    return ((np.stack((x, y, z), axis=1) + 0.5) / resolution).astype('f4')
def freeze_samples(mask, velocity, geometry, case_id, path):
    """Uniform grid-volume subsets without replacement, disjoint by construction.

    First 8192 fit points also support the fixed half-count sensitivity check.
    velocity layout is (3,z,y,x); values are used directly, no interpolation.
    """
    valid = np.flatnonzero(mask.reshape(-1))
    assert velocity.shape == (3,) + mask.shape
    assert len(valid) >= 32768
    rng = np.random.Generator(np.random.PCG64(20260912 + case_id))
    chosen = rng.choice(valid, 32768, replace=False)
    targets = velocity.reshape(3, -1)[:, chosen].T.copy()
    assert np.isfinite(targets).all()
    result = dict(fit_indices=chosen[:16384], assessment_indices=chosen[16384:], fit_points=point_coordinates(chosen[:16384], len(mask)), assessment_points=point_coordinates(chosen[16384:], len(mask)), fit_target=targets[:16384], assessment_target=targets[16384:], geometry=np.asarray(geometry, dtype='f4'), fluid_count=np.asarray(len(valid)))
    if Path(path).exists():
        with np.load(path) as old:
            assert set(old.files) == set(result)
            assert all((np.array_equal(old[k], v) for k, v in result.items()))
    else:
        np.savez(path, **result)
    return result