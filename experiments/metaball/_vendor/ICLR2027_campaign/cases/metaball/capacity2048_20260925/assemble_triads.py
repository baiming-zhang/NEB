






import time
import numpy as np


def triad_blocks(read_chunk, points, modes, xp=np, *, point_block=1024,
                 mode_block=16, first_mode=0, stop_mode=None, progress=None):






    end_mode = modes if stop_mode is None else min(stop_mode, modes)
    if min(points, modes, point_block, mode_block) <= 0:
        raise ValueError('Positive block dimensions required')
    if not 0 <= first_mode < end_mode <= modes:
        raise ValueError('Invalid assembly mode interval')
    for start in range(first_mode, end_mode, mode_block):
        end = min(start+mode_block, end_mode)
        tensor = xp.zeros((end-start, modes, modes), dtype='f8')
        tick = time.perf_counter()
        for off in range(0, points, point_block):
            test, velocity, curl = read_chunk(off, min(off+point_block, points))
            arrays = []
            for value in (test, velocity, curl):
                kwargs = {'blocking': True} if hasattr(xp, 'cuda') and isinstance(value, np.ndarray) else {}
                arrays.append(xp.asarray(value, dtype='f8', **kwargs))
            test, velocity, curl = arrays
            expected = (min(point_block, points-off), 3, modes)
            if any(value.shape != expected for value in arrays):
                raise ValueError('Derived fields must include every mode and requested point')

            for a,b,c,sign in ((0,1,2,1),(0,2,1,-1),(1,2,0,1),
                               (1,0,2,-1),(2,0,1,1),(2,1,0,-1)):
                product = velocity[:,b,start:end].T[:,None,:] * test[:,a,:].T[None,:,:]
                tensor += sign * (product @ curl[:,c,:])
            if hasattr(xp, 'cuda'):
                xp.cuda.get_current_stream().synchronize()
            if progress:
                progress(start, end, off+len(test), time.perf_counter()-tick)
        yield start, tensor


def pack_small(blocks, modes, *, weight):

    if modes > 16:
        raise ValueError('Dense collection is restricted to the algebraic check')
    raw = np.empty((modes,modes,modes))
    seen = np.zeros(modes, dtype=bool)
    for start, block in blocks:
        raw[start:start+len(block)] = block
        seen[start:start+len(block)] = True
    if not seen.all():
        raise ValueError('Incomplete triad blocks')
    out = []
    for j,k in zip(*np.triu_indices(modes)):
        out.append(raw[j,:,k] if j == k else raw[j,:,k]+raw[k,:,j])
    return weight*np.column_stack(out)
