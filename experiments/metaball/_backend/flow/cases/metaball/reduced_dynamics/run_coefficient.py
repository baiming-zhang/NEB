




import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gc
import importlib.util
import json
from _neb_paths import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('_data/metaball/reduced_dynamics')
RUN = OUT / 'coefficient_k512'
from prepare_coefficient_fields import header, write


def environment(*, pool_limit_gib=4):
    spec = importlib.util.spec_from_file_location('coefficient_grid_template',
        HERE.parent / 'fixed_geometry_k256_rollout_20260921/run.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.guard_module.RUN = RUN
    done, peaks = old.guard_module.start(pool_limit_gib=pool_limit_gib)
    return old, done, peaks


class Fields:
    def __init__(self):
        self.ready = json.loads((RUN / 'derived_fields_complete.json').read_text())
        self.n, self.k = self.ready['points'], self.ready['K']
        self.bpath = Path(self.ready['original_fields'])
        self.dpath = Path(self.ready['derived_fields'])
        bs, bt, self.bo = header(self.bpath)
        ds, dt, self.do = header(self.dpath)
        assert bs == (self.n, 3, self.k) and bt == np.dtype('f4')
        assert ds == (self.n, 6, self.k) and dt == np.dtype('f8')
        self.bs, self.ds = self.bpath.open('rb'), self.dpath.open('rb')

    def read(self, first, count, cp):
        self.bs.seek(self.bo + first * 3 * self.k * 4)
        self.ds.seek(self.do + first * 6 * self.k * 8)
        b = np.fromfile(self.bs, dtype='f4', count=count*3*self.k).reshape(count,3,self.k)
        d = np.fromfile(self.ds, dtype='f8', count=count*6*self.k).reshape(count,6,self.k)

        b = cp.asarray(np.ascontiguousarray(b.transpose(1,2,0)), dtype='f8', blocking=True)
        d = cp.asarray(np.ascontiguousarray(d.transpose(1,2,0)), blocking=True)
        return b, d[:3], d[3:]


def add_tile(raw, b, p, w, cp, mode_block):
    k, points = b.shape[1:]
    for first in range(0, k, mode_block):
        last = min(k, first+mode_block)
        target = raw[first:last].reshape(-1,k)
        for a,bb,c,sign in ((0,1,2,1),(0,2,1,-1),(1,2,0,1),
                            (1,0,2,-1),(2,0,1,1),(2,1,0,-1)):
            product = (b[bb,first:last,None,:] * p[a,None,:,:]).reshape(-1,points)
            target += sign * (product @ w[c].T)


def benchmark(cp):
    f = Fields()
    sync = cp.cuda.get_current_stream().synchronize
    raw = cp.zeros((f.k,f.k,f.k), dtype='f8')
    results = []
    for points, block in ((512,16),(1024,32),(2048,32)):
        t = time.perf_counter()
        b,p,w = f.read(0,points,cp)
        sync(); read_seconds = time.perf_counter()-t

        t = time.perf_counter()
        add_tile(raw,b,p,w,cp,block)
        sync(); elapsed = time.perf_counter()-t
        row = dict(points=points,mode_block=block,read_seconds=read_seconds,
                   compute_seconds=elapsed,full_assembly_estimate_seconds=(elapsed+read_seconds)*f.n/points)
        results.append(row)
        print(json.dumps(row),flush=True)
        del b,p,w
    write(RUN/'point_outer_gpu_benchmark.json',dict(K=f.k,backend='CuPy FP64',
          scope='All-mode real-data tile timing; full-quadrature assembly still pending',results=results))


def assemble(cp):
    f=Fields()
    marker=RUN/'quadratic_complete.json'
    if marker.exists():
        receipt=json.loads(marker.read_text())
        assert receipt['contract']==f.ready['contract']
        q=np.load(RUN/'quadratic.npy',mmap_mode='r')
        assert q.shape==(f.k,f.k*(f.k+1)//2) and q.dtype==np.dtype('f8')
        return
    sync=cp.cuda.get_current_stream().synchronize


    from assemble_triads import triad_blocks
    b,p,w=f.read(0,17,cp)
    raw=cp.zeros((f.k,f.k,f.k),dtype='f8')
    add_tile(raw,b,p,w,cp,32)
    reader=lambda first,last:(p[:,:,first:last].transpose(2,0,1),
                              b[:,:,first:last].transpose(2,0,1),
                              w[:,:,first:last].transpose(2,0,1))
    worst=0.
    for first,expected in triad_blocks(reader,17,f.k,cp,point_block=17,mode_block=8,stop_mode=8):
        worst=float(cp.linalg.norm(raw[first:first+8]-expected)/cp.linalg.norm(expected))
        assert worst<2e-12
    del b,p,w,expected
    raw.fill(0)
    checkpoint=RUN/'assembly_checkpoint.npz'
    first_point=0; previous=0.
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            assert str(z['contract'])==json.dumps(f.ready['contract'],sort_keys=True)
            first_point=int(z['points']);previous=float(z['seconds'])
            raw[:]=cp.asarray(z['tensor'],blocking=True)
    begin=time.perf_counter();last_save=begin;last_report=begin
    for first in range(first_point,f.n,2048):
        if (RUN/'user_requested_stop.json').exists():
            raise RuntimeError('New coefficient run stopped by user; do not restart')
        count=min(2048,f.n-first)
        b,p,w=f.read(first,count,cp)
        add_tile(raw,b,p,w,cp,32)
        sync(); del b,p,w
        now=time.perf_counter();completed=first+count
        if now-last_report>=30 or completed==f.n:
            elapsed=previous+now-begin
            row=dict(phase='full_quadrature_assembly',K=f.k,points=completed,total_points=f.n,
                     seconds=elapsed,remaining_seconds=elapsed*(f.n/completed-1),pid=os.getpid(),unix=time.time())
            write(RUN/'accelerated_progress.json',row);print(json.dumps(row),flush=True)
            last_report=now
        if now-last_save>=600 or completed==f.n:
            temp=checkpoint.with_suffix('.tmp.npz')
            host=cp.asnumpy(raw)
            np.savez(temp,tensor=host,points=completed,seconds=previous+time.perf_counter()-begin,
                     contract=json.dumps(f.ready['contract'],sort_keys=True))
            temp.replace(checkpoint);del host
            last_save=time.perf_counter()

    pairs=f.k*(f.k+1)//2
    temp=RUN/'quadratic.tmp.npy'
    with temp.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream,dict(shape=(f.k,pairs),fortran_order=False,descr='<f8'))
        jj,kk=cp.triu_indices(f.k)
        for i in range(f.k):
            values=raw[jj,i,kk]+cp.where(jj==kk,0.,raw[kk,i,jj])
            cp.asnumpy(values/128**3).tofile(stream)
    temp.replace(RUN/'quadratic.npy')
    sync()
    write(marker,dict(K=f.k,points=f.n,contract=f.ready['contract'],
          seconds=previous+time.perf_counter()-begin,precision='float64',
          packed_shape=[f.k,pairs],all_nodes=True,all_modes=True,
          actual_tile_equivalence_relative=worst,source='Original rotational generator, full quadrature'))
    del raw,values
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
