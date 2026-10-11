





from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from _neb_paths import Path
import sys
import threading
import time
import traceback

import numpy as np
import psutil
import scipy.sparse as sp
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ASSETS = Path('_data/metaball/gauge_repair_k128_20261006/route2_physical_complex_q1')
MAPPING = CAMPAIGN / 'implementation/source_curl_mapping.py'
SOURCE_STAGE = CAMPAIGN / 'qualification/source_curl_mapping_cpu'
PAIRS = np.array([(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=True,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


class ReadonlyTetrahedralLocator:
    def __init__(self,p,t):
        self.p,self.t = p,t
        self.vertices = p[t]
        self.inv = np.linalg.inv((self.vertices[:,1:]-self.vertices[:,:1]).transpose(0,2,1))
        self.tree = cKDTree(self.vertices.mean(1))
        self.vertices.setflags(write=False);self.inv.setflags(write=False)

    def locate(self,points,log=lambda *args,**kwargs:None):
        points = np.asarray(points)
        selected = np.full(len(points),-1,dtype='i8')
        pending = np.arange(len(points))
        searches = []
        for k in (16,64,256,1024):
            if len(pending)==0: break
            remaining = []
            for lo in range(0,len(pending),1024):
                ids = pending[lo:lo+1024]
                candidates = self.tree.query(points[ids],k=k,workers=2)[1]
                local = np.einsum('qkij,qkj->qki',self.inv[candidates],points[ids,None]-self.vertices[candidates,0])
                bary = np.concatenate((1-local.sum(2)[:,:,None],local),axis=2)
                inside = (bary.min(2)>=-2e-10)&(bary.max(2)<=1+2e-10)
                found,pick = inside.any(1),inside.argmax(1)
                selected[ids[found]] = candidates[np.flatnonzero(found),pick[found]]
                remaining.extend(ids[~found].tolist())
            searches.append(dict(candidates=k,points_tested=len(pending),not_found=len(remaining)))
            pending = np.array(remaining,dtype='i8')
        if len(pending):
            raise ValueError(f'All original points required: {len(pending)} mesh points unlocated')
        local = np.einsum('nij,nj->ni',self.inv[selected],points-self.vertices[selected,0])
        bary = np.concatenate((1-local.sum(1)[:,None],local),axis=1)
        receipt = dict(all_points=len(points),cell_barycentric_min=float(bary.min()),
            cell_barycentric_max=float(bary.max()),occupied_cells=int(len(np.unique(selected))),
            original_total_cells=len(self.t),search=searches,points_removed=0)
        log('complete_tetrahedral_source_location',**receipt)
        return selected,self.inv,receipt


def assemble_curl_B(p,t,free,cell,inv,J,det):
    edges = np.unique(np.sort(t[:,PAIRS].reshape(-1,2),axis=1),axis=0)
    codes = edges[:,0].astype('i8')*len(p)+edges[:,1]
    local = np.sort(t[:,PAIRS],axis=2)
    edgeids = np.searchsorted(codes,local[:,:,0]*len(p)+local[:,:,1])
    slots = np.full(len(edges),-1,dtype='i8');slots[free] = np.arange(len(free))
    local = slots[edgeids[cell]]
    gradients = np.concatenate((-inv.sum(1)[:,None,:],inv),axis=1)[cell]
    orientation = np.stack([1-2*(t[cell,a]>t[cell,b]) for a,b in PAIRS],axis=1)
    curls = np.stack([2*np.cross(gradients[:,a],gradients[:,b])*orientation[:,j,None]
        for j,(a,b) in enumerate(PAIRS)],axis=1)
    curls = np.einsum('nai,nei->nea',J,curls)/det[:,None,None]
    rows = np.broadcast_to(3*np.arange(len(cell))[:,None,None]+np.arange(3)[None,None,:],curls.shape)
    cols = np.broadcast_to(local[:,:,None],curls.shape)
    keep = cols>=0
    B = sp.coo_matrix((curls[keep],(rows[keep],cols[keep])),shape=(3*len(cell),len(free))).tocsr()
    B.sum_duplicates();B.eliminate_zeros()
    return B


def qualify(args):
    if not (0 < args.timeout <= 7200): raise ValueError('Finite7200s limit required.')
    out = Path(args.output)
    if (out/'complete.json').exists() or (out/'start.json').exists(): raise RuntimeError('Existing qualification must not be repeated.')
    out.mkdir(parents=True,exist_ok=True)
    started,done = time.perf_counter(),threading.Event()
    process,creation = psutil.Process(),psutil.Process().create_time()
    def guard():
        row = dict(unix=time.time(),pid=os.getpid(),creation=creation,seconds=time.perf_counter()-started,
            rss=process.memory_info().rss,hostfree=psutil.virtual_memory().available,CPU_only=True)
        with (out/'resources.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        if row['seconds']>=args.timeout or row['rss']>=8*2**30 or row['hostfree']<=9*2**30:
            write(out/'bounded_stop.json',row);raise RuntimeError('Actual bounded CPU stop.')
    def watch():
        while not done.is_set():
            try:guard()
            except BaseException:os._exit(124)
            done.wait(2.)
    watcher = threading.Thread(target=watch,daemon=True)
    required = [Path(__file__),MAPPING,ASSETS/'complex.npz',SOURCE_STAGE/'complete.json',
        SOURCE_STAGE/'source_inverse_map.npz',SOURCE_STAGE/'B_source_curl.npz']
    try:
        inputs = {str(path):sha(path) for path in required}
        source = json.loads((SOURCE_STAGE/'complete.json').read_text(encoding='utf-8-sig'))
        if source.get('qualified') is not True or source.get('source_points')!=103244:raise ValueError('Complete actual source mapping prerequisite absent.')
        write(out/'start.json',dict(pid=os.getpid(),creation=creation,CPU_only=True,source_sha256=inputs,unix=time.time()))
        guard();watcher.start()
        spec = importlib.util.spec_from_file_location('locator_literal_source_mapping',MAPPING)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with np.load(ASSETS/'complex.npz',allow_pickle=False) as z:p,t,free = z['p'].copy(),z['t'].copy(),z['free'].copy()
        with np.load(SOURCE_STAGE/'source_inverse_map.npz',allow_pickle=False) as z:
            points,J,det,saved_cell = z['reference_points'].copy(),z['J'].copy(),z['det'].copy(),z['cell_ids'].copy()
        if len(points)!=103244 or len(t)!=222627 or len(free)!=239415:raise ValueError('All original points/cells/relative edges required.')
        timings = []
        with threadpool_limits(limits=2):
            tic = time.perf_counter();literal_cell,literal_inv,literal_check = module.locate_tetrahedra(p,t,points,lambda *a,**kw:None)
            literal_seconds = time.perf_counter()-tic
            if not np.array_equal(literal_cell,saved_cell):raise AssertionError('Literal actual source cell assignment changed.')
            tic = time.perf_counter();cached = ReadonlyTetrahedralLocator(p,t);setup = time.perf_counter()-tic
            for repeat in range(3):
                tic = time.perf_counter();cell,inv,check = cached.locate(points)
                timings.append(time.perf_counter()-tic)
                if not np.array_equal(cell,literal_cell) or not np.array_equal(inv,literal_inv) or check!=literal_check:
                    raise AssertionError('Cached locator changed any actual cell/inverse/check record.')
            tic = time.perf_counter();B = assemble_curl_B(p,t,free,cell,inv,J,det);assemble = time.perf_counter()-tic
            saved_B = sp.load_npz(SOURCE_STAGE/'B_source_curl.npz').tocsr()
            if B.shape!=saved_B.shape or not all(np.array_equal(a,b) for a,b in
                ((B.data,saved_B.data),(B.indices,saved_B.indices),(B.indptr,saved_B.indptr))):
                raise AssertionError('Actual all-point physical curl CSR values/order changed.')
        if any(sha(path)!=digest for path,digest in inputs.items()):raise RuntimeError('Bound readonly input changed.')
        if any(name in sys.modules for name in ('cupy','torch')):raise AssertionError('CPU cache qualification imported CUDA.')
        write(out/'complete.json',dict(status='complete',qualified=True,K=32,
            scope='Only complete103244-source static tetrahedral locator/B equivalence; fresh6144-query/all32 ANN runtime not yet qualified.',
            all_original_points=103244,all_original_cells=222627,all_relative_edges=239415,
            original_cell_inverse_and_location_checks_bitwise=True,all_full_B_CSR_arrays_bitwise=True,
            locator_setup_seconds=setup,literal_complete_source_location_seconds=literal_seconds,
            cached_complete_source_location_seconds=timings,B_assembly_seconds=assemble,
            cached_scope='Potential-added immutable p/t, tetrahedral inverse and KD tree only; no shared geometry/ANN/Taylor cache',
            source_sha256=inputs,CPU_only=True,CUDA_imported=False,optimizer_updates=0,
            actual_seconds=time.perf_counter()-started,pid=os.getpid(),creation=creation))
    except BaseException:
        write(out/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-started));raise
    finally:
        done.set()
        if watcher.is_alive():watcher.join(timeout=3.)
