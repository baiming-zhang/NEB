import argparse
import json
import os
from _neb_paths import Path
import threading
import time
import traceback
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import numpy as np
import psutil
import scipy.sparse as sp
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = Path('_data/metaball/physical_complex/route2_physical_complex_q1')
SOURCE = ROOT / 'flow/cases/metaball/results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz'
CONFIG = ROOT / 'exports/coulomb_vel_matched_20261006/protocol.json'
PAIRS = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(p)

class RadialCPU:

    def __init__(self, p, t, geometry, config, log):
        self.p, self.t = (p, t)
        self.geometry = geometry
        self.c = config
        self.center = geometry[1, :3]
        self.log = log
        faces = np.sort(t[:, np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])].reshape(-1, 3), axis=1)
        unique, first, count = np.unique(faces, axis=0, return_index=True, return_counts=True)
        boundary = count == 1
        if np.any((count != 1) & (count != 2)):
            raise ValueError('Original tetrahedral mesh has nonmanifold faces')
        faces = unique[boundary]
        cells = first[boundary] // 4
        if len(faces) != 25132:
            raise ValueError('Incomplete original true-wall mesh')
        v = p[faces]
        q = v - self.center
        normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        sign = np.where(np.sum(normal * (v.mean(1) - p[t[cells]].mean(1)), axis=1) > 0, 1.0, -1.0)
        normal *= sign[:, None]
        fo, fi, go, gi = self.level_grad(v.mean(1))
        outer = np.abs(fo) / np.maximum(np.linalg.norm(go, axis=1), 1e-30) < np.abs(fi) / np.maximum(np.linalg.norm(gi, axis=1), 1e-30)
        self.surfaces = []
        self.stats = []
        for flag in (False, True):
            ids = np.flatnonzero(outer == flag)
            qq = q[ids]
            nn = normal[ids]
            plane = np.sum(nn * qq[:, 0], axis=1)
            margin = plane if flag else -plane
            if np.min(margin) <= 1e-08:
                raise ValueError('Original radial wall orientation failed')
            cone = np.stack([np.cross(qq[:, 0], qq[:, 1]), np.cross(qq[:, 1], qq[:, 2]), np.cross(qq[:, 2], qq[:, 0])], axis=1)
            cone *= np.sign(np.sum(cone[:, 0] * qq[:, 2], axis=1))[:, None, None]
            cone /= np.linalg.norm(cone, axis=2)[:, :, None]
            directions = qq.mean(1)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            self.surfaces.append(dict(normal=nn, plane=plane, cone=cone, tree=cKDTree(directions), ids=ids))
            self.stats.append(dict(faces=len(ids), margin_min=float(margin.min()), vertex_radius_max=float(np.linalg.norm(qq, axis=2).max())))
        if [s['faces'] for s in self.stats] != [5678, 19454]:
            raise ValueError('Original two boundary components do not match')
        displacement = np.linalg.norm(geometry[:, :3] - self.center, axis=1)
        radii = geometry[:, 3]
        bounds = displacement + radii
        top = bounds.max()
        self.true_inner_lo = geometry[1, 3]
        self.true_inner_hi = float(top + config['metaball_softmin_tau'] * np.log(np.exp((bounds - top) / config['metaball_softmin_tau']).sum()))
        self.true_outer_lo = float(config['outer_half_extent'] - np.linalg.norm(self.center - 0.5))
        self.true_outer_hi = float(np.linalg.norm(config['outer_half_extent'] + np.abs(self.center - 0.5)))

    def level_grad(self, x):
        q = (x - 0.5) / self.c['outer_half_extent']
        power = self.c['outer_superellipsoid_power']
        fo = 1 - np.sum(q ** power, axis=-1)
        go = -power / self.c['outer_half_extent'] * q ** (power - 1)
        delta = x[..., None, :] - self.geometry[:, :3]
        length = np.linalg.norm(delta, axis=-1)
        dist = length - self.geometry[:, 3]
        low = dist.min(-1)
        w = np.exp(-(dist - low[..., None]) / self.c['metaball_softmin_tau'])
        fi = low - self.c['metaball_softmin_tau'] * np.log(w.sum(-1))
        w /= w.sum(-1)[..., None]
        gi = np.sum(w[..., None] * delta / np.maximum(length[..., None], 1e-30), axis=-2)
        return (fo, fi, go, gi)

    def mesh_ends(self, directions):
        ends = []
        for wall, surface in enumerate(self.surfaces):
            selected = np.full(len(directions), -1, dtype='i8')
            tested = []
            pending = np.arange(len(directions))
            for k in (16, 64, 256, 1024, len(surface['ids'])):
                if len(pending) == 0:
                    break
                kk = min(k, len(surface['ids']))
                chunk = max(1, min(2048, 400000 // kk))
                new_pending = []
                for lo in range(0, len(pending), chunk):
                    query = pending[lo:lo + chunk]
                    candidates = surface['tree'].query(directions[query], k=kk, workers=2)[1]
                    if kk == 1:
                        candidates = candidates[:, None]
                    values = np.einsum('qi,qkji->qkj', directions[query], surface['cone'][candidates])
                    hit = np.all(values >= -2e-12, axis=2)
                    found = hit.any(1)
                    chosen = hit.argmax(1)
                    selected[query[found]] = candidates[np.flatnonzero(found), chosen[found]]
                    new_pending.extend(query[~found].tolist())
                tested.append(dict(candidates=kk, points_tested=len(pending), not_found=len(new_pending)))
                pending = np.array(new_pending, dtype='i8')
            if len(pending):
                raise ValueError('An original physical source ray was not covered')
            nn = surface['normal'][selected]
            den = np.sum(nn * directions, axis=1)
            radius = surface['plane'][selected] / den
            derivative = -radius[:, None] * nn / den[:, None]
            ends.append((radius, derivative))
            self.log('complete_radial_wall_ray_location', wall=wall, points=len(directions), search=tested)
        return ends

    def true_ends(self, directions):
        ends = []
        for inner in (True, False):
            low = np.full(len(directions), self.true_inner_lo if inner else self.true_outer_lo)
            high = np.full(len(directions), self.true_inner_hi if inner else self.true_outer_hi)
            for _ in range(45):
                mid = (low + high) / 2
                fo, fi, _, _ = self.level_grad(self.center + mid[:, None] * directions)
                below = fi < 0 if inner else fo > 0
                low = np.where(below, mid, low)
                high = np.where(below, high, mid)
            radius = (low + high) / 2
            fo, fi, go, gi = self.level_grad(self.center + radius[:, None] * directions)
            gradient = gi if inner else go
            derivative = np.sum(gradient * directions, axis=1)
            if not np.all(derivative > 0 if inner else derivative < 0):
                raise ValueError('Original true-wall radial uniqueness failed')
            dr = -radius[:, None] * gradient / derivative[:, None]
            ends.append((radius, dr))
        return ends

    def radial(self, directions, radius, source, target):
        a, da = source[0]
        b, db = source[1]
        A, dA = target[0]
        B, dB = target[1]
        k = (B - A) / (b - a)
        s = (radius - a) / (b - a)
        rho = A + s * (B - A)
        tangent = (1 - s[:, None]) * dA + s[:, None] * dB - k[:, None] * ((1 - s[:, None]) * da + s[:, None] * db)
        tangent -= directions * np.sum(tangent * directions, axis=1)[:, None]
        grad = k[:, None] * directions + tangent / radius[:, None]
        projector = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
        J = (rho / radius)[:, None, None] * projector + directions[:, :, None] * grad[:, None, :]
        det = k * (rho / radius) ** 2
        return (self.center + rho[:, None] * directions, J, det, s)

    def inverse(self, physical):
        delta = physical - self.center
        radius = np.linalg.norm(delta, axis=1)
        directions = delta / radius[:, None]
        mesh, true = (self.mesh_ends(directions), self.true_ends(directions))
        reference, Jinv, detinv, fraction = self.radial(directions, radius, true, mesh)
        reconstructed, J, det, _ = self.radial(directions, np.linalg.norm(reference - self.center, axis=1), mesh, true)
        certificate = dict(inverse_point_max=float(np.max(np.linalg.norm(reconstructed - physical, axis=1))), inverse_J_max=float(np.max(np.abs(J @ Jinv - np.eye(3)))), forward_det_formula_max=float(np.max(np.abs(np.linalg.det(J) - det))), reference_fraction_min=float(fraction.min()), reference_fraction_max=float(fraction.max()), forward_det_min=float(det.min()), all_source_points=len(physical))
        if certificate['inverse_point_max'] >= 1e-10 or certificate['inverse_J_max'] >= 1e-08 or certificate['forward_det_min'] <= 0 or (certificate['reference_fraction_min'] < -1e-09) or (certificate['reference_fraction_max'] > 1 + 1e-09):
            raise ValueError(certificate)
        return (reference, J, det, certificate)

def locate_tetrahedra(p, t, points, log):
    vertices = p[t]
    inv = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1))
    tree = cKDTree(vertices.mean(1))
    selected = np.full(len(points), -1, dtype='i8')
    pending = np.arange(len(points))
    searches = []
    for k in (16, 64, 256, 1024):
        if len(pending) == 0:
            break
        remaining = []
        for lo in range(0, len(pending), 1024):
            ids = pending[lo:lo + 1024]
            candidates = tree.query(points[ids], k=k, workers=2)[1]
            local = np.einsum('qkij,qkj->qki', inv[candidates], points[ids, None] - vertices[candidates, 0])
            bary = np.concatenate((1 - local.sum(2)[:, :, None], local), axis=2)
            inside = (bary.min(2) >= -2e-10) & (bary.max(2) <= 1 + 2e-10)
            found = inside.any(1)
            pick = inside.argmax(1)
            selected[ids[found]] = candidates[np.flatnonzero(found), pick[found]]
            remaining.extend(ids[~found].tolist())
        searches.append(dict(candidates=k, points_tested=len(pending), not_found=len(remaining)))
        pending = np.array(remaining, dtype='i8')
    if len(pending):
        raise ValueError(f'All original points required: {len(pending)} mesh points unlocated')
    local = np.einsum('nij,nj->ni', inv[selected], points - vertices[selected, 0])
    bary = np.concatenate((1 - local.sum(1)[:, None], local), axis=1)
    certificate = dict(all_points=len(points), cell_barycentric_min=float(bary.min()), cell_barycentric_max=float(bary.max()), occupied_cells=int(len(np.unique(selected))), original_total_cells=len(t), search=searches, points_removed=0)
    log('complete_tetrahedral_source_location', **certificate)
    return (selected, inv, certificate)

def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise RuntimeError('Completed source mapping must not be repeated')
    start = time.perf_counter()
    creation = psutil.Process().create_time()
    done = threading.Event()

    def log(phase, **kw):
        row = dict(phase=phase, pid=os.getpid(), creation=creation, unix=time.time(), stage_elapsed_seconds=time.perf_counter() - start, **kw)
        write(output / 'progress.json', row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    def watch():
        while not done.is_set():
            r = dict(pid=os.getpid(), creation=creation, unix=time.time(), rss=psutil.Process().memory_info().rss, hostfree=psutil.virtual_memory().available, seconds=time.perf_counter() - start, CPU_only=True)
            with (output / 'resources.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(r) + '\n')
            if r['rss'] >= 8 * 2 ** 30 or r['hostfree'] <= 9 * 2 ** 30 or r['seconds'] >= args.timeout:
                write(output / 'bounded_stop.json', r)
                os._exit(124)
            done.wait(2)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with threadpool_limits(limits=2):
            with np.load(ASSETS / 'complex.npz') as z:
                p, t, free, H = (z['p'].copy(), z['t'].copy(), z['free'].copy(), z['H'].copy())
            with np.load(SOURCE) as z:
                physical = z['fit'].copy()
            if len(physical) != 103244:
                raise ValueError('Changed original complete source points')
            protocol = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
            config = protocol['run_settings']['32']['vel_config']
            geometry = np.array(config['fixed_training_geometry'], dtype='f8')
            mapping = RadialCPU(p, t, geometry, config, log)
            reference, J, det, geometry_certificate = mapping.inverse(physical)
            log('complete_original_source_inverse_map', **geometry_certificate)
            cell, inv, location_certificate = locate_tetrahedra(p, t, reference, log)
            edges = np.sort(t[:, np.array(PAIRS)].reshape(-1, 2), axis=1)
            edges = np.unique(edges, axis=0)
            codes = edges[:, 0].astype('i8') * len(p) + edges[:, 1]
            local = np.sort(t[:, np.array(PAIRS)], axis=2)
            edgeids = np.searchsorted(codes, local[:, :, 0] * len(p) + local[:, :, 1])
            slots = np.full(len(edges), -1, dtype='i8')
            slots[free] = np.arange(len(free))
            local = slots[edgeids[cell]]
            gradients = np.concatenate((-inv.sum(1)[:, None, :], inv), axis=1)[cell]
            orientation = np.stack([1 - 2 * (t[cell, a] > t[cell, b]) for a, b in PAIRS], axis=1)
            curls = np.stack([2 * np.cross(gradients[:, a], gradients[:, b]) * orientation[:, j, None] for j, (a, b) in enumerate(PAIRS)], axis=1)
            curls = np.einsum('nai,nei->nea', J, curls) / det[:, None, None]
            rows = np.broadcast_to(3 * np.arange(len(physical))[:, None, None] + np.arange(3)[None, None, :], curls.shape)
            cols = np.broadcast_to(local[:, :, None], curls.shape)
            keep = cols >= 0
            B = sp.coo_matrix((curls[keep], (rows[keep], cols[keep])), shape=(3 * len(physical), len(free))).tocsr()
            B.sum_duplicates()
            B.eliminate_zeros()
            sp.save_npz(output / 'B_source_curl.npz', B)
            np.savez(output / 'source_inverse_map.npz', source_points=physical, reference_points=reference, J=J, det=det, cell_ids=cell, relative_edge_slots=local)
            G = sp.load_npz(ASSETS / 'G.npz')
            bg = B @ G
            gradient_relative = float(np.max(np.abs(bg.data), initial=0) / np.max(np.abs(B.data)))
            harmonic_relative = float(np.linalg.norm(B @ H) / max(np.linalg.norm(B.data) * np.linalg.norm(H), 1e-300))
            rng = np.random.default_rng(20261008)
            coefficient = rng.normal(size=(len(free), 2))
            cotangent = rng.normal(size=(3 * len(physical), 2))
            bc = B @ coefficient
            dot = abs(float(np.sum(bc * cotangent) - np.sum(coefficient * (B.T @ cotangent))))
            dot /= max(float(np.linalg.norm(bc) * np.linalg.norm(cotangent)), 1e-300)
            if max(gradient_relative, harmonic_relative, dot) >= 1e-10:
                raise AssertionError(('Physical curl sampling kernel/adjoint failure', gradient_relative, harmonic_relative, dot))
            inputs = {str(path): None for path in [SOURCE, CONFIG, ASSETS / 'complex.npz', ASSETS / 'G.npz', ROOT / 'flow/cases/metaball/piola_radial_geometry.py']}
            result = dict(status='complete', qualified=True, source_points=len(physical), all_points_retained=True, source_order_preserved=True, original_relative_edges=len(free), original_total_cells=len(t), B_shape=list(B.shape), B_nnz=B.nnz, B_sparse_bytes=int(B.data.nbytes + B.indices.nbytes + B.indptr.nbytes), gradient_kernel_relative=gradient_relative, harmonic_kernel_relative=harmonic_relative, transpose_dot_relative=dot, geometry_certificate=geometry_certificate, location_certificate=location_certificate, geometry=geometry.tolist(), inputs=inputs, outputs={str(path): None for path in [output / 'B_source_curl.npz', output / 'source_inverse_map.npz']}, integration_weights='None: unweighted evaluation B; production physical source weights must be independently bound', scope='Physical compatible Whitney curl operator at original103244 sources; no ANN, inverse solver, optimizer or whole2000 budget qualified', CPU_only=True, CUDA_imported=False, optimizer_updates=0, seconds=time.perf_counter() - start, pid=os.getpid(), creation=creation)
            write(output / 'complete.json', result)
            log('complete', qualified=True, B_nnz=B.nnz)
    except BaseException:
        write(output / 'failure.json', dict(error=traceback.format_exc(), pid=os.getpid(), creation=creation, seconds=time.perf_counter() - start))
        raise
    finally:
        done.set()
        watcher.join(timeout=3)
