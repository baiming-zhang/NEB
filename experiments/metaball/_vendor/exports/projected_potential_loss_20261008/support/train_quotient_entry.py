







from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from _neb_paths import Path
import sys
import threading
import time
import traceback
import types

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[1]
OLD_SHARED = ROOT / 'exports/potential_k128_resume_e1949_20261007/shared.py'
OLD_RUNTIME = OLD_SHARED.parent / 'runtime_vel_shared.py'
OLD_MATCHED = OLD_SHARED.parent / 'runtime_matched.py'
HISTORICAL_CONFIG = Path('_data/metaball/fixed_geometry_k128_20260921/config.json')
INITIAL = HISTORICAL_CONFIG.parent / 'checkpoint_e0000.npz'
K, SOURCE_POINTS, QUERY_POINTS, INDEPENDENT_POINTS = 32, 103244, 6144, 16384
EPOCHS, VALIDATIONS, CHECKPOINT_EVERY = 2000, 80, 100
VELOCITY_ACTUAL_SECONDS = 3042.3763707716716
NORMAL_TRAINING_REFERENCE_SECONDS = 3 * VELOCITY_ACTUAL_SECONDS
N_A = 'Curl-driven canonical quotient has no Coulomb multiplier, rho, or previous constraint residual.'














def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()













def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + '\n')
        stream.flush()


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module





























































































































































































class MethodPipeline:

    def __init__(self, protocol, config, source_points, independent_points, output, device_backend=None):
        import scipy.sparse as sp
        self.sp, self.protocol, self.output = sp, protocol, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        implementation = CAMPAIGN / 'implementation'
        sys.path.insert(0, str(implementation))
        selected_inverse = Path(protocol.get('canonical_lift_module', implementation / 'canonical_lift.py'))
        canonical = module_from_path('new_quotient_canonical_lift', selected_inverse)
        self.mapping = module_from_path('new_quotient_source_mapping', implementation / 'source_curl_mapping.py')
        metric = module_from_path('new_quotient_metric_objective', CAMPAIGN / 'math/projected_metric_objective.py')
        self.metric = metric.projected_metric_objective
        self.volume = float(protocol['physical_volume'])
        if not math.isfinite(self.volume) or self.volume <= 0:
            raise ValueError('Actual positive physical volume with bound provenance is required.')
        inverse_name = protocol.get('canonical_lift_class', 'CanonicalLift')
        if inverse_name not in ('CanonicalLift', 'TreeCanonicalLift'):
            raise ValueError('Bind the exact inverse factory class explicitly.')
        inverse_factory = getattr(canonical, inverse_name)
        self.inverse = inverse_factory(self.output / 'canonical', log=self.log,
            **protocol.get('canonical_factor_options', {}))
        self.solve_options = dict(protocol.get('canonical_solve_options', {}))
        if inverse_name == 'TreeCanonicalLift' and 'minimum_refinement' not in self.solve_options:
            raise ValueError('Bind the tree inverse minimum refinement from its actual qualification.')
        self.M = self.inverse.M
        self.metric_backend = protocol.get('metric_backend', 'numpy')
        if self.metric_backend not in ('numpy', 'cupy'):
            raise ValueError('The Potential-only metric backend must be explicitly numpy or cupy.')
        self.xp, self.metric_M = np, self.M
        if self.metric_backend == 'cupy':
            if device_backend is None:
                raise ValueError('The already loaded original native CuPy backend is required.')
            import cupyx.scipy.sparse as device_sparse
            self.xp, self.metric_M = device_backend, device_sparse.csr_matrix(self.M)
        assets = self.mapping.ASSETS
        with np.load(assets / 'complex.npz', allow_pickle=False) as z:
            self.p, self.t, self.free = z['p'].copy(), z['t'].copy(), z['free'].copy()
        geometry = np.asarray(config['fixed_training_geometry'], 'f8')
        self.radial = self.mapping.RadialCPU(self.p, self.t, geometry, config, self.log)
        self.locator = None
        if protocol.get('operator_static_mesh_cache', False):
            locator_module = module_from_path('new_quotient_static_locator', HERE / 'sampling_locator.py')
            self.locator = locator_module.ReadonlyTetrahedralLocator(self.p, self.t)
        pairs = np.array(self.mapping.PAIRS)
        edges = np.unique(np.sort(self.t[:, pairs].reshape(-1, 2), axis=1), axis=0)
        codes = edges[:, 0].astype('i8')*len(self.p) + edges[:, 1]
        local = np.sort(self.t[:, pairs], axis=2)
        edge_ids = np.searchsorted(codes, local[:, :, 0]*len(self.p)+local[:, :, 1])
        slots = np.full(len(edges), -1, dtype='i8'); slots[self.free] = np.arange(len(self.free))
        self.cell_slots = slots[edge_ids]

        source_path = Path(protocol['source_curl_operator'])
        self.source = sp.load_npz(source_path)
        if self.source.shape != (3*SOURCE_POINTS, len(self.free)):
            raise ValueError('Complete original source/relative-edge operator shape required.')
        with np.load(protocol['source_inverse_map'], allow_pickle=False) as z:
            if not np.array_equal(z['source_points'], source_points):
                raise ValueError('Actual source point order changed.')
        self.independent = self.operator(independent_points, 'fixed_independent16384')

    def log(self, phase, **values):
        append(self.output / 'operator_progress.jsonl', dict(phase=phase, unix=time.time(), **values))

    def operator(self, points, label):

        points = np.asarray(points, dtype='f8')
        reference, J, det, mapping_receipt = self.radial.inverse(points)
        if self.locator is None:
            cells, inv, location_receipt = self.mapping.locate_tetrahedra(self.p, self.t, reference, self.log)
        else:
            cells, inv, location_receipt = self.locator.locate(reference, self.log)
        gradients = np.concatenate((-inv.sum(1)[:, None, :], inv), axis=1)[cells]
        orientation = np.stack([1-2*(self.t[cells, a] > self.t[cells, b]) for a, b in self.mapping.PAIRS], axis=1)
        curls = np.stack([2*np.cross(gradients[:, a], gradients[:, b])*orientation[:, j, None]
            for j, (a, b) in enumerate(self.mapping.PAIRS)], axis=1)
        curls = np.einsum('nai,nei->nea', J, curls)/det[:, None, None]
        rows = np.broadcast_to(3*np.arange(len(points))[:, None, None]+np.arange(3)[None, None, :], curls.shape)
        columns = np.broadcast_to(self.cell_slots[cells][:, :, None], curls.shape)
        keep = columns >= 0
        B = self.sp.coo_matrix((curls[keep], (rows[keep], columns[keep])), shape=(3*len(points), len(self.free))).tocsr()
        B.sum_duplicates(); B.eliminate_zeros()

        bg, bh = B @ self.inverse.G, B @ self.inverse.H
        gradient = float(np.max(np.abs(bg.data), initial=0)/np.max(np.abs(B.data)))
        harmonic = float(np.linalg.norm(bh)/max(np.linalg.norm(B.data)*np.linalg.norm(self.inverse.H), 1e-300))
        if max(gradient, harmonic) >= 1e-10:
            raise FloatingPointError('New sampled curl operator failed complete gradient/harmonic kernel qualification.')
        self.log(label, points=len(points), points_sha256=array_sha(points), B_shape=list(B.shape),
            kernel_gradient=gradient, kernel_harmonic=harmonic, mapping=mapping_receipt, location=location_receipt)
        return B

    def evaluate(self, source_u, other_u, other_B, *, reverse, label):
        Us, Uq = np.asarray(source_u, 'f8'), np.asarray(other_u, 'f8')
        n = (len(Us)+len(Uq))//3
        if Us.shape != (3*SOURCE_POINTS, K) or Uq.shape[1] != K or len(Uq)%3:
            raise ValueError('All source+query fields and all 32 modes are required.')
        weight = self.volume/n
        rhs = weight*(self.source.T @ Us + other_B.T @ Uq)
        Z, forward = self.inverse.solve(rhs, label=label+'_full32_lift', **self.solve_options)
        U = np.concatenate((Us, Uq), axis=0)
        xp = self.xp
        objective = self.metric(xp.asarray(Z), xp.asarray(U), self.metric_M, self.volume, n, xp=xp)
        result = dict(objective=objective, lift_forward=forward, source_points=SOURCE_POINTS,
            other_points=len(Uq)//3, union_points=n, positive_point_weight=weight,
            numerator_and_lift_RHS_share_identical_union=True,
            full_linear_fields=dict(source_U=Us,other_U=Uq,lift_RHS=rhs,lift_Z=Z))
        if reverse:
            gz = objective['GZ'] if self.metric_backend == 'numpy' else xp.asnumpy(objective['GZ'])
            direct = objective['GU'] if self.metric_backend == 'numpy' else xp.asnumpy(objective['GU'])
            AZ, adjoint = self.inverse.adjoint(gz, label=label+'_full32_adjoint', **self.solve_options)

            split = len(Us)
            result['GU_source'] = direct[:split] + weight*(self.source @ AZ)
            result['GU_other'] = direct[split:] + weight*(other_B @ AZ)
            result['lift_adjoint'] = adjoint
        return result

    def close(self):
        self.inverse.close()


def synchronized(cm, phases, key, function):
    cm.cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    value = function()
    cm.cp.cuda.get_current_stream().synchronize()
    phases[key] = time.perf_counter()-start
    return value
















def full_forward_reverse(cm, pipeline, state, epoch):

    phases, c, model = {}, state['c'], state['model']
    points = synchronized(cm, phases, 'original_query_sampler', lambda: cm.rt.fluid_points(c, state['sampling_geometry'], QUERY_POINTS, c.seed+100000+epoch))

    query = synchronized(cm, phases, 'original_query_geometry', lambda: cm.rt.NeuralPoints(points, state['geometry'], c, 'f4'))
    source = synchronized(cm, phases, 'original_source_geometry', lambda: cm.rt.NeuralPoints(state['source_points'], state['geometry'], c, 'f4'))
    uq = synchronized(cm, phases, 'original_query_ANN_forward', lambda: cm.rt.grouped8.network_fields(model, query, 'u')['u'])
    us = synchronized(cm, phases, 'original_source_ANN_forward', lambda: cm.rt.grouped8.network_fields(model, source, 'u')['u'])
    Bq = synchronized(cm, phases, 'Potential_added_query_curl_sampling_operator', lambda: pipeline.operator(cm.cp.asnumpy(points), f'epoch_{epoch}_query6144'))
    result = synchronized(cm, phases, 'Potential_added_full32_lift_metric_adjoint', lambda: pipeline.evaluate(cm.cp.asnumpy(us), cm.cp.asnumpy(uq), Bq, reverse=True, label=f'epoch_{epoch}'))
    gradients = synchronized(cm, phases, 'original_query_ANN_parameter_reverse', lambda: cm.rt.grouped8.network_backward(model, query, curl=cm.cp.asarray(result['GU_other'], 'f4')))
    gradients = synchronized(cm, phases, 'original_source_ANN_parameter_reverse', lambda: cm.rt.grouped8.network_backward(model, source, curl=cm.cp.asarray(result['GU_source'], 'f4'), grads=gradients))
    return result, gradients, phases


def scheduled_validation(cm, pipeline, state, epoch):

    c, model, rows, phases = state['c'], state['model'], [], {}
    start = time.perf_counter()
    for index in range(8):
        token, seed = 10000000+index, c.seed+9000000+index
        geo = cm.cp.asarray(cm.geometry_from_token(c, token), 'f8')
        points = cm.rt.fluid_points(c, geo, 4096, seed)
        data = cm.rt.NeuralPoints(points, geo, c, 'f4')
        fields = cm.rt.grouped8.network_fields(model, data, 'au')
        u = fields['u'].astype('f8')
        mass = cm.cp.zeros((K, K), 'f8')
        for lo in range(0, len(u), 3072):
            block = u[lo:lo+3072]
            mass += block.T @ block / 4096
        row = dict(index=index, token=token, sample_seed=seed, points=4096,
            geometry_sha256=array_sha(cm.cp.asnumpy(geo)), points_sha256=array_sha(cm.cp.asnumpy(points)),
            true_analytic_velocity_mass_diagonal=cm.cp.asnumpy(cm.cp.diag(mass)).tolist(),
            true_analytic_velocity_mass_eigenvalues=cm.cp.asnumpy(cm.cp.linalg.eigvalsh((mass+mass.T)*.5)).tolist(),
            new_quotient_lift_evaluated=False,
            scope='Historical shared geometry and ANN fields; actual velocity mass and legacy raw-A diagnostics.')
        try:
            legacy = cm.train.bounded_potential_objective(fields['a'], fields['u'], 4096)
            row.update(legacy_raw_A_Rayleigh_mean=float(cm.cp.mean(legacy['losses'])),
                legacy_raw_A_canonical_gram=float(legacy['gram_max']))
        except (ValueError, FloatingPointError) as error:
            row['legacy_raw_A_diagnostic_rejection'] = str(error)
        rows.append(row)
        del data, fields, u, mass
        cm.clear()
    cm.cp.cuda.get_current_stream().synchronize()
    phases['original_eight_geometry_shared_fields_and_legacy_diagnostics'] = time.perf_counter()-start
    def fixed():
        source = cm.rt.NeuralPoints(state['source_points'], state['geometry'], c, 'f4')
        independent = cm.rt.NeuralPoints(state['independent_points'], state['geometry'], c, 'f4')
        us = cm.rt.grouped8.network_fields(model, source, 'u')['u']
        ui = cm.rt.grouped8.network_fields(model, independent, 'u')['u']
        return pipeline.evaluate(cm.cp.asnumpy(us), cm.cp.asnumpy(ui), pipeline.independent, reverse=False, label=f'validation_{epoch}_fixed_source_independent')
    result = synchronized(cm, phases, 'extra_fixed_source103244_independent16384_true_new_quotient', fixed)
    objective = result['objective']
    return dict(epoch=epoch, original_geometry_diagnostics=rows,
        fixed_geometry_new_quotient_mean=float(objective['loss_mean']),
        fixed_geometry_new_quotient_all_modes=K, fixed_source_points=SOURCE_POINTS,
        independent_points=INDEPENDENT_POINTS, fixed_gram_max=float(objective['gram_max']),
        fixed_minimum_relative_mass=float(np.min(objective['relative_masses'])), phases=phases)
