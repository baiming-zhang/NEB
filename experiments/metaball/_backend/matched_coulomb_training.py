from __future__ import annotations
import math, time

def make_state(cm):
    """"""
    c, geo, x, q = cm.inputs()
    model, initialization = fresh_model(cm, c)
    source, assessment = (cm.rt.NeuralPoints(x, geo, c, 'f4'), cm.rt.NeuralPoints(q, geo, c, 'f4'))
    h, hq = (cm.wall_test_scalar(x, geo, c), cm.wall_test_scalar(q, geo, c))
    initial = cm.metrics(model, assessment, hq)
    validation_geometry = []
    for i in range(c.validation_geometries):
        token, sample_seed = (10000000 + i, c.seed + 9000000 + i)
        validation_geometry.append(dict(index=i, token=token, sample_seed=sample_seed))
    return dict(c=c, geo=geo, sampling_geo=cm.cp.asarray(c.fixed_training_geometry, 'f4'), x=x, q=q, model=model, source=source, assessment=assessment, h=h, hq=hq, initial=initial, initialization=initialization, velocity_validation=validation_geometry, scale=initial['mean_potential_Rayleigh'], rho=1.0, lam=cm.cp.zeros((len(x), model.K), 'f8'), eta=cm.cp.zeros(model.K, 'f8'), previous_residual=None)

def learning_rate(c, epoch):
    return c.min_learning_rate + (c.learning_rate - c.min_learning_rate) * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / 1999))

def full_epoch(cm, span, state, epoch, lr=None):
    """"""
    cp, model, c = (cm.cp, state['model'], state['c'])
    if model.epoch != epoch - 1:
        raise ValueError('Epochs must advance consecutively from fresh epoch zero')
    tic = time.perf_counter()
    sample_seed = c.seed + 100000 + epoch
    query = cm.rt.NeuralPoints(cm.rt.fluid_points(c, state['sampling_geo'], 6144, sample_seed), state['geo'], c, 'f4')
    fq = cm.rt.grouped8.network_fields(model, query, 'au')
    obj = cm.train.bounded_potential_objective(fq['a'], fq['u'], 6144)
    grads = cm.rt.grouped8.network_backward(model, query, value=obj['GA'], curl=obj['GU'])
    f = cm.rt.grouped8.network_fields(model, state['source'], 'ad')
    cost, ga, gd, residual, moment, info = span.span_augmented(cp, f['a'], f['d'], state['h'], state['lam'], state['eta'], state['rho'], state['scale'])
    grads = cm.rt.grouped8.network_backward(model, state['source'], value=ga, div=gd, grads=grads)
    lr = learning_rate(c, epoch) if lr is None else lr
    norms = model.adam(*grads, lr, c.grad_clip)
    if model.epoch != epoch:
        raise RuntimeError('Optimizer epoch mismatch')
    row = dict(epoch=epoch, sample_seed=sample_seed, lr=lr, rho_used=state['rho'], physical_Rayleigh_mean=float(cp.mean(obj['losses'])), constraint_cost_mean=float(cp.mean(cost)), source_raw_div_rms=float(cp.sqrt(cp.mean(residual * residual))), source_weak_wall_moment_max=float(cp.max(cp.abs(moment))), sort=cp.asnumpy(obj['order']).tolist(), gradient_min=float(norms.min()), gradient_max=float(norms.max()), span_metric=info, query_points=6144, constraint_source_points=103244, all_modes=model.K)
    del query, fq, obj, grads, f, cost, ga, gd, residual, moment, norms
    cm.clear()
    row['seconds'] = time.perf_counter() - tic
    return row

def full_dual_update(cm, span, state, epoch):
    cp = cm.cp
    tic = time.perf_counter()
    f = cm.rt.grouped8.network_fields(state['model'], state['source'], 'ad')
    dual_only = getattr(span, 'span_dual_only', None)
    if dual_only is None:
        _, _, _, residual, moment, info = span.span_augmented(cp, f['a'], f['d'], state['h'], state['lam'], state['eta'], state['rho'], state['scale'])
    else:
        residual, moment, info = dual_only(cp, f['a'], f['d'], state['h'], state['lam'], state['eta'], state['rho'], state['scale'])
    rms = float(cp.sqrt(cp.mean(residual * residual)))
    state['lam'] += state['rho'] * residual
    state['eta'] += state['rho'] * moment
    entry = dict(epoch=epoch, rho_used=state['rho'], raw_div_rms=rms, lambda_norm=float(cp.linalg.norm(state['lam'])), eta_norm=float(cp.linalg.norm(state['eta'])), span=info)
    if state['previous_residual'] is not None and rms > 0.75 * state['previous_residual']:
        state['rho'] = min(8.0, state['rho'] * 2)
    state['previous_residual'] = rms
    entry['rho_next'] = state['rho']
    del f, residual, moment
    cm.clear()
    entry['seconds'] = time.perf_counter() - tic
    return entry

def _fields_metrics(cm, fields, data, h):
    cp = cm.cp
    fields = {key: cp.asarray(value, dtype='f8') for key, value in fields.items()}
    mass = cp.sum(fields['a'] * fields['a'], axis=0) / data.n
    curlmass = cp.sum(fields['u'] * fields['u'], axis=0) / data.n
    divmass = cp.sum(fields['d'] * fields['d'], axis=0) / data.n
    moment = (h.T @ fields['a']).reshape(-1) / (data.n * cp.sqrt(mass) * cp.sqrt(cp.sum(h * h) / data.n))
    obj = cm.train.bounded_potential_objective(fields['a'], fields['u'], data.n)
    row = dict(points=data.n, all_modes=fields['a'].shape[1], ANN_precision=str(data.x.dtype), mean_potential_Rayleigh=float(cp.mean(obj['losses'])), gram_max=float(obj['gram_max']), divergence_over_potential_mass=cm.values(divmass / mass), divergence_over_curl_mass=cm.values(divmass / cp.maximum(curlmass, 1e-300)), divergence_over_curl_summary=cm.distribution(divmass / cp.maximum(curlmass, 1e-300)), weak_wall_scalar_moment=cm.values(moment), weak_wall_scalar_moment_max=float(cp.max(cp.abs(moment))), potential_mass=cm.values(mass), curl_mass=cm.values(curlmass))
    return (row, obj)

def full_validation(cm, state, epoch):
    """"""
    cp, model, c = (cm.cp, state['model'], state['c'])
    tic, details = (time.perf_counter(), [])
    if len(state['velocity_validation']) != 8:
        raise ValueError('Complete original eight-geometry validation inputs are required')
    for saved in state['velocity_validation']:
        i, token, sample_seed = (saved['index'], saved['token'], saved['sample_seed'])
        geometry = cm.geometry_from_token(c, token)
        points = cm.rt.fluid_points(c, geometry, c.validation_points, sample_seed)
        data = cm.rt.NeuralPoints(points, geometry, c, 'f4')
        fields = cm.rt.grouped8.network_fields(model, data, 'au')
        obj = cm.train.bounded_potential_objective(fields['a'], fields['u'], data.n)
        physical_a = cp.asarray(fields['a'], dtype='f8')
        raw_norm = cp.sqrt(cp.sum(physical_a * physical_a, axis=0) / data.n)
        indices = obj['order']
        relative_mass = 1 / (raw_norm[indices] * obj['T'][indices, cp.arange(model.K)]) ** 2
        details.append(dict(index=i, token=token, sample_seed=sample_seed, points=data.n, all_modes=model.K, geometry=geometry.tolist(), mean_rayleigh=float(cp.mean(obj['losses'])), gram_max=float(obj['gram_max']), minimum_relative_mass=float(relative_mass.min()), objective='Original physical Potential curl/potential Rayleigh; geometry-local two-pass GS'))
        del fields, physical_a, obj, raw_norm, indices, relative_mass, data, points, geometry
        cm.clear()
    velocity_schedule_seconds = time.perf_counter() - tic
    source, assessment = (state['source'], state['assessment'])
    fs = cm.rt.grouped8.network_fields(model, source, 'aud')
    fq = cm.rt.grouped8.network_fields(model, assessment, 'aud')
    src, obj = _fields_metrics(cm, fs, source, state['h'])
    val, independent_obj = _fields_metrics(cm, fq, assessment, state['hq'])
    transform, canon = (obj['T'], {})
    for name, fields, n in [('source', fs, source.n), ('independent', fq, assessment.n)]:
        a, u, d = (cp.asarray(fields[key], dtype='f8') @ transform for key in ('a', 'u', 'd'))
        canon[name] = dict(points=n, all_modes=model.K, div_curl=cm.distribution(cp.sum(d * d, 0) / cp.sum(u * u, 0)), div_mass=cm.distribution(cp.sum(d * d, 0) / cp.sum(a * a, 0)))
    canon['scope'] = 'Fixed103244 source GS transform applied unchanged to original fixed16384 independent fields'
    result = dict(epoch=epoch, validation_index=epoch // 25, seed=c.seed, validation_geometries=8, validation_points_per_geometry=4096, details=details, full8geometry4096_validation_seconds=velocity_schedule_seconds, source=src, assessment=val, canonical=canon, all_modes=model.K, assessment_seed=919000, scope='Original velocity8x4096 schedule plus complete fixed103244/16384 sampled-Coulomb independent diagnostics')
    del fs, fq, obj, independent_obj, transform, a, u, d
    cm.clear()
    result['seconds'] = time.perf_counter() - tic
    return result

def fresh_model(cm, c):
    model = cm.train.Model(c, cm._matched_K, c.seed)
    return (model, dict(seed=c.seed, K=model.K, epoch=0))
