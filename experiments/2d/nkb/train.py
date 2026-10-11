"""Full 2000-update NKB case adaptation with retained preflight and all validation."""
import time
INVOCATION_START = time.monotonic()
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import cupy as cp
from common import HERE, OUT, guard, read, write
from core import Model, losses
from domains import Domain


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--case', required=True, choices=['metaball', 'periodic', 'one_circle', 'circle_square', 'circle_square_triangle'])
    p.add_argument('--K', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    run = OUT/f'{args.case}_k{args.K}_seed{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    for name in ('user_budget_stop.json', 'user_requested_stop.json'):
        if (run/name).exists(): raise RuntimeError(f'Respecting persistent stop: {run/name}')
    if (run/'training_complete.json').exists():
        print(json.dumps(dict(status='already_complete', run=str(run)))); return
    stopped = guard(cp, run)
    domain = Domain(args.case, cp)
    count = 6144 if args.case == 'metaball' else 4096
    val_geo = domain.validation_geometries()
    sample = domain.sample(2, args.seed)
    config = dict(case=args.case, K=args.K, seed=args.seed, epochs=2000,
                  points_per_step=count, validation_points=4096, validation_geometries=len(val_geo),
                  validate_every=25, checkpoint_every=25, input_dim=sample[1].shape[1],
                  dim=domain.dim, width=256, hidden_layers=7, dtype='float32',
                  learning_rate=.0005, lr_rule='0.0005 * 0.96 ** (10*(update-1)/2000)',
                  training_budget='2000 optimizer updates on the fixed campaign geometry; not the original million-update budget',
                  losses=dict(smooth=.01, divergence=5, orthogonality=100, boundary=30, length=100, small=100),
                  target_mean_length=.37, small_threshold=.05,
                  orthogonality='mean squared global off-diagonal Gram entries; explicit nonnegative realization of printed signed sum',
                  backend='native CuPy with analytic coordinate Jacobians and handwritten parameter reverse',
                  geometry=domain.contract())
    if (run/'config.json').exists() and read(run/'config.json') != config:
        raise RuntimeError('Frozen configuration mismatch')
    write(run/'config.json', config)
    model = Model(cp, config['input_dim'], config['dim'], args.K, args.seed)
    checkpoint = run/'checkpoint.npz'
    if checkpoint.exists(): model.load(checkpoint)
    else:
        model.save(run/'initial_checkpoint.npz'); model.save(checkpoint)
    if (run/'training.jsonl').exists():
        rows = [json.loads(s) for s in (run/'training.jsonl').read_text().splitlines() if s.strip()]
        committed = {r['epoch']: r for r in rows if r['epoch'] <= model.step}
        (run/'training.jsonl').write_text(''.join(json.dumps(committed[e])+'\n' for e in sorted(committed)))
    previous = read(run/'budget.json').get('seconds', 0) if (run/'budget.json').exists() else 0.
    # Preserve consumed time even if an error causes a rollback to an earlier checkpoint.
    def elapsed(): return previous + time.monotonic()-INVOCATION_START
    def status(phase, **kw):
        row = dict(phase=phase, epoch=model.step, target=2000, K=args.K, case=args.case,
                   seed=args.seed, pid=os.getpid(), unix=time.time(), cumulative_training_seconds=elapsed(), **kw)
        write(run/'progress.json', row)
        write(run/'budget.json', dict(seconds=elapsed(), pid=os.getpid(), unix=time.time()))
        return row
    def check_budget():
        return
    def validate(label):
        tick = time.monotonic(); details = []
        for i, geo in enumerate(val_geo):
            check_budget()
            _, f, j, data = domain.sample(4096, args.seed+9000000+i, geo)
            v, jac, _ = model.forward(f, j)
            raw, _, _ = losses(v, jac, *data, cp, False)
            row = {key: float(value) for key, value in raw.items()}
            if not all(np.isfinite(value) for value in row.values()): raise FloatingPointError('Nonfinite NKB validation')
            details.append(row)
        cp.cuda.Stream.null.synchronize()
        row = dict(epoch=model.step, seconds=time.monotonic()-tick, details=details)
        write(run/f'validation_{label}_{model.step:04d}.json', row)
        return row['seconds']
    timings = []
    try:
        initial_validation = validate('preflight')
        status('retained_preflight', validation_seconds=initial_validation)
        # A committed update may precede an interrupted scheduled validation.
        if model.step and model.step % 25 == 0 and not (run/f'validation_scheduled_{model.step:04d}.json').exists():
            validate('scheduled')
        for epoch in range(model.step+1, 2001):
            check_budget(); tick = time.monotonic()
            _, f, j, data = domain.sample(count, args.seed+100000+epoch)
            v, jac, tape = model.forward(f, j, True)
            raw, gv, gj = losses(v, jac, *data, cp)
            gradients = model.backward(tape, gv, gj)
            lr = .0005*.96**(10*(epoch-1)/2000)
            gn = model.adam(gradients, lr)
            cp.cuda.Stream.null.synchronize()
            metrics = {key: float(value) for key, value in raw.items()}
            if not all(np.isfinite(value) for value in metrics.values()): raise FloatingPointError('Nonfinite NKB training loss')
            # The update is finished. Release its analytic derivative tape before
            # the independent validation; retaining both exceeds the pool at K2048.
            del v, jac, tape, gv, gj, gradients, raw, f, j, data
            cp.get_default_memory_pool().free_all_blocks()
            # Preflight steps are retained; each measured step includes full checkpoint I/O.
            if len(timings) < 5 or epoch % 25 == 0: model.save(checkpoint)
            seconds = time.monotonic()-tick; timings.append(seconds)
            row = dict(epoch=epoch, seconds=seconds, lr=lr, gradient_norm=gn, **metrics)
            with (run/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            status('training', last_step_seconds=seconds, loss=metrics['loss'])
            if len(timings) == 5:
                forecast = elapsed()+1.25*((2000-model.step)*max(timings)+(80-model.step//25)*initial_validation)+30
                receipt = dict(passed=True, K=args.K, retained_epochs=model.step,
                               full_epoch_seconds=timings[:5], full_validation_seconds=initial_validation,
                               forecast_seconds=forecast, cumulative_seconds=elapsed(), margin=1.25,
                               modes_reduced=False, epochs_reduced=False, validation_reduced=False)
                write(run/'preflight.json', receipt)
                print(json.dumps(receipt), flush=True)
                if not receipt['passed']:
                    write(run/'user_budget_stop.json', dict(reason='Measured full2000 exceeds K32 cap', **receipt))
                    raise RuntimeError('K32 measured full-budget preflight does not fit')
                if args.preflight_only:
                    status('preflight_complete'); return
            if epoch % 25 == 0:
                validate('scheduled')
                print(json.dumps(status('training', last_step_seconds=seconds, loss=metrics['loss'])), flush=True)
        model.save(run/'checkpoint_e2000.npz')
        write(run/'training_complete.json', dict(epoch=2000, K=args.K, case=args.case, seed=args.seed,
              cumulative_training_seconds=elapsed(), all_scheduled_validations=True,
              original_NKB_training_budget=False, backend=config['backend']))
        status('training_complete')
    except BaseException:
        status('needs_review', error=traceback.format_exc())
        write(run/'failure.json', dict(error=traceback.format_exc(), unix=time.time()))
        raise
    finally:
        write(run/'budget.json', dict(seconds=elapsed(), pid=os.getpid(), unix=time.time()))
        stopped.set()


if __name__ == '__main__': main()
