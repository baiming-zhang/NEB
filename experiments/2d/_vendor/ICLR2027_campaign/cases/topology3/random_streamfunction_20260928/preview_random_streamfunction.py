"""CPU-only direct random-streamfunction preview; no GT, training, or ROM.

IID physical-grid values are interpolated by a C4 tensor quintic spline.
Local, analytic boundary conditioning gives constant traces and zero boundary
Laplacian, without a harmonic solve or an energy-ratio prescription. All
production spatial derivatives are analytic truncated Taylor arithmetic.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / 'baseline_fresh_v1' / 'src'))
import geometry as geo

CASES = ('one_circle', 'circle_square', 'circle_square_triangle')
COLLAR = .35
LENGTH = 4.


class Jet:
    """Taylor coefficients d_x^i d_y^j f/(i!j!), total order <= n."""
    def __init__(self, values, n=5):
        self.n = n
        self.keys = [(i, total-i) for total in range(n+1) for i in range(total+1)]
        self.index = {key: k for k, key in enumerate(self.keys)}
        values = np.asarray(values, dtype=float)
        self.a = np.zeros((len(values), len(self.keys)))
        self.a[:, 0] = values

    @classmethod
    def variable(cls, values, axis, n=5):
        obj = cls(values, n)
        obj.a[:, obj.index[(1, 0) if axis == 0 else (0, 1)]] = 1.
        return obj

    def truncate(self, n):
        obj = Jet(self.a[:, 0], n)
        obj.a[:] = self.a[:, :len(obj.keys)]
        return obj

    def derivative(self, axis):
        obj = Jet(self.a[:, 0] * 0, self.n-1)
        for k, (i, j) in enumerate(obj.keys):
            src = (i+1, j) if axis == 0 else (i, j+1)
            obj.a[:, k] = self.a[:, self.index[src]] * (i+1 if axis == 0 else j+1)
        return obj

    def __add__(self, other):
        obj = self.truncate(self.n)
        if isinstance(other, Jet):
            assert self.n == other.n
            obj.a += other.a
        else:
            obj.a[:, 0] += other
        return obj

    __radd__ = __add__

    def __neg__(self):
        obj = self.truncate(self.n)
        obj.a *= -1
        return obj

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        obj = Jet(self.a[:, 0]*0, self.n)
        if not isinstance(other, Jet):
            obj.a = self.a * np.asarray(other)[..., None] if np.ndim(other) else self.a * other
            return obj
        assert self.n == other.n
        for k, (i, j) in enumerate(self.keys):
            for p in range(i+1):
                for q in range(j+1):
                    obj.a[:, k] += self.a[:, self.index[(p, q)]] * other.a[:, self.index[(i-p, j-q)]]
        return obj

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (other.power(-1) if isinstance(other, Jet) else 1/other)

    def compose(self, coefficients):
        residual = self.truncate(self.n)
        residual.a[:, 0] = 0
        result = Jet(coefficients[0], self.n)
        term = Jet(np.ones(len(self.a)), self.n)
        for k in range(1, self.n+1):
            term = term * residual
            result.a += term.a * coefficients[k][:, None]
        return result

    def power(self, exponent):
        # Nonnegative integer powers have no zero-base singularity.
        if int(exponent) == exponent and exponent >= 0:
            result = Jet(np.ones(len(self.a)), self.n)
            base = self
            count = int(exponent)
            while count:
                if count & 1:
                    result = result * base
                count >>= 1
                if count:
                    base = base * base
            return result
        value = self.a[:, 0]
        coefficients = []
        factor = 1.
        for k in range(self.n+1):
            if k:
                factor *= (exponent-k+1)/k
            coefficients.append(factor * value**(exponent-k))
        return self.compose(coefficients)

    def log(self):
        value = self.a[:, 0]
        return self.compose([np.log(value)] + [((-1)**(k-1)/k)*value**(-k) for k in range(1, self.n+1)])

    def exp(self):
        value = np.exp(self.a[:, 0])
        return self.compose([value/math.factorial(k) for k in range(self.n+1)])


def coordinates(points, case):
    x, y = (Jet.variable(points[:, i], i) for i in range(2))
    q = [1-((x-5)/4).power(12)-(y/4).power(12)]
    for spec in geo.get_obstacles(case):
        c, s = np.cos(spec.rotation), np.sin(spec.rotation)
        xx = c*(x-spec.x)+s*(y-spec.y)
        yy = -s*(x-spec.x)+c*(y-spec.y)
        if spec.kind == 'circle':
            value = (xx.power(2)+yy.power(2))/(spec.size**2)-1
        elif spec.kind == 'square':
            value = (xx/spec.size).power(spec.power)+(yy/spec.size).power(spec.power)-1
        else:
            planes = [(np.cos(a)*xx+np.sin(a)*yy)/spec.size-1
                      for a in np.pi/2+2*np.pi*np.arange(3)/3]
            shift = np.max(np.stack([p.a[:, 0] for p in planes]), axis=0)
            value = sum(((p-shift)*spec.smooth_k).exp() for p in planes).log()/spec.smooth_k+shift
        q.append(value)
    return q


def loops_for(case, n=1536):
    return [geo.outer_boundary(0., n)] + [geo.obstacle_boundary(s, n) for s in geo.get_obstacles(case)]


class RandomField:
    def __init__(self, case, sample=0):
        self.case = case
        self.sample = sample
        self.seed = [20260928, CASES.index(case), sample]
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        self.x = np.linspace(-1, 11, 25)
        self.y = np.linspace(-6, 6, 25)
        self.raw = self.rng.standard_normal((25, 25))
        self.spline = RectBivariateSpline(self.x, self.y, self.raw, kx=5, ky=5, s=0)
        self.loops = loops_for(case)
        self.constants = []
        for loop in self.loops:
            ds = .5*(np.linalg.norm(loop-np.roll(loop, 1, axis=0), axis=1)
                     +np.linalg.norm(loop-np.roll(loop, -1, axis=0), axis=1))
            self.constants.append(float(np.average(self.spline.ev(loop[:, 0], loop[:, 1]), weights=ds)))
        self.scale = 1.

    def spline_jet(self, points):
        result = Jet(np.zeros(len(points)), 4)
        for k, (i, j) in enumerate(result.keys):
            result.a[:, k] = self.spline.ev(points[:, 0], points[:, 1], dx=i, dy=j)/(math.factorial(i)*math.factorial(j))
        return result

    def evaluate_block(self, points):
        q = coordinates(points, self.case)
        # Bounded defining functions keep the transfinite weights well scaled.
        s = [(v/(1+v.power(2)).power(.5)).truncate(4) for v in q]
        products = []
        for j in range(len(s)):
            prod = Jet(np.ones(len(points)), 4)
            for i, value in enumerate(s):
                if i != j:
                    prod = prod*value
            products.append(prod)
        denominator = sum(products)
        raw = self.spline_jet(points)
        preliminary = raw
        correction_factor = Jet(np.zeros(len(points)), 2)
        for j, value in enumerate(q):
            ell = (1+value).log()
            gx, gy = ell.derivative(0), ell.derivative(1)
            ell = ell.truncate(4)
            norm2 = gx.power(2)+gy.power(2)+(ell/LENGTH).power(2)+1e-30
            distance2 = ell.power(2)/norm2
            weight = (products[j]/denominator).power(3)
            collar = (-(distance2/(COLLAR**2)).power(2)).exp()
            blend = weight*collar
            preliminary = preliminary+blend*(self.constants[j]-raw)
            correction_factor = correction_factor+(distance2*blend*.5).truncate(2)
        lap = preliminary.derivative(0).derivative(0)+preliminary.derivative(1).derivative(1)
        field = preliminary.truncate(2)-correction_factor*lap
        field.a *= self.scale
        psi = field.a[:, 0]
        ux = field.a[:, field.index[(0, 1)]]
        uy = -field.a[:, field.index[(1, 0)]]
        omega = -2*(field.a[:, field.index[(2, 0)]]+field.a[:, field.index[(0, 2)]])
        return np.column_stack((psi, ux, uy, omega))

    def evaluate(self, points, chunk=1024):
        points = np.asarray(points)
        output = np.empty((len(points), 4))
        for start in range(0, len(points), chunk):
            output[start:start+chunk] = self.evaluate_block(points[start:start+chunk])
        return output


def checks(field):
    # Shift the angular grid: these are independent of the nodes used for means.
    records = []
    for j, loop in enumerate(loops_for(field.case, 2053)):
        values = field.evaluate(loop)
        grad = geo.outer_gradient(loop) if j == 0 else geo.obstacle_gradient(loop, geo.get_obstacles(field.case)[j-1])
        normal = grad/np.linalg.norm(grad, axis=1)[:, None]
        records.append(dict(component=j, trace_spread=float(np.ptp(values[:, 0])),
                            trace_expected=field.scale*field.constants[j],
                            trace_max_error=float(np.max(abs(values[:, 0]-field.scale*field.constants[j]))),
                            normal_speed_max=float(np.max(abs(np.sum(values[:, 1:3]*normal, axis=1)))),
                            boundary_vorticity_max=float(np.max(abs(values[:, 3])))))
    # Finite differences are validation only, not production derivatives.
    rng = np.random.default_rng(47321)
    candidates = np.column_stack((rng.uniform(1.1, 8.9, 60), rng.uniform(-3.9, 3.9, 60)))
    points = candidates[geo.fluid_mask(candidates, field.case)][:10]
    eps = 2e-5
    center = field.evaluate(points)
    shifted = [field.evaluate(points+direction) for direction in (np.array([eps, 0]), np.array([-eps, 0]), np.array([0, eps]), np.array([0, -eps]))]
    px, mx, py, my = shifted
    numerical_velocity = np.column_stack(((py[:, 0]-my[:, 0])/(2*eps), -(px[:, 0]-mx[:, 0])/(2*eps)))
    numerical_omega = (px[:, 2]-mx[:, 2]-py[:, 1]+my[:, 1])/(2*eps)
    verror = np.linalg.norm(center[:, 1:3]-numerical_velocity)/max(np.linalg.norm(center[:, 1:3]), 1e-15)
    werror = np.linalg.norm(center[:, 3]-numerical_omega)/max(np.linalg.norm(center[:, 3]), 1e-15)
    assert max(r['normal_speed_max'] for r in records) < 1e-8
    assert max(r['boundary_vorticity_max'] for r in records) < 1e-7
    assert max(r['trace_max_error'] for r in records) < 1e-9
    assert verror < 1e-5 and werror < 1e-5, (verror, werror)
    return dict(boundaries=records, analytic_velocity_relative_check=float(verror),
                analytic_vorticity_relative_check=float(werror), passed=True,
                divergence='Mixed derivatives of the same C2 scalar cancel analytically',
                scope='Initial-field kinematics only; no NS trajectory, GT convergence, harmonic energy or method errors evaluated')


def decorate(ax, case):
    from matplotlib.patches import Polygon
    boundary = geo.outer_boundary(0, 1536)
    ax.plot(*np.vstack((boundary, boundary[0])).T, color='black', lw=.75, zorder=6)
    for spec in geo.get_obstacles(case):
        ax.add_patch(Polygon(geo.obstacle_boundary(spec, 512), facecolor='#ECEDEF', edgecolor='black', lw=.75, zorder=7))
    ax.set(xlim=(.9, 9.1), ylim=(-4.1, 4.1), aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_all(payloads, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm, SymLogNorm
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    speedmap = LinearSegmentedColormap.from_list('speed', ['#F7FAFC', '#AECDE1', '#578DB7', '#245682'])
    vortmap = LinearSegmentedColormap.from_list('vorticity', ['#326C9B', '#B8D1E3', '#FCFCFA', '#EABCB8', '#BE4B45'])
    wmax = max(float(np.nanmax(abs(p['omega']))) for p in payloads)
    speed_norm = PowerNorm(.5, vmin=0, vmax=.5)
    omega_norm = SymLogNorm(linthresh=.25, linscale=.75, vmin=-wmax, vmax=wmax, base=10)
    speed_levels = .5*np.linspace(0, 1, 33)**2
    positive_levels = np.geomspace(.05, wmax, 25)
    omega_levels = np.r_[-positive_levels[::-1], 0., positive_levels]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.1), constrained_layout=True)
    for col, data in enumerate(payloads):
        case = data['case']
        x, y = data['x'], data['y']
        speed = np.hypot(data['ux'], data['uy'])
        # Fully vector contour fields in PDF; PNG is an export of these same data.
        top = axes[0, col].contourf(x, y, speed, levels=speed_levels, cmap=speedmap, norm=speed_norm)
        axes[0, col].streamplot(x, y, np.ma.masked_invalid(data['ux']), np.ma.masked_invalid(data['uy']),
                              color='#1B2730', linewidth=.48, density=1.35, arrowsize=.6, minlength=.035, maxlength=1.8)
        bottom = axes[1, col].contourf(x, y, data['omega'], levels=omega_levels, cmap=vortmap, norm=omega_norm)
        for ax in axes[:, col]:
            decorate(ax, case)
        axes[0, col].set_title(f'{chr(97+col)}  {col+1} hole'+('s' if col else '')+' | sample 00', loc='left', fontsize=11, pad=8)
    axes[0, 0].set_ylabel('Speed + streamlines', fontsize=11, labelpad=10)
    axes[1, 0].set_ylabel('Signed vorticity', fontsize=11, labelpad=10)
    fig.colorbar(top, ax=list(axes[0]), shrink=.82, pad=.012, ticks=[0, .02, .05, .1, .2, .35, .5], label='Speed (sqrt color scale)')
    fig.colorbar(bottom, ax=list(axes[1]), shrink=.82, pad=.012, ticks=[-100, -10, -1, 0, 1, 10, 100], label='Vorticity (symmetric-log color scale)')
    fig.savefig(out/'three_topology_random_preview.png', dpi=220)
    fig.savefig(out/'three_topology_random_preview.pdf')
    plt.close(fig)
    for data in payloads:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
        speed = np.hypot(data['ux'], data['uy'])
        a = axes[0].contourf(data['x'], data['y'], speed, levels=speed_levels, cmap=speedmap, norm=speed_norm)
        axes[0].streamplot(data['x'], data['y'], np.ma.masked_invalid(data['ux']), np.ma.masked_invalid(data['uy']),
                           color='#1B2730', linewidth=.5, density=1.5, arrowsize=.65, minlength=.035)
        b = axes[1].contourf(data['x'], data['y'], data['omega'], levels=omega_levels, cmap=vortmap, norm=omega_norm)
        axes[0].set_title('Speed + streamlines')
        axes[1].set_title('Signed vorticity')
        for ax in axes:
            decorate(ax, data['case'])
        fig.colorbar(a, ax=axes[0], shrink=.82, ticks=[0, .02, .05, .1, .2, .35, .5], label='sqrt color scale')
        fig.colorbar(b, ax=axes[1], shrink=.82, ticks=[-100, -10, -1, 0, 1, 10, 100], label='symmetric-log color scale')
        fig.savefig(out/f"{data['case']}_sample00.png", dpi=220)
        fig.savefig(out/f"{data['case']}_sample00.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=321)
    parser.add_argument('--out', type=Path, default=ROOT/'exports/topology_random_streamfunction_preview_20260928')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        payloads = []
        for case in CASES:
            with np.load(args.out/f'{case}_sample00.npz') as saved:
                payloads.append(dict(case=case, **{key:saved[key] for key in ('x', 'y', 'psi', 'ux', 'uy', 'omega')}))
        plot_all(payloads, args.out)
        (args.out/'display_mapping.json').write_text(json.dumps(dict(
            speed='Common 0 to 0.5, square-root color mapping',
            vorticity='Common full symmetric range, symmetric-log mapping with linthresh 0.25',
            clipping=False, underlying_fields_changed=False,
            original_linear_figure='three_topology_random_preview_linear.png'), indent=2), encoding='utf-8')
        print('Replotted stored fields; no regeneration or numerical changes.', flush=True)
        return
    tick = time.perf_counter()
    protocol = dict(kind='CPU-only initial-field preview', master_seed=20260928, sample_index=0,
                    cases=CASES, random_values='IID N(0,1) on a physical 25x25 lattice',
                    physical_x=[-1, 11], physical_y=[-6, 6], lattice_spacing=.5,
                    interpolation='Exact tensor quintic spline, C4 before boundary conditioning',
                    boundary_values='Arc-length means of the same raw random spline, no separately drawn amplitudes',
                    boundary_collar_scale=COLLAR,
                    boundary_construction='Local transfinite trace conditioning; psi=P-A*Laplacian(P), with A=grad(A)=0 and Laplacian(A)=1 on each boundary',
                    final_regularity='At least C2', derivatives='Analytic spline derivatives and order-5 Taylor algebra',
                    harmonic_solve=False, prescribed_harmonic_ratio=False, seed_selection=False,
                    normalization='One global scale per field to sampled maximum speed 0.5',
                    display_grid=args.grid, ns_or_rom_run=False, formal_60_field_evaluation_started=False)
    (args.out/'preview_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    payloads, receipts = [], []
    axis_x = np.linspace(1, 9, args.grid)
    axis_y = np.linspace(-4, 4, args.grid)
    xx, yy = np.meshgrid(axis_x, axis_y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    for case in CASES:
        start = time.perf_counter()
        field = RandomField(case)
        mask = geo.fluid_mask(points, case)
        values = field.evaluate(points[mask])
        boundary_values = field.evaluate(np.vstack(field.loops))
        peak = max(np.max(np.linalg.norm(values[:, 1:3], axis=1)), np.max(np.linalg.norm(boundary_values[:, 1:3], axis=1)))
        field.scale = .5/peak
        values *= field.scale
        full = np.full((len(points), 4), np.nan)
        full[mask] = values
        assert np.isfinite(values).all()
        check = checks(field)
        payload = dict(case=case, x=axis_x, y=axis_y)
        for k, key in enumerate(('psi', 'ux', 'uy', 'omega')):
            payload[key] = full[:, k].reshape(xx.shape)
        payloads.append(payload)
        np.savez_compressed(args.out/f'{case}_sample00.npz', **payload, raw_lattice=field.raw,
                            raw_x=field.x, raw_y=field.y, trace_constants=field.constants,
                            global_scale=field.scale, seed_hierarchy=field.seed)
        receipt = dict(case=case, sample_index=0, seed=field.seed, scale=field.scale,
                       trace_constants_scaled=[c*field.scale for c in field.constants],
                       peak_sampled_speed=float(np.max(np.linalg.norm(values[:, 1:3], axis=1))),
                       rms_speed=float(np.sqrt(np.mean(np.sum(values[:, 1:3]**2, axis=1)))),
                       vorticity_min=float(np.min(values[:, 3])), vorticity_max=float(np.max(values[:, 3])),
                       checks=check, cpu_seconds=time.perf_counter()-start)
        receipts.append(receipt)
        (args.out/f'{case}_checks.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(dict(case=case, phase='initial_field_checked', seconds=receipt['cpu_seconds'], checks_passed=True)), flush=True)
    plot_all(payloads, args.out)
    summary = dict(previews_complete=3, total_previews=3, formal_evaluations_complete=0,
                   all_initial_kinematic_checks_passed=True, records=receipts,
                   total_cpu_wall_seconds=time.perf_counter()-tick, user_visual_review='pending',
                   no_training=True, no_gpu=True, no_harmonic_energy_tuning=True)
    (args.out/'preview_complete.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, seconds=summary['total_cpu_wall_seconds'], out=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
