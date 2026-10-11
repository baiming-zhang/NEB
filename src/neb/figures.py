"""Redraw compact paper comparisons from the accompanying measurements."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

PALETTE = {"LEB": "#2878B5", "SF-R": "#C82423", "VRQ": "#F28E2B",
           "CCQ": "#C82423", "NKB": "#7A6C99"}
NAMES = {"Laplacian": "LEB", "Potential": "SF-R", "Velocity": "VRQ", "Vel": "VRQ"}


def paired_statistics(path):
    groups = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (NAMES.get(row["method"], row["method"]), int(row["K"]))
            case = int(row["case"])
            if case in groups[key]:
                raise ValueError(f"Duplicate case {case} for {key}")
            value = float(row["relative_L2"])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid relative L2 value for {key}")
            groups[key][case] = value
    result = {}
    for key, cases in groups.items():
        if set(cases) != set(range(20)):
            raise ValueError(f"Expected the same 20 fields for {key}")
        values = np.asarray([cases[i] for i in range(20)], dtype=np.float64)
        result[key] = (float(values.mean()), float(values.std(ddof=1)))
    return result


def reproduce(root, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import NullLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    stats = paired_statistics(root / "data/paper/metaball20.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.25), constrained_layout=True)
    for method in ["LEB", "VRQ", "CCQ", "NKB"]:
        capacities = sorted(k for m, k in stats if m == method)
        ax.errorbar(capacities, [stats[method, k][0] for k in capacities],
                    yerr=[stats[method, k][1] for k in capacities],
                    color=PALETTE[method], marker="o", markersize=4, capsize=3,
                    linewidth=1.6, label=method)
    ax.set(xlabel="Basis capacity K", ylabel="Initial relative L2 error", xscale="log")
    ax.set_xticks([32, 64, 128, 256], labels=[32, 64, 128, 256])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False, ncol=2)
    _save(fig, output / "metaball20")
    plt.close(fig)

    data = json.loads((root / "data/paper/metaball_capacity.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for original, method in [("Numerical", "LEB"), ("Eigenfluid velocity", "VRQ"),
                              ("NKB adaptation", "NKB")]:
        rows = sorted((r for r in data["rows"] if r["method"] == original), key=lambda r: r["K"])
        for ax, field in zip(axes, ["initial_error", "final_error"]):
            ax.plot([r["K"] for r in rows], [r[field] for r in rows], marker="o",
                    color=PALETTE[method], label=method, linewidth=1.6, markersize=4)
            ax.set(xlabel="Basis capacity K", ylabel="Relative velocity error (%)", xscale="log")
            ax.set_xticks([128, 256, 512, 1024], labels=[128, 256, 512, 1024])
            ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_title("Initial representation")
    axes[1].set_title("Autonomous evolution")
    axes[0].legend(frameon=False)
    _save(fig, output / "metaball_capacity")
    plt.close(fig)
    print(f"PNG and vector PDF figures written to {output.resolve()}")
    reproduce_measurements(root, output)


def _family(name):
    name = name.lower()
    if 'laplacian' in name:
        return 'LEB'
    if 'potential' in name:
        return 'SF-R'
    if 'velocity' in name:
        return 'VRQ'
    if name == 'nkb':
        return 'NKB'
    raise ValueError(f'Unknown method {name}')


def reproduce_measurements(root, output, only=None):
    """Plot the supplied measured scalars and curves with their native statistics."""
    from matplotlib import pyplot as plt
    data = Path(root) / 'data/paper'
    def read(name):
        return json.loads((data / name).read_text(encoding='utf-8'))
    def finish(fig, name):
        if only is None or name in only:
            _save(fig, Path(output) / name)
        plt.close(fig)
    retained = read('retained_k1024.json')
    cases = ['one_circle', 'circle_square', 'circle_square_triangle']
    for field, name, ylabel in [
        ('captured_energy_fraction', 'random_k1024_initial_energy_capture', 'Initial captured energy (%)'),
        ('final', 'random_k1024_final_velocity_errors', 'Final relative velocity error (%)')]:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
        for ax, case in zip(axes, cases):
            rows = [r for r in retained['records'] if r['case'] == case]
            labels = list(dict.fromkeys(r['label'] for r in rows))
            for i, label in enumerate(labels):
                pair = sorted([r for r in rows if r['label'] == label], key=lambda r: r['sample'])
                if [r['sample'] for r in pair] != [0, 1]:
                    raise ValueError('Two original paired fields required for each method')
                values = 100*np.asarray([r[field] if field == 'captured_energy_fraction' else r['matched'][field] for r in pair])
                family = _family(label.split()[0])
                ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt='o', capsize=3, color=PALETTE[family])
            display=[s.replace('Potential','SF-R').replace('Velocity','VRQ').replace('Laplacian','LEB').replace(' (scaled H)','\nscaled H').replace(' P+H','\nP+H').replace(' P','\nP') for s in labels]
            ax.set(xticks=range(len(labels)), xticklabels=display,
                   title=case.replace('_',' '), ylabel=ylabel)
            ax.tick_params(axis='x', labelrotation=0, labelsize=7)
        finish(fig, name)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for col, case in enumerate(cases):
        for sample in (0, 1):
            ax = axes[sample, col]
            for r in retained['harmonic']:
                if (r['case'], r['sample']) != (case, sample):
                    continue
                family = _family(r['method'])
                label = family + ' P+H' + (' scaled' if 'scaledH' in r['method'] else '')
                values = np.asarray(r['H_subspace_projection_fraction'])
                if len(values) != len(r['times']) or not np.isfinite(values).all():
                    raise ValueError('Complete finite harmonic curves required')
                ax.plot(r['times'], 100*values, color=PALETTE[family], linestyle=':' if 'scaledH' in r['method'] else '-', label=label)
            ax.set(title=f'{case.replace("_", " ")} / field {sample}', xlabel='Time', ylabel='Energy in own harmonic span (%)')
            ax.legend(frameon=False, fontsize=7)
    finish(fig, 'random_k1024_trajectory_harmonic_fraction')
    periodic = read('periodic_merger.json')
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    for method, records in periodic['records'].items():
        curves = np.asarray([r['metrics']['relative_velocity_L2'] for r in records])
        if curves.ndim != 2 or curves.shape[1] != periodic['protocol']['frames'] or not np.isfinite(curves).all():
            raise ValueError('Complete finite periodic seed curves required')
        mean = 100*curves.mean(0); sd = 100*curves.std(0, ddof=1)
        times = np.linspace(0, periodic['protocol']['final_time'], curves.shape[1]);family = _family(method)
        ax.plot(times, mean, color=PALETTE[family], label=family)
        ax.fill_between(times, np.maximum(mean-sd, 1e-12), mean+sd, color=PALETTE[family], alpha=.15)
    ax.set(xlabel='Time', ylabel='Relative velocity error (%)', yscale='log');ax.legend(frameon=False)
    finish(fig, 'periodic_merger_seed_errors')
    nkb = read('nkb_comparison.json')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for method, values in nkb['topology_final_percent'].items():
        family = _family(method);axes[0].plot([1,2,3], values, 'o-', color=PALETTE[family], label=family)
    for method, values in nkb['metaball_final_percent'].items():
        family = _family(method);axes[1].plot(nkb['metaball_modes'], values, 'o-', color=PALETTE[family], label=family)
    axes[2].bar(range(len(nkb['periodic_labels'])), nkb['periodic_means'], yerr=nkb['periodic_sd'], capsize=3,
                color=[PALETTE[_family(s)] for s in nkb['periodic_labels']])
    axes[0].set(xlabel='Holes', ylabel='Final relative velocity error (%)');axes[1].set(xlabel='K',xscale='log')
    from matplotlib.ticker import NullLocator
    axes[1].set_xticks(nkb['metaball_modes'],labels=[str(k) for k in nkb['metaball_modes']]);axes[1].xaxis.set_minor_locator(NullLocator())
    axes[2].set(xticks=range(len(nkb['periodic_labels'])),xticklabels=[_family(s) for s in nkb['periodic_labels']],ylabel='Periodic velocity error (%)')
    axes[0].legend(frameon=False,fontsize=7)
    finish(fig, 'nkb_completed_comparison')
    gauge = read('gauge_solver.json');fig, axes = plt.subplots(1, 2, figsize=(8,4.2))
    fig.subplots_adjust(left=.09,right=.99,bottom=.34,top=.97,wspace=.3)
    colors={'fourier':PALETTE['SF-R'],'weak':PALETTE['VRQ'],'fmm':'#777777'}
    for r in gauge['rows']:
        axes[0].scatter(r['forward']+r['reverse'],100*r['eps'],color=colors[r['kind']],label=r['label'])
        axes[1].plot(np.arange(1,33),100*np.asarray(r['all32']),color=colors[r['kind']],label=r['label'])
    axes[0].set(xscale='log',yscale='log',xlabel='Forward + reverse (s)',ylabel='Maximum Poisson residual (%)')
    axes[1].set(yscale='log',xlabel='Neural column',ylabel='Poisson residual (%)')
    for ax in axes:ax.axhline(1,color='#777777',ls='--')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=6.5)
    finish(fig, 'gauge_solver_ablation')
    coefficient=read('coefficient_acceleration.json');fig,axes=plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
    for i,r in enumerate(coefficient['rows']):
        color=[PALETTE['SF-R'],PALETTE['LEB']][i];label=f'K={r["K"]}';rows=r['metrics'];times=[a['time'] for a in rows]
        axes[0].plot(times,[100*a['matched_dt_gt_relative_velocity_L2'] for a in rows],color=color,label=label)
        axes[1].plot(times,[a['mass_energy'] for a in rows],color=color,label=label)
        costs=r['stage_seconds'];axes[2].bar(np.arange(4)+(-.18 if i==0 else .18),list(costs.values()),width=.36,color=color,label=label)
    ref=coefficient['reference'];axes[1].plot([r['time'] for r in ref],[r['fluid_energy'] for r in ref],color='#777777',ls='--',label='Numerical GT')
    axes[0].set(xlabel='Time',ylabel='Velocity error (%)');axes[1].set(xlabel='Time',ylabel='Fluid kinetic energy')
    axes[2].set(yscale='log',ylabel='Wall time (s)',xticks=range(4),xticklabels=['NS online','ROM online','Tensor assembly','3 frames']);axes[2].tick_params(axis='x',labelrotation=40)
    for ax in axes:ax.legend(frameon=False,fontsize=7)
    finish(fig, 'case3_coefficient_acceleration')
    capacity = read('initial_capacity_training.json')
    fig, axes = plt.subplots(1, 2, figsize=(8,3), constrained_layout=True)
    for method, family in [('velocity','VRQ'),('laplacian','LEB')]:
        rows=[r for r in capacity['initial_errors'] if method in r]
        axes[0].plot([r['K'] for r in rows],[100*r[method] for r in rows],'o-',color=PALETTE[family],label=family)
    timing=capacity['training'];total=np.asarray([r['inclusive_training_seconds']/3600 for r in timing])
    forward=np.asarray([r['mean_stage_seconds']['neural_forward_analytic_derivatives']*2000/3600 for r in timing])
    backward=np.asarray([r['mean_stage_seconds']['neural_recompute_and_analytic_backward']*2000/3600 for r in timing])
    bottom=np.zeros(len(timing))
    for values,color,label in [(forward,PALETTE['LEB'],'Forward'),(backward,PALETTE['SF-R'],'Backward'),(total-forward-backward,'#D2D9DE','Other / validation')]:
        axes[1].bar(np.arange(len(timing)),values,bottom=bottom,color=color,label=label);bottom+=values
    axes[0].set(xscale='log',xlabel='K',ylabel='Initial velocity error (%)')
    axes[1].set(xticks=range(len(timing)),xticklabels=[r['K'] for r in timing],xlabel='K',ylabel='VRQ training time (h)')
    for ax in axes:ax.legend(frameon=False,fontsize=7)
    finish(fig,'case3_initial_capacity_training')


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
