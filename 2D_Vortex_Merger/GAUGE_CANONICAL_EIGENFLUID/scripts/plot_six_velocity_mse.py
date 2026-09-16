"""All six velocity-MSE trajectories against frame number in one axes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/comparison_64_1024'
data = json.loads((OUT / 'completion.json').read_text(encoding='utf-8'))
assert data['complete']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(11.5, 7.5), layout='constrained')
methods = [('canonical', 'Gauge-canonical Eigenfluid', '#3C5488', 'o'),
           ('raw', 'Eigenfluid without gauge', '#E64B35', 's'),
           ('laplacian', 'Analytic Laplacian', '#00A087', '^')]
frames = np.arange(1, 62)
for k in (64, 1024):
    for method, label, color, marker in methods:
        values = np.asarray(data['metrics'][str(k)][method]['velocity_mse'])
        assert values.shape == (61,) and np.isfinite(values).all() and (values > 0).all()
        ax.plot(frames, values, color=color, linestyle='-' if k == 64 else '--',
                linewidth=2.0, marker=marker, markersize=4.2, markevery=5,
                markerfacecolor='white', markeredgewidth=1.1,
                label=f'{label} · {k} modes')
ax.set_yscale('log')
ax.set(xlim=(0, 62), ylim=(3e-20, 3e-4), xlabel='Frame number',
       ylabel='Velocity MSE relative to GT',
       title='Vortex merger: velocity error over 61 frames')
ax.set_xticks([1, 10, 20, 30, 40, 50, 61])
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax.yaxis.set_minor_locator(NullLocator())
ax.grid(axis='both', which='major', color='#DFE3E8', linewidth=.7)
ax.set_axisbelow(True)
ax.legend(loc='lower right', ncol=2, fontsize=9.5, frameon=True,
          facecolor='white', edgecolor='#DFE3E8', framealpha=.98,
          handlelength=3.1, columnspacing=1.5, borderpad=1.1)
initial = data['metrics']['1024']['laplacian']['velocity_mse'][0]
ax.annotate(f'1024-mode Laplacian, frame 1: {initial:.2e}',
            xy=(1, initial), xytext=(9, 2e-17), fontsize=10, color='#00775F',
            arrowprops=dict(arrowstyle='->', color='#00775F', linewidth=1.1))
fig.get_layout_engine().set(rect=(0, .045, 1, .955))
fig.text(.5, .016, 'Frame 1 = t = 0 projection; frames 2–61 = autonomous rollout.  Solid: 64 modes; dashed: 1024 modes.',
         ha='center', fontsize=10, color='#444444')
stem = OUT / 'velocity_mse_six_methods_by_frame'
fig.savefig(stem.with_suffix('.png'), dpi=220, facecolor='white')
fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
plt.close(fig)
print(stem.with_suffix('.png'))
print(stem.with_suffix('.pdf'))
