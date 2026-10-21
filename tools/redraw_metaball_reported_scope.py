"""Redraw the reported Metaball comparisons from their supplied measurements."""
from pathlib import Path
import os,csv,json,argparse
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
ROOT=Path(__file__).resolve().parents[1]
COLORS={'Laplacian':'#858b98','Velocity':'#526e9f','Potential':'#ee6a5f','NKB':'#9aa6c4'}
def save(fig,stem):
    fig.savefig(stem.with_suffix('.png'),dpi=360)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':1.1})
    with (ROOT/'data/paper_results/mainfour_NEW20_460_measurements.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');capacities=[32,64,128,256,512];methods=['Laplacian','Velocity','Potential','NKB'];handles=[];stats=[]
    for j,method in enumerate(methods):
        color=COLORS[method];handles.append(Rectangle((0,0),1,1,color=color,alpha=.8))
        for i,K in enumerate(capacities):
            selected=sorted((r for r in rows if r['display_method']==method and int(r['K'])==K),key=lambda r:int(r['case']));assert [int(r['case']) for r in selected]==list(range(20))
            v=100*np.asarray([float(r['relative_L2']) for r in selected]);mean=v.mean();sd=v.std(ddof=1);x=i+(j-1.5)*.16
            ax.vlines(x,v.min(),v.max(),color=color,lw=1.3);ax.hlines([v.min(),v.max()],x-.055,x+.055,color=color,lw=1.3)
            ax.add_patch(Rectangle((x-.065,mean-sd),.13,2*sd,color=color,alpha=.65,lw=0));ax.hlines(mean,x-.08,x+.08,color=color,lw=2)
            stats.append(dict(method=method,K=K,n=20,mean=float(mean),sample_SD=float(sd),min=float(v.min()),max=float(v.max())))
    ax.set(xticks=range(5),xticklabels=capacities,xlim=(-.6,4.6),ylim=(0,106),xlabel='Number of modes, $K$',ylabel=r'Relative velocity $L^2$ error (%)');ax.grid(axis='y',color='#dfe3e8',lw=.7);ax.set_axisbelow(True)
    ax.legend(handles,methods,loc='upper center',bbox_to_anchor=(.5,1.16),ncol=4,frameon=False,columnspacing=1.2,handlelength=1.5)
    save(fig,a.output/'case3_large_eddy20_method_names_20261021_v1')
    data=json.loads((ROOT/'data/paper/initial_capacity_training.json').read_text(encoding='utf-8'));assert max(r['K'] for r in data['initial_errors'])==1024 and max(r['K'] for r in data['training'])==1024
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),layout='constrained')
    for method,label,color in [('velocity','Velocity','#cc514c'),('laplacian','Laplacian','#3979af')]:
        vals=[r for r in data['initial_errors'] if method in r];axes[0].plot([r['K'] for r in vals],[100*r[method] for r in vals],'-o',color=color,lw=1.8,ms=4,label=label)
    ticks=[r['K'] for r in data['initial_errors']];axes[0].set(xscale='log',xlabel='Number of modes, $K$',ylabel='Initial velocity error (%)');axes[0].set_xticks(ticks,labels=ticks);axes[0].xaxis.set_minor_locator(NullLocator());axes[0].legend(frameon=False,fontsize=9)
    timing=data['training'];total=np.array([r['inclusive_training_seconds']/3600 for r in timing]);forward=np.array([r['mean_stage_seconds']['neural_forward_analytic_derivatives']*2000/3600 for r in timing]);back=np.array([r['mean_stage_seconds']['neural_recompute_and_analytic_backward']*2000/3600 for r in timing]);bottom=np.zeros(len(timing))
    for v,color,label in [(forward,'#3979af','Forward'),(back,'#cc514c','Backward'),(total-forward-back,'#cdd4db','Other / validation')]:
        assert np.all(v>=0);axes[1].bar(range(len(timing)),v,bottom=bottom,color=color,label=label,width=.65);bottom+=v
    axes[1].set(xticks=range(len(timing)),xticklabels=[r['K'] for r in timing],xlabel='Number of modes, $K$',ylabel='Velocity training time (h)');axes[1].legend(frameon=False,fontsize=8)
    for i,ax in enumerate(axes):ax.text(-.13,1.02,'ab'[i],transform=ax.transAxes,fontweight='bold');ax.grid(axis='y',color='#e2e6eb',lw=.6);ax.set_axisbelow(True)
    save(fig,a.output/'case3_initial_capacity_training')
    (a.output/'figure_data_summary.json').write_text(json.dumps(dict(main_figure_groups=20,main_figure_measurements=400,main_figure_max_K=512,initial_capacity_max_K=1024,statistics=stats),indent=2),encoding='utf-8')
if __name__=='__main__':main()
