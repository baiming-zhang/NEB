"""Compact paired Metaball panels at final print size, from measured data."""
from pathlib import Path
import os,csv,json,argparse
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
COLORS={'Velocity':'#dd7b70','Laplacian':'#568fc4','NKB':'#5aa3a8','Potential':'#8b7b9d'}
def axes_style(ax):
    ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_axisbelow(True)
    ax.grid(axis='y',color='#e5e9ed',lw=.65)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(length=3,width=.8,pad=4,color='#3e4853',labelcolor='#26313c')
    for end in [(1.025,0),(0,1.025)]:
        ax.annotate('',xy=end,xytext=(0,0),xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',lw=.85,color='#38424d',mutation_scale=9,shrinkA=0,shrinkB=0),annotation_clip=False)
    ax.set_xlabel('Number of modes, $K$',labelpad=7)
def distribution(ax,rows):
    methods=['Laplacian','Velocity','Potential','NKB'];ks=[32,64,128,256,512];stats=[]
    for j,m in enumerate(methods):
        for i,k in enumerate(ks):
            selected=sorted((r for r in rows if r['display_method']==m and int(r['K'])==k),key=lambda r:int(r['case']))
            assert [int(r['case']) for r in selected]==list(range(20))
            v=100*np.array([float(r['relative_L2']) for r in selected]);mean=v.mean();sd=v.std(ddof=1);x=i+(j-1.5)*.17;c=COLORS[m]
            ax.vlines(x,v.min(),v.max(),color=c,lw=1)
            ax.hlines([v.min(),v.max()],x-.042,x+.042,color=c,lw=1)
            ax.add_patch(Rectangle((x-.052,mean-sd),.104,2*sd,facecolor=c,alpha=.28,edgecolor='none'))
            ax.hlines(mean,x-.067,x+.067,color=c,lw=1.9)
            stats.append(dict(method=m,K=k,n=20,mean=float(mean),sample_SD=float(sd),min=float(v.min()),max=float(v.max())))
    ax.set_xticks(range(len(ks)),labels=ks);ax.set_xlim(-.55,4.55);axes_style(ax)
    return stats
def evolution(ax,rows):
    mapping={'Velocity':'Eigenfluid velocity','Laplacian':'Numerical','NKB':'NKB adaptation'};ks=[128,256,512,1024]
    used=[]
    for j,(m,key) in enumerate(mapping.items()):
        for i,k in enumerate(ks):
            r=next(r for r in rows if r['K']==k and r['method']==key);x=i+(j-1)*.22;c=COLORS[m];a=r['initial_error'];b=r['final_error']
            ax.annotate('',xy=(x,b-1.2),xytext=(x,a+1.2),arrowprops=dict(arrowstyle='-|>',color=c,lw=1.15,mutation_scale=8,shrinkA=0,shrinkB=0))
            ax.plot(x,a,'o',mfc='white',mec=c,mew=1.35,ms=4.3,zorder=3)
            ax.plot(x,b,'o',mfc=c,mec=c,mew=.8,ms=4.3,zorder=3)
            used.append(dict(method=m,K=k,initial_error=a,final_error=b,color=c))
    ax.set_xticks(range(len(ks)),labels=ks);ax.set_xlim(-.6,3.6);axes_style(ax)
    return used
def save(fig,p):
    fig.savefig(p.with_suffix('.pdf'));fig.savefig(p.with_suffix('.png'),dpi=360);plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'text.color':'#26313c','axes.labelcolor':'#26313c','pdf.fonttype':42,'ps.fonttype':42})
    with (a.repo/'data/paper_results/mainfour_NEW20_460_measurements.csv').open(encoding='utf-8-sig',newline='') as f:random=list(csv.DictReader(f))
    capacity=json.loads((a.repo/'data/paper/metaball_capacity.json').read_text(encoding='utf-8'))['rows'];assert max(r['K'] for r in capacity)==1024
    fig,axs=plt.subplots(1,2,figsize=(6.5,3.4),sharey=True)
    fig.subplots_adjust(left=.105,right=.98,bottom=.20,top=.745,wspace=.19)
    stats=distribution(axs[0],random);values=evolution(axs[1],capacity)
    axs[0].set_ylabel('Relative velocity error (%)',labelpad=8)
    fig.text(.105,.975,'(a) 20 randomized fields',ha='left',va='top',fontsize=10,fontweight='bold',color='#26313c')
    fig.text(.58,.975,'(b) One-flow evolution',ha='left',va='top',fontsize=10,fontweight='bold',color='#26313c')
    legend=[Line2D([],[],marker='s',ls='none',ms=6,mfc=COLORS[m],mec='none',label=m) for m in ['Laplacian','Velocity','Potential','NKB']]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.31,.915),ncol=2,frameon=False,handletextpad=.35,columnspacing=1.05,borderpad=0)
    fig.legend(handles=[h for h in legend if h.get_label()!='Potential'],loc='upper center',bbox_to_anchor=(.78,.915),ncol=3,frameon=False,handletextpad=.3,columnspacing=.65,borderpad=0)
    times=[Line2D([],[],marker='o',ls='none',ms=4.5,mfc='white',mec='#566270',label='Initial'),Line2D([],[],marker='o',ls='none',ms=4.5,mfc='#566270',mec='#566270',label='Final')]
    axs[1].legend(handles=times,loc='lower right',bbox_to_anchor=(1,1.015),ncol=2,frameon=False,handletextpad=.25,columnspacing=.7,borderpad=0)
    save(fig,a.output/'case3_representation_evolution_combined')
    # Standalone exports retain the same method colors and arrowed axes.
    for name,drawer,data in [('case3_large_eddy20_method_names_20261021_v1',distribution,random),('case3_capacity_to1024',evolution,capacity)]:
        f,ax=plt.subplots(figsize=(6.5,2.7));f.subplots_adjust(left=.105,right=.98,bottom=.24,top=.80);drawer(ax,data);ax.set_ylabel('Relative velocity error (%)',labelpad=8)
        subset=['Laplacian','Velocity','Potential','NKB'] if drawer is distribution else ['Laplacian','Velocity','NKB']
        ax.legend(handles=[h for h in legend if h.get_label() in subset],loc='lower center',bbox_to_anchor=(.5,1.07),ncol=len(subset),frameon=False,handletextpad=.35,columnspacing=1.1,borderpad=0)
        save(f,a.output/name)
    (a.output/'figure_data_summary.json').write_text(json.dumps(dict(final_figure_width_inches=6.5,base_font_pt=10,method_colors=COLORS,capacity_color_gradients_removed=True,all_axes_have_arrowheads=True,main_randomfield_measurements=400,main_randomfield_max_K=512,evolution_max_K=1024,distribution_statistics=stats,evolution_values=values),indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
