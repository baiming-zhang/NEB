
from _neb_paths import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

BLUE='#3C5488';RED='#E64B35'

def finish(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def filled(values,mask):
    a=np.full(mask.shape,np.nan);a[mask]=values;return np.ma.masked_invalid(a)

def field(ax,points,mask,v,axes,norm,title,stream=True):
    x=points[:,axes[0]].reshape(mask.shape);y=points[:,axes[1]].reshape(mask.shape)
    speed=np.linalg.norm(v,axis=1);c=ax.pcolormesh(x,y,filled(speed,mask),cmap='RdBu_r',norm=norm,shading='auto',rasterized=True)
    if stream:
        vx=filled(v[:,axes[0]],mask);vy=filled(v[:,axes[1]],mask)

        if np.ptp(x[0])<1e-12:x,y,vx,vy=x.T,y.T,vx.T,vy.T
        ax.streamplot(x[0],y[:,0],vx,vy,color='#333333',density=.65,linewidth=.35,arrowsize=.45)
    ax.set(xlabel='xyz'[axes[0]],ylabel='xyz'[axes[1]],title=title,aspect='equal',xlim=(0,1),ylim=(0,1))
    ax.set_title(title,fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_facecolor('#dedede');return c

def baseline(run,metrics):

    d=Path(run);z=np.load(d/'reference_K32_trajectory.npz');q=z['coefficients'];times=z['times']
    fig,ax=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    ax[0].plot(times,z['energy'],color=BLUE);ax[0].set(xlabel='Time',ylabel='Energy',title='Numerical Laplacian K32')
    for key,label,style in [('dt_halving_relative','dt halving','-'),('quadrature_relative','nested quadrature','--')]:
        ax[1].semilogy(times,np.maximum([r[key] for r in metrics['rows']],1e-16),color=BLUE,ls=style,label=label)
    ax[1].set(xlabel='Time',ylabel='Relative velocity difference',title='Full autonomous horizon');ax[1].legend(fontsize=8)
    ax[2].semilogy(range(1,33),np.maximum(metrics['normal_velocity_relative_all32'],1e-18),color=BLUE,label='Normal velocity')
    ax[2].semilogy(range(1,33),np.maximum(metrics['tangential_weak_vorticity_relative_all32'],1e-18),color=RED,label='Tangential weak vorticity')
    ax[2].set(xlabel='Mode',ylabel='Relative trace defect',title='All 32 boundary modes');ax[2].legend(fontsize=7)
    finish(fig,d/'baseline_summary')
    for plane,axes in [('xy',(0,1)),('xz',(0,2)),('yz',(1,2))]:
        z=np.load(d/f'plane_{plane}_reference.npz');frames=[0,14,29];values=np.stack([z['basis']@q[f] for f in frames])
        norm=Normalize(0,max(float(np.linalg.norm(values,axis=2).max()),1e-30));fig,ax=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
        fixed=({'xy':2,'xz':1,'yz':0})[plane];cut=float(np.mean(z['points'][:,fixed]))
        for a,f,v in zip(ax,frames,values):im=field(a,z['points'],z['fluid_mask'],v,axes,norm,f'Laplacian32 / t={f*.01:.2f}')
        fig.colorbar(im,ax=ax,label='Speed',shrink=.7)
        fig.suptitle(f'{plane.upper()} slice: {"xyz"[fixed]}={cut:.3f}; planar streamlines; code units',fontsize=11)
        finish(fig,d/f'baseline_rollout_{plane}')
        np.savez(d/f'baseline_plot_arrays_{plane}.npz',points=z['points'],fluid_mask=z['fluid_mask'],frames=frames,velocity=values)
    text='# Numerical Laplacian K32 baseline\n\n'+metrics['baseline_stage']+'\n\n'
    text+='Same original physical initial field, full-mass projection, full weak-adjoint curl-curl stiffness and rotational tensor; autonomous t=0 to0.29 with dt0.001 and0.0005. No learned training objective is used in this stage.\n\n'
    text+=f'Held-out original-field reconstruction: relative L2={metrics["initial_relative_l2"]:.8g}; component MSE={metrics["initial_mse"]:.8g}.\n\n'
    text+='This is a finite numerical reference, not an analytic or continuum-certified solution. Prior fitted reference and Piola reports retain their discretization and mapping uncertainty. Full time-step/quadrature/trace diagnostics are in reference_K32_metrics.json; no physical accuracy PASS or large learned-training permission is inferred from successful execution.\n\n'
    for name in ['baseline_summary']+[f'baseline_rollout_{p}' for p in ('xy','xz','yz')]:text+=f'![{name}]({name}.png)\n\n'
    (d/'BASELINE_REPORT.md').write_text(text,encoding='utf-8')
