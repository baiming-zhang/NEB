"""Plot completed periodic trajectories; no training or numerical evolution."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gt',type=Path,required=True)
    p.add_argument('--trajectory',action='append',required=True,help='LABEL=fields_dt0.005.npz (repeat)')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.gt) as z:gt={k:z[k] for k in ('u','v','omega','time')}
    fig,axes=plt.subplots(1,3,figsize=(11,3.3));colors=['#E64B35','#4DBBD5','#777777','#3C5488']
    den=np.sum(gt['u']**2+gt['v']**2,axis=(1,2));E0=.5*np.mean(gt['u'][0]**2+gt['v'][0]**2);Z0=.5*np.mean(gt['omega'][0]**2)
    rows={}
    for j,item in enumerate(a.trajectory):
        label,path=item.split('=',1)
        with np.load(path) as z:u,v,w,t=[z[k] for k in ('u','v','omega','time')]
        assert np.array_equal(t,gt['time']) and u.shape==gt['u'].shape
        err=np.sqrt(np.sum((u-gt['u'])**2+(v-gt['v'])**2,axis=(1,2))/den)
        E=.5*np.mean(u*u+v*v,axis=(1,2))/E0;Z=.5*np.mean(w*w,axis=(1,2))/Z0
        for axis,values in zip(axes,(100*err,E,Z)):axis.plot(t/np.pi,values,label=label,color=colors[j%len(colors)],lw=1.4)
        rows[label]=dict(relative_velocity_L2=err.tolist(),normalized_energy=E.tolist(),normalized_enstrophy=Z.tolist())
    for axis,title in zip(axes,('Relative velocity error (%)','Energy / initial GT','Enstrophy / initial GT')):
        axis.set(xlabel='Time / π',ylabel=title);axis.spines[['top','right']].set_visible(False)
    axes[0].set_yscale('symlog',linthresh=.01);axes[0].legend(frameon=False);fig.tight_layout()
    fig.savefig(a.output/'periodic_diagnostics.png',dpi=300);fig.savefig(a.output/'periodic_diagnostics.pdf');plt.close(fig)
    (a.output/'plot_data.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
