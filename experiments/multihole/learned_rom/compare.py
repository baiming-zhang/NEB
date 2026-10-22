"""Reconstruct analytic P/PH potentials on reference nodes and full-S energy errors."""
from pathlib import Path
import argparse,json,os,sys,time

def read_reference(path):
    import numpy as np
    with np.load(path) as z:
        key="t" if "t" in z.files else "times"
        return z["xy"],z["psi"],z[key]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('operators','trajectory','reference','reference-stiffness','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sample',type=int,default=0)
    a=parser.parse_args();assert a.sample in range(20)
    import numpy as np,cupy as cp
    from scipy.sparse import load_npz
    from ..runtime import guard
    top=Path(__file__).resolve().parents[2]/'2d/_backend/flow/cases/topology3';learned=top/'learned_basis'
    os.environ['NEB_2D_VENDOR_ROOT']=str(top.parents[2])
    sys.path[:0]=[str(learned/'positive/src'),str(learned/'positive/scripts'),str(learned/'harmonic/scripts'),str(learned)]
    from neb2d.basisnet.metaball2d import hardcoded_backend as be
    from train_hardcoded import FP32_CONFIG_KEYS
    import harmonic_jet2
    from train_harmonic import load_checkpoint
    meta=json.loads((a.operators/'operators.json').read_text())
    cpmeta=json.loads(Path(meta['checkpoint']).with_suffix('.json').read_text());config=dict(cpmeta['config'])
    for key in (*FP32_CONFIG_KEYS,'mlp_compute_dtype'):config[key]='float64'
    be.F32=cp.float64;be._compute_dtype=lambda cfg:cp.float64;cfg=be.cfg_namespace(config)
    with np.load(meta['checkpoint']) as z:model=be.HardcodedModel(cfg,{key:z[key] for key in z.files})
    hmodel=None
    if meta['harmonic']:
        hm,ha=load_checkpoint(Path(meta['harmonic_checkpoint']));hc=dict(hm['config'])
        for key in (*FP32_CONFIG_KEYS,'mlp_compute_dtype'):hc[key]='float64'
        hc=be.cfg_namespace(hc);hmodel=be.HardcodedModel(hc,ha)
        xy=harmonic_jet2.xy_jets2;harmonic_jet2.xy_jets2=lambda points,dtype:xy(points,cp.float64)
    with np.load(a.trajectory) as z:t=z['t'];coeff=z['coefficients'][:,:,a.sample]
    xy,truth,rt=read_reference(a.reference)
    assert np.allclose(rt,t,rtol=0,atol=1e-12) and truth.shape==(len(t),len(xy))
    S=load_npz(a.reference_stiffness);assert S.shape==(len(xy),len(xy))
    a.output.mkdir(parents=True,exist_ok=False);stop=guard(cp,a.output);start=time.perf_counter()
    try:
        B=np.empty((len(xy),meta['K']))
        selected=meta['selected_positive_modes']
        for first in range(0,len(xy),1024):
            chunk=xy[first:first+1024];values=np.empty((len(chunk),cfg.modes))
            for mode,f,_ in model.forward_stream_jet1(cfg,chunk,0.,0.,False):values[:,int(mode.mode_index)]=cp.asnumpy(f.value[:,0])
            B[first:first+len(chunk),:len(selected)]=values[:,selected]
            if hmodel is not None:
                for j,(_,f,_) in enumerate(harmonic_jet2.learned_harmonic_fields(hmodel,hc,chunk,False)):B[first:first+len(chunk),len(selected)+j]=cp.asnumpy(f.value[:,0])
        prediction=coeff@B.T;delta=prediction-truth
        norm=np.einsum('tn,tn->t',truth,(S@truth.T).T);error=np.einsum('tn,tn->t',delta,(S@delta.T).T)
        assert np.all(norm>0) and np.all(error>=-1e-10*norm)
        relative=np.sqrt(np.maximum(error,0)/norm)
        np.savez(a.output/'reference_comparison.npz',t=t,psi=prediction,reference_psi=truth,relative_energy_error=relative)
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(4.5,3));ax.plot(t,relative,color='#E64B35');ax.set(xlabel='Time',ylabel='Relative velocity energy error');fig.tight_layout()
        fig.savefig(a.output/'relative_energy.png',dpi=300);fig.savefig(a.output/'relative_energy.pdf');plt.close(fig)
        (a.output/'comparison.json').write_text(json.dumps(dict(sample=a.sample,K=meta['K'],case=meta['case'],measure='consistent reference stiffness energy of nodal analytic streamfunction reconstruction',actual_seconds=time.perf_counter()-start),indent=2))
    finally:stop.set()
if __name__=='__main__':main()
