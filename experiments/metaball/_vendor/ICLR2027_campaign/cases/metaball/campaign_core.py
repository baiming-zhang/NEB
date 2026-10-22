
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '4')
import sys, json, hashlib, time
from _neb_paths import Path
import numpy as np
from scipy.ndimage import label
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(WORKSPACE/'GAUGE_CANONICAL_EIGENFLUID'))
from gauge_core import Complex, legacy, CONFIG, json_write

def cfg():
    return legacy().load_config(CONFIG, {})

def mask_for(n, geometry):
    xyz = (np.indices((n,n,n)).reshape(3,-1).T+.5)/n
    return legacy().fluid_mask(cfg(),xyz,np.asarray(geometry)).reshape(n,n,n)

def topology(mask):

    n = mask.shape[0]
    vertices=np.zeros((n+1,)*3,bool)
    edges=[np.zeros(tuple(n if j==a else n+1 for j in range(3)),bool) for a in range(3)]
    faces=[np.zeros(tuple(n+1 if j==a else n for j in range(3)),bool) for a in range(3)]
    for i in (0,1):
        for j in (0,1):
            for k in (0,1): vertices[i:i+n,j:j+n,k:k+n] |= mask
    for a in range(3):
        other=[j for j in range(3) if j!=a]
        for i in (0,1):
            for j in (0,1):
                sl=[slice(0,n)]*3
                sl[other[0]]=slice(i,i+n); sl[other[1]]=slice(j,j+n)
                edges[a][tuple(sl)] |= mask
        for i in (0,1):
            sl=[slice(0,n)]*3; sl[a]=slice(i,i+n)
            faces[a][tuple(sl)] |= mask
    counts=[int(vertices.sum()),sum(int(x.sum()) for x in edges),sum(int(x.sum()) for x in faces),int(mask.sum())]


    b0=int(label(mask,structure=np.ones((3,3,3)))[1])
    b2=int(label(~np.pad(mask,1))[1]-1)
    chi=counts[0]-counts[1]+counts[2]-counts[3]
    return dict(b0=b0,b1=b0+b2-chi,b2=b2,euler=chi,cell_counts=counts,
                method='absolute closed cubical Euler characteristic and complement components (Alexander duality)')

def hashes():
    names=['potential_eigenfluids_experiment_pack/potential_eigenfluids_experiment_manual.md',
           'potential_eigenfluids_experiment_pack/potential_eigenfluids_experiment_manifest.yaml',
           'ICLR2027_Potential_Eigenfluids/ICLR2027_Potential_Eigenfluids.tex',
           'GAUGE_CANONICAL_EIGENFLUID/gauge_core.py',
           '3d_metaball_reproduction_latest/cases/3D_Metaball/scripts/train_validate.py']
    return {p:hashlib.sha256((WORKSPACE/p).read_bytes()).hexdigest() for p in names}

def gauge_tests(c):
    rng=np.random.default_rng(177)
    a=c.project(rng.normal(size=len(c.edges))); a/=np.linalg.norm(a)
    chi=np.prod(np.sin(np.pi*c.vertices/c.n),axis=1)
    grad=c.G@chi; grad/=np.linalg.norm(grad)
    base=c.C@a; rq=float(base@base)
    records=[]
    for kind,v in [('gradient',grad)]+[('relative_harmonic',h) for h in c.H.T]:
        for alpha in (0,1,2,4,8,1000):
            raw=a+alpha*v; p=c.project(raw); co=c.project(raw,harmonic=False)
            curl=c.C@raw
            records.append(dict(kind=kind,alpha=alpha,
                curl_relative_change=float(np.linalg.norm(curl-base)/np.linalg.norm(base)),
                raw_rayleigh=float(curl@curl/(raw@raw)),
                coulomb_rayleigh=float(np.linalg.norm(c.C@co)**2/(co@co)),
                canonical_rayleigh=float(np.linalg.norm(c.C@p)**2/(p@p)),
                canonical_relative_change=float(np.linalg.norm(p-a))))
    random=rng.normal(size=(len(c.edges),3)); p=c.project(random)
    return dict(records=records,
        cg_max=float(np.max(abs((c.C@c.G).data),initial=0)),
        dc_max=float(np.max(abs((c.D@c.C).data),initial=0)),
        harmonic_curl_norm=float(np.linalg.norm(c.C@c.H)),
        canonical_divergence=float(np.linalg.norm(c.G.T@p)/np.linalg.norm(p)),
        harmonic_leakage=float(np.linalg.norm(c.H.T@p)/np.linalg.norm(p)),
        curl_change=float(np.linalg.norm(c.C@(p-random))/np.linalg.norm(c.C@random)),
        max_perturbation_change=max(r['canonical_relative_change'] for r in records),
        max_rayleigh_relative_change=max(abs(r['canonical_rayleigh']/rq-1) for r in records),
        wall_normal_residual=None,tangential_vorticity_trace_residual=None,
        prescribed_trace_values={'wall_normal_velocity':0.,'wall_tangential_vorticity':0.},
        wall_normal_evidence='Structural boundary elimination; no independent measured smooth-wall residual')

def state(phase,progress,**kwargs):
    record=dict(updated_unix=time.time(),phase=phase,process=os.getpid(),command=sys.argv,
                progress=progress,errors=[],next=None,eta=None,**kwargs)
    json_write(ROOT/'status.json',record)
    (ROOT/'STATUS.md').write_text('# Metaball campaign\n\n'+ '\n'.join(f'{k}: {v}\n' for k,v in record.items()),encoding='utf-8')
