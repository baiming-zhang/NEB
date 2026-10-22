"""Execute the complete public NKB projection CLI with explicit CPU doubles.

Synthetic terminal metadata and resource callbacks are test-only. The original
NKB forward and quintic target mathematics run on NumPy; CUDA is not imported.
"""
from pathlib import Path
import importlib.util,json,sys,tempfile,threading,types
import numpy as np
ROOT=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    core=load(ROOT/'nkb/core.py','core')
    mod=load(ROOT/'_backend/flow/cases/topology3/random_streamfunctions/preview_random_streamfunction.py','projection_test_fields')
    fake=types.ModuleType('cupy');fake.__getattr__=lambda key:getattr(np,key)
    fake.asnumpy=np.asarray
    sys.modules['cupy']=fake
    # These spectral-only imports are inactive in the NKB branch.
    sys.modules['neb2d.basisnet.metaball2d.hardcoded_backend']=types.ModuleType('neb2d.basisnet.metaball2d.hardcoded_backend')
    sys.modules['harmonic_common']=types.SimpleNamespace(learned_harmonic_fields=None)
    sys.modules['train_harmonic']=types.SimpleNamespace(load_checkpoint=None)
    sys.modules['runtime']=types.SimpleNamespace(guard=lambda *args:threading.Event(),read=lambda path:json.loads(Path(path).read_text()),write=lambda path,data:Path(path).write_text(json.dumps(data)))
    with tempfile.TemporaryDirectory(prefix='neb_public_projection_cpu_') as directory:
        out=Path(directory);fields=out/'fields';fields.mkdir()
        for i in range(20):
            f=mod.RandomField('one_circle',i)
            np.savez(fields/f'sample{i:02d}.npz',raw_lattice=f.raw,seed_hierarchy=f.seed,trace_constants=f.constants,global_scale=1.)
        rng=np.random.default_rng(42);points=rng.uniform([1.1,-3.9],[8.9,3.9],(137,2));points=points[mod.geo.fluid_mask(points,'one_circle')]
        np.savez(out/'quadrature.npz',points=points,weights=np.ones(len(points))/len(points))
        model=core.Model(np,2,2,2,seed=7,width=8,hidden=7,dtype='float32');model.step=2000;model.save(out/'checkpoint.npz')
        cfg=dict(case='one_circle',K=2,input_dim=2,dim=2,seed=7,width=8,hidden_layers=7)
        (out/'checkpoint.json').write_text(json.dumps(dict(epoch=2000,config=cfg)))
        module=load(ROOT.parents[0]/'multihole/evaluate.py','neb_public_projection')
        sys.argv=['evaluate.py','--basis-family','NKB','--checkpoint',str(out/'checkpoint.npz'),'--fields',str(fields),'--quadrature',str(out/'quadrature.npz'),'--output',str(out/'result')]
        module.main()
        with np.load(out/'result/projection.npz') as z:
            assert z['relative_L2'].shape==(20,) and np.isfinite(z['relative_L2']).all()
            assert np.allclose(z['mass']@z['coefficients'],z['rhs'],rtol=1e-5,atol=1e-10)
        assert len(np.loadtxt(out/'result/per_case.csv',delimiter=',',skiprows=1))==20
    print(json.dumps(dict(public_projection_write_return=True,original_NKB_and_quintic_math=True,test_only_synthetic_terminal=True,NumPy_CuPy_API_double=True,resource_guard_double=True,actual_CUDA_imported=False,actual_training_or_fullpoint_science=False)))
if __name__=='__main__':main()
