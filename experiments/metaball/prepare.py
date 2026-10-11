"""Regenerate fitted geometry, source points, independent DNS and paired targets."""
from __future__ import annotations
import argparse, ast, importlib.util, json, os, sys, time, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE/'_vendor'
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def nodes(p,names,ns):
    tree=ast.parse(p.read_text('utf-8'))
    selected=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in selected}!=set(names):raise ValueError('Scientific interface changed')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*selected],type_ignores=[])),str(p),'exec'),ns)
def config(args):return json.loads(Path(args.config).read_text('utf-8'))
def mesh(args):
    import gmsh,meshio,scipy.sparse as sp,scipy.sparse.linalg as sla
    from scipy.special import logsumexp
    from scipy.sparse.csgraph import connected_components
    from skimage.measure import marching_cubes
    from skfem import MeshTet,Basis,ElementTetN1,asm,BilinearForm
    from skfem.helpers import dot,curl
    cfg=SimpleNamespace(**config(args));out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ns=dict(np=np,gmsh=gmsh,meshio=meshio,logsumexp=logsumexp,marching_cubes=marching_cubes,MeshTet=MeshTet,cfg=lambda:cfg,
        sp=sp,sla=sla,connected_components=connected_components,Basis=Basis,ElementTetN1=ElementTetN1,asm=asm,BilinearForm=BilinearForm,dot=dot,curl=curl)
    case=V/'ICLR2027_campaign/cases/metaball'
    nodes(case/'fitted_metaball_v2.py',['mesh_metaball'],ns)
    nodes(case/'fitted_shell_probe.py',['mass','stiffness'],ns)
    nodes(case/'fitted_core_v2.py',['FittedComplex'],ns)
    start=time.perf_counter();m=ns['mesh_metaball'](cfg.fixed_training_geometry,args.h,args.surface_resolution,out)
    complex_=ns['FittedComplex'](m);values,a,residual=complex_.spectrum(args.K)
    if residual.max()>=1e-6:raise FloatingPointError('Complete generalized residual failed')
    np.savez_compressed(out/'modes.npz',p=m.p,t=m.t,free_edges=complex_.free,potential=a,eigenvalues=values)
    (out/'complete.json').write_text(json.dumps(dict(K=args.K,mesh=complex_.metadata,residuals=residual.tolist(),actual_seconds=time.perf_counter()-start),indent=2))
def source(args):
    import cupy as cp
    case=V/'ICLR2027_campaign/cases/metaball'
    c=SimpleNamespace(**config(args));geo=cp.asarray(c.fixed_training_geometry,'f8')
    ns=dict(np=np,cp=cp);nodes(case/'native_cupy_core.py',['levels','fluid_points'],ns)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    # Exact original PCG64/FP32 candidate generation, membership and ordering.
    fit=ns['fluid_points'](c,geo,103244,2026090901).astype('f8')
    np.savez(out/'samples.npz',fit=fit)
    independent=ns['fluid_points'](c,geo,16384,919000).astype('f8')
    np.save(out/'independent16384.npy',independent,allow_pickle=False)
def physical(args):
    import cupy as cp,scipy.sparse as sp,scipy.sparse.linalg as sla
    from scipy.sparse.csgraph import connected_components
    from threadpoolctl import threadpool_limits
    from skfem import MeshTet
    from train import resource
    case=V/'ICLR2027_campaign/cases/metaball';sys.path.insert(0,str(case))
    # Piola geometry imports the unchanged native analytic geometry backend.
    mapping=load('metaball_radial_geometry',case/'piola_radial_geometry.py')
    c=SimpleNamespace(**config(args));out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source_file=V/'exports/gauge_repair_k128_20261006/compatible_physical_complex.py'
    tree=ast.parse(source_file.read_text('utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # Only local input/output declarations are adapted. Assembly, complete
    # quadrature, project/transpose and all kernel tests retain original AST.
    for n in ast.walk(main):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name=='source':n.value=ast.parse('mesh_input',mode='eval').body
            if name=='cfg':n.value=ast.parse('public_config',mode='eval').body
            if name=='data_dir':n.value=ast.parse('dest',mode='eval').body
    def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False))
    def log(dest,phase,**values):
        resource(cp,out);print(json.dumps(dict(phase=phase,**values)),flush=True)
    ns=dict(np=np,cp=cp,sp=sp,sla=sla,Path=Path,time=time,MeshTet=MeshTet,RadialMap=mapping.RadialMap,threadpool_limits=threadpool_limits,
        connected_components=connected_components,PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)],LEVEL=1,
        mesh_input=Path(args.mesh),public_config=dict(geometry=c.fixed_training_geometry),config_namespace=lambda:c,
        read=lambda p:json.loads(Path(p).read_text()),write=write,progress=log,resource_check=lambda d:resource(cp,out))
    nodes(source_file,['quadrature'],ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),str(source_file),'exec'),ns)
    resource(cp,out)
    result=ns['main'](out);write(out/'complete.json',result)
def source_map(args):
    module=load('metaball_curl_source',V/'exports/projected_potential_loss_20261008/implementation/source_curl_mapping.py')
    module.ASSETS=Path(args.physical);module.SOURCE=Path(args.samples)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    # The original parser accepts the velocity configuration in this wrapper.
    p=out/'source_config.json';p.write_text(json.dumps({'run_settings':{'32':{'vel_config':config(args)}}}))
    module.CONFIG=p;module.run(SimpleNamespace(output=out,timeout=7200))

def volume(args):
    """Original all-cell q1 positive physical-volume quadrature."""
    module=load('public_physical_volume',V/'exports/projected_potential_loss_20261008/support/physical_volume.py')
    mapping_module=load('public_volume_mapping',V/'exports/projected_potential_loss_20261008/implementation/source_curl_mapping.py')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with np.load(Path(args.physical)/'complex.npz') as z:
        p,t=z['p'].copy(),z['t'].copy();bary,weights=z['points'].copy(),z['weights'].copy()
    mapping=mapping_module.RadialCPU(p,t,np.asarray(config(args)['fixed_training_geometry'],'f8'),config(args),lambda *a,**k:None)
    cells=np.zeros(len(t),'f8')
    for first in range(0,len(t),2048):
        from run import guard
        guard();stop=min(first+2048,len(t));vertices=p[t[first:stop]]
        reference=np.abs(np.linalg.det((vertices[:,1:]-vertices[:,:1]).transpose(0,2,1)))/6
        if np.any(reference<=0):raise FloatingPointError('Positive tetrahedral volumes required')
        values=np.zeros(stop-first,'f8')
        for point,weight in zip(bary,weights):
            coordinates=np.einsum('v,nva->na',point,vertices);det,_=module.forward_map(mapping,coordinates)
            values+=reference*det*weight
        cells[first:stop]=values
    if not np.isfinite(cells).all() or np.any(cells<=0):raise FloatingPointError('Positive complete physical volume required')
    np.save(out/'cell_volumes.npy',cells,allow_pickle=False)
    (out/'complete.json').write_text(json.dumps(dict(physical_volume=float(np.sum(cells,dtype=np.float64)),cells=len(t),quadrature_points=len(bary)),indent=2))
def dns(args):
    module=load('metaball_dns',V/'3d_metaball_reproduction_latest/cases/3D_Metaball/k2048_joint_ns/scripts/generate_independent_dns_gt_128.py')
    import dataclasses
    original_loader=module.load_training_module
    def training_module():
        training=original_loader();original=training.Config
        allowed={f.name for f in dataclasses.fields(original)}
        training.Config=lambda **values:original(**{k:v for k,v in values.items() if k in allowed})
        return training
    module.load_training_module=training_module
    module.CONFIG_PATH=Path(args.config);module.run_dns(Path(args.output),module.DNSConfig(substeps_per_frame=16))
    mask=np.load(Path(args.output)/'fluid_mask.npy',allow_pickle=False)
    np.save(Path(args.output)/'fluid_flat_indices.npy',np.flatnonzero(mask.reshape(-1)),allow_pickle=False)
def random20(args):
    import torch, dataclasses
    # Load only the original initializer's numerical definitions; no CUDA or
    # rendering side effects accompany this CPU target generation path.
    ns=dict(np=np,torch=torch,math=math,dataclass=dataclasses.dataclass,Any=object)
    source=V/'3d_metaball_reproduction_latest/cases/3D_Metaball/k2048_joint_ns/scripts/generate_independent_dns_gt_128.py'
    nodes(source,['DNSConfig','SpectralOperators','independent_initial_velocity','damp_normal_component','PseudoSpectralDNS'],ns)
    d=Path(args.dns);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    arrays={k:np.load(d/f,allow_pickle=False) for k,f in {'solid_fraction':'solid_fraction.npy','fluid_mask':'fluid_mask.npy','normal_xyz':'boundary_normal_xyz.npy','boundary_band':'boundary_band.npy'}.items()}
    flat=np.flatnonzero(arrays['fluid_mask'].reshape(-1))
    if len(flat)!=1599754:raise ValueError('Original complete fluid-node count required')
    torch.set_num_threads(1)
    for i,seed in enumerate(range(20260826,20260846)):
        from run import guard
        guard();c=ns['DNSConfig'](seed=seed,substeps_per_frame=16)
        solver=ns['PseudoSpectralDNS'](c,device=torch.device('cpu'),**arrays)
        state=solver.initialize();target=solver.operators.ifft_real(state).permute(1,2,3,0).reshape(-1,3).numpy()[flat].astype('f8')
        np.save(out/f'case_{i:02d}.npy',target,allow_pickle=False)
    np.save(out/'fluid_flat_indices.npy',flat,allow_pickle=False)
def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='stage',required=True)
    for name,func in [('mesh',mesh),('source',source),('physical',physical),('source-map',source_map),('volume',volume),('dns',dns),('random20',random20)]:
        a=s.add_parser(name);a.add_argument('--config',default=str(HERE/'configs/vrq_ccq_k32.json'));a.add_argument('--output',required=True);a.set_defaults(func=func)
        if name=='mesh':a.add_argument('--h',type=float,default=.025);a.add_argument('--surface-resolution',type=int,default=80);a.add_argument('--K',type=int,default=1024)
        if name=='physical':a.add_argument('--mesh',required=True)
        if name=='volume':a.add_argument('--physical',required=True)
        if name=='source-map':a.add_argument('--physical',required=True);a.add_argument('--samples',required=True)
        if name=='random20':a.add_argument('--dns',required=True)
    args=p.parse_args();sys.path.insert(0,str(V));args.func(args)
if __name__=='__main__':main()
