"""Portable mesh, analytic target, full-mass operators and native P3 reference CLI."""
from pathlib import Path
import argparse,json,time,hashlib,os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .geometry import get_obstacles,fluid_mask
from .mesh_builder import DomainMesh,build_domain_mesh
from .random_fields import RandomField,loops_for,CASES

def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def normalized_field(case,sample):
    field=RandomField(case,sample)
    x=np.linspace(1,9,321);y=np.linspace(-4,4,321);xx,yy=np.meshgrid(x,y);points=np.column_stack((xx.ravel(),yy.ravel()))
    values=field.evaluate(points[fluid_mask(points,case)])
    boundary=field.evaluate(np.vstack(field.loops))
    peak=max(np.max(np.linalg.norm(values[:,1:3],axis=1)),np.max(np.linalg.norm(boundary[:,1:3],axis=1)))
    field.scale=.5/peak
    return field

def save_field(path,field):
    np.savez_compressed(path,case=field.case,sample=field.sample,seed_hierarchy=field.seed,raw_x=field.x,raw_y=field.y,raw_lattice=field.raw,trace_constants=field.constants,global_scale=field.scale)
def load_field(path,case=None,sample=None):
    with np.load(path,allow_pickle=False) as z:
        field=RandomField.__new__(RandomField);field.seed=z['seed_hierarchy'].tolist();field.case=str(z['case']) if 'case' in z else case;field.sample=int(z['sample']) if 'sample' in z else sample;
        if field.case not in CASES or field.sample not in range(20) or field.seed != [20260928,CASES.index(field.case),field.sample]:raise ValueError('Field case/sample seed hierarchy mismatch')
        field.x=z['raw_x'];field.y=z['raw_y'];field.raw=z['raw_lattice'];field.constants=z['trace_constants'].tolist();field.scale=float(z['global_scale'])
    field.spline=RectBivariateSpline(field.x,field.y,field.raw,kx=5,ky=5,s=0);field.loops=loops_for(field.case)
    return field

def save_mesh(path,mesh):
    np.savez_compressed(path,xy=mesh.xy,tri=mesh.tri,boundary_edges=mesh.boundary_edges,boundary_edge_labels=mesh.boundary_edge_labels,case=mesh.case_name,tau=mesh.tau,max_area=mesh.max_area)
def load_mesh(path,case):
    with np.load(path,allow_pickle=False) as z:
        xy,tri,edges,labels=(z[k] for k in ('xy','tri','boundary_edges','boundary_edge_labels'))
        if 'case' in z and str(z['case'])!=case:raise ValueError('Mesh/geometry mismatch')
        tau=float(z['tau']) if 'tau' in z else 0.;area=float(z['max_area']) if 'max_area' in z else .004
    nodes=tuple(np.unique(edges[labels==j]) for j in range(len(get_obstacles(case))+1))
    return DomainMesh(xy,tri,edges,labels,nodes,case,get_obstacles(case),tau,area)

def adaptive_mesh(case,field,output,passes=7,max_triangles=750000):
    from . import mesh_refinement as original
    from .gpu_field import GPUField
    original.use_native();cp=original.cp;device_field=GPUField(field);info=original.initial_mesh(case);records=[]
    for iteration in range(passes):
        mesh=original.mesh_from(info,case);metrics,errors,areas=original.interpolation_metrics(mesh,device_field,field)
        records.append(dict(iteration=iteration,triangles=len(mesh.tri),vertices=len(mesh.xy),**metrics));save_mesh(output/f'mesh{iteration}.npz',mesh)
        if metrics['relative_velocity_error']<.007 or len(mesh.tri)>=max_triangles or iteration==passes-1:break
        order=np.argsort(errors)[::-1];count=int(np.searchsorted(np.cumsum(errors[order]),.7*errors.sum()))+1;marked=order[:count]
        if len(mesh.tri)+3*count>max_triangles:break
        info.element_volumes.setup()
        for i in range(len(mesh.tri)):info.element_volumes[i]=-1.
        for i in marked:info.element_volumes[int(i)]=float(areas[i]/4)
        info=original.triangle.refine(info,min_angle=28)
        cp.get_default_memory_pool().free_all_blocks()
    write_json(output/'mesh_refinement.json',dict(case=case,records=records,criterion='70% squared analytic interpolation error; marked area /4',selected_initial_mesh=str(output/f'mesh{iteration}.npz')))
    return mesh

def native_guard(cp,ram_limit=62500000000,gpu_limit=15500000000):
    import psutil
    def guard():
        vm=psutil.virtual_memory();free,total=cp.cuda.runtime.memGetInfo()
        if psutil.Process().memory_info().rss>=ram_limit or vm.total-vm.available>=ram_limit or vm.available<=9*2**30:raise MemoryError('RAM upper bound or host-free reserve')
        if total-free>=gpu_limit or free<=500000000:raise MemoryError('GPU upper bound or free reserve')
    return guard

def run_reference(mesh,field,output,degree=3,dt=.001,final_time=1.,snapshot_dt=.02,viscosity=1/750,timeout_seconds=7200,pool_bytes=4*2**30):
    from .high_order_reference import Space,HighOrderNS
    from .gpu_field import GPUField
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=pool_bytes);guard=native_guard(cp);guard();device_field=GPUField(field)
    space=Space(mesh,degree);fem=HighOrderNS(space,viscosity);omega,gamma,psi=fem.projected_initial(field,device_field)
    initial_error=space.analytic_error(psi,device_field,order=7)
    guard()
    frames,seconds=fem.advance(omega,gamma,dt,final_time,snapshot_dt,'reference',timeout_seconds=timeout_seconds,frame_callback=lambda *_:guard())
    np.savez_compressed(output/'trajectory.npz',t=np.asarray([v[0] for v in frames]),psi=np.stack([cp.asnumpy(v[1]) for v in frames]),omega=np.stack([cp.asnumpy(v[2]) for v in frames]),gamma=np.stack([cp.asnumpy(v[3]) for v in frames]),xy=space.xy_cpu,cells=space.cells_cpu)
    from cupyx.scipy.sparse import save_npz
    save_npz(str(output/'mass.npz'),fem.M);save_npz(str(output/'stiffness.npz'),fem.S)
    np.savez_compressed(output/'harmonic_reference.npz',H=cp.asnumpy(fem.H),period=cp.asnumpy(fem.period),harmonic_gram=cp.asnumpy(fem.harmonic_gram),free_nodes=cp.asnumpy(fem.F))
    write_json(output/'reference.json',dict(case=field.case,sample=field.sample,degree=degree,dt=dt,T=final_time,snapshot_dt=snapshot_dt,viscosity=viscosity,frames=len(frames),nodes=space.n,triangles=space.nt,initial_error=initial_error,actual_native_integration_seconds=seconds,method='original fullmass vorticity/circulation IMEX midpoint/Crank-Nicolson; native CG true-residual correction',independent_temporal_spatial_convergence_certified=False))
    return frames

def vortex_parameters(config,sample):
    if not 0<=sample<config['gt_seeds']:raise ValueError('Sample outside fixed ensemble')
    children=np.random.SeedSequence(config['master_seed']).spawn(config['gt_seeds'])
    ratios=np.random.Generator(np.random.PCG64(config['master_seed']+17)).uniform(*config['harmonic_energy_ratio_range'],size=config['gt_seeds'])
    return dict(seed=int(children[sample].generate_state(1,dtype=np.uint64)[0]),vortex_count_range=tuple(config['vortex_count_range']),harmonic_energy_ratio=float(ratios[sample]),target_peak_speed=config['initial_peak_speed'],support_radius_range=tuple(config['vortex_support_radius_range']))


def run_vortex_reference(mesh,output,seed,dt=.001,final_time=1.,snapshot_dt=.02,viscosity=1/750,timeout_seconds=7200,initial_parameters=None):
    """Original compact-Wendland vortex ensemble and harmonic energy normalization."""
    from .solver_sparse import TopologyNS,make_random_initial,pack_snapshots
    import cupy as cp
    guard=native_guard(cp);guard();model=TopologyNS(mesh,viscosity)
    parameters=initial_parameters or dict(seed=seed,vortex_count_range=(3,7),harmonic_energy_ratio=.12,target_peak_speed=.5,support_radius_range=(.42,.68))
    specification,state=make_random_initial(model,**parameters)
    write_json(output/'initial_specification.json',specification.as_dict())
    omega,gamma,psi,reconstruction_error=state
    np.savez_compressed(output/'initial_state.npz',psi=cp.asnumpy(psi),cell_velocity=cp.asnumpy(model.velocity(psi)),xy=mesh.xy,tri=mesh.tri)
    frames,seconds=model.advance(omega,gamma,dt,final_time,snapshot_dt,'vortex-reference',timeout_seconds=timeout_seconds)
    guard();np.savez_compressed(output/'trajectory.npz',**pack_snapshots(frames),xy=mesh.xy,tri=mesh.tri)
    from cupyx.scipy.sparse import save_npz
    save_npz(str(output/'mass.npz'),model.M);save_npz(str(output/'stiffness.npz'),model.S)
    write_json(output/'reference.json',dict(case=mesh.case_name,seed=seed,dt=dt,T=final_time,snapshot_dt=snapshot_dt,viscosity=viscosity,initial_reconstruction_error=reconstruction_error,actual_native_integration_seconds=seconds,degree=1,initial_ensemble='original compact Wendland C4; saved complete physical sample specification'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('mesh','field','operators','adaptive-mesh','trajectory','vortex-trajectory','controls'),required=True);parser.add_argument('--case',choices=CASES,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mesh',type=Path);parser.add_argument('--field',type=Path);parser.add_argument('--fields',type=Path);parser.add_argument('--vortex-config',type=Path);parser.add_argument('--modes',type=int,default=32);parser.add_argument('--positive-only',action='store_true');parser.add_argument('--sample',type=int,default=0);parser.add_argument('--max-area',type=float,default=.004);parser.add_argument('--outer-points',type=int,default=2048);parser.add_argument('--obstacle-points',type=int,default=1024);parser.add_argument('--degree',type=int,choices=(2,3),default=3);parser.add_argument('--dt',type=float,default=.001);parser.add_argument('--final-time',type=float,default=1.);parser.add_argument('--snapshot-dt',type=float,default=.02);parser.add_argument('--viscosity',type=float,default=1/750);parser.add_argument('--timeout-seconds',type=float,default=7200);parser.add_argument('--passes',type=int,default=7);parser.add_argument('--max-triangles',type=int,default=750000);a=parser.parse_args(argv)
    if a.sample not in range(20) or min(a.max_area,a.dt,a.final_time,a.snapshot_dt,a.timeout_seconds)<=0:parser.error('Positive parameters and fixed sample0..19 required')
    if a.mode in ('operators','trajectory','vortex-trajectory','controls') and a.mesh is None:parser.error('--mesh required')
    if a.field and a.fields:parser.error('Choose --field or --fields')
    if a.fields:a.field=a.fields/f'sample{a.sample:02d}.npz'
    if a.field and not a.field.is_file():parser.error('Field specification does not exist')
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    if a.mode=='mesh':save_mesh(a.output/'mesh.npz',build_domain_mesh(a.case,0.,a.max_area,a.outer_points,a.obstacle_points))
    elif a.mode=='field':save_field(a.output/'field.npz',normalized_field(a.case,a.sample))
    elif a.mode=='operators':
        if a.mesh is None:parser.error('--mesh required')
        from .fem_cpu import assemble_operators
        from scipy.sparse import save_npz
        space,mass,stiffness,checks=assemble_operators(load_mesh(a.mesh,a.case),a.degree)
        save_npz(a.output/'mass.npz',mass);save_npz(a.output/'stiffness.npz',stiffness);np.savez_compressed(a.output/'space.npz',xy=space.xy_cpu,cells=space.cells_cpu)
        from .fem_cpu import export_quadrature
        export_quadrature(space,a.output/'quadrature.npz')
        write_json(a.output/'operators.json',dict(degree=a.degree,nodes=space.n,triangles=space.nt,quadrature_order=5,basis_checks=checks,CPU_operator_construction=True,GT_trajectory_generated=False))
    elif a.mode=='controls':
        from .laplacian_controls import generate
        import cupy as cp
        guard=native_guard(cp);guard();model,columns,rates,diagnostics=generate(load_mesh(a.mesh,a.case),a.modes,not a.positive_only,a.viscosity);guard()
        np.savez_compressed(a.output/'controls.npz',xy=cp.asnumpy(model.xy),tri=cp.asnumpy(model.tri),potentials=cp.asnumpy(columns),eigenvalues=cp.asnumpy(rates))
        write_json(a.output/'controls.json',dict(total_modes=a.modes,harmonic_dimension=0 if a.positive_only else model.harmonic_dimension,degree=1,diagnostics=diagnostics,construction='original dense consistent-mass P1 control'))
    elif a.mode=='vortex-trajectory':
        config=json.loads(a.vortex_config.read_text(encoding='utf-8')) if a.vortex_config else json.loads((Path(__file__).parent/'configs'/f'{a.case}_random20.json').read_text(encoding='utf-8'))
        if config['subcase']!=a.case:raise ValueError('Vortex protocol/geometry mismatch')
        parameters=vortex_parameters(config,a.sample)
        run_vortex_reference(load_mesh(a.mesh,a.case),a.output,parameters['seed'],a.dt,a.final_time,a.snapshot_dt,a.viscosity,a.timeout_seconds,parameters)
    else:
        field=load_field(a.field,a.case,a.sample) if a.field else normalized_field(a.case,a.sample)
        if field.case!=a.case or field.sample!=a.sample:raise ValueError('Field case/sample mismatch')
        if a.mode=='adaptive-mesh':adaptive_mesh(a.case,field,a.output,a.passes,a.max_triangles)
        else:
            if a.mesh is None:parser.error('--mesh required')
            run_reference(load_mesh(a.mesh,a.case),field,a.output,a.degree,a.dt,a.final_time,a.snapshot_dt,a.viscosity,a.timeout_seconds)
    write_json(a.output/'run.json',dict(mode=a.mode,case=a.case,sample=a.sample,actual_wall_seconds=time.perf_counter()-started,source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}));print(json.dumps({'output':str(a.output.resolve()),'mode':a.mode}))
if __name__=='__main__':main()
