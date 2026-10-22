"""Full 2000-update Metaball training with analytic native CuPy derivatives."""
from __future__ import annotations
import argparse, ast, gc, hashlib, importlib.util, json, math, os, shutil, sys, time
from pathlib import Path
from types import SimpleNamespace
for _thread_key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_thread_key]="1"
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE/'_vendor'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def write(path,j):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def append(p,j):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(j,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(32*2**20),b''):h.update(block)
    return h.hexdigest()
def source_binding(out,args,config):
    source={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*.py'))}
    paths=[Path(args.config)]
    for name in ('samples','source_operator','source_map','factor_options'):
        if getattr(args,name,None):paths.append(Path(getattr(args,name)))
    if args.physical:paths.extend(Path(args.physical)/name for name in ('M1.npz','S1.npz','G.npz','C.npz','complex.npz','complete.json'))
    inputs={str(p.resolve()):sha(p) for p in paths}
    protocol=dict(method=args.method,K=args.K,epochs=25 if args.timing_only else 2000,scheduled_validations=0 if args.timing_only else 80,full_checkpoint_every=100,
        config=config,canonical_class=args.canonical_class,physical_volume=args.physical_volume,
        resource_limits=dict(RSS=62_500_000_000,hostused=62_500_000_000,GPUused=15_500_000_000,hostfree_reserve=9*2**30,GPUfree_reserve=500_000_000),input_sha256=inputs)
    write(out/'protocol.json',protocol);write(out/'source_manifest.json',dict(source_sha256=source,input_sha256=inputs))
    return dict(protocol_sha256=sha(out/'protocol.json'),source_manifest_sha256=sha(out/'source_manifest.json'),input_sha256=inputs)
def resource(cp,out):
    import psutil
    free,total=cp.cuda.runtime.memGetInfo();vm=psutil.virtual_memory();pool=cp.get_default_memory_pool()
    row=dict(unix=time.time(),RSS=psutil.Process().memory_info().rss,hostfree=vm.available,hostused=vm.used,
        GPUfree=int(free),GPUused=int(total-free),pool_reserved=int(pool.total_bytes()),pool_live=int(pool.used_bytes()))
    append(out/'resources.jsonl',row)
    if row['RSS']>=62_500_000_000 or row['hostused']>=62_500_000_000 or row['hostfree']<=9*2**30 or row['GPUused']>=15_500_000_000 or row['GPUfree']<=500_000_000:
        write(out/'resource_stop.json',row);raise RuntimeError('RAM/GPU aggregate resource limit or reserve exceeded')
    outside=max(0,row['GPUused']-pool.total_bytes());pool.set_limit(size=max(1,min(12*2**30,15_500_000_000-outside)))
    return row
def native_context(config,K,out,samples):
    from prepare import nodes
    shared=load('metaball_shared',V/'exports/potential_k128_resume_e1949_20261007/shared.py')
    shared.configuration=lambda requested:(None,dict(config),{}) if requested==K else (_ for _ in ()).throw(ValueError('Explicit K mismatch'))
    shared.SOURCE=Path(samples)
    # The original loader's output assignment is routed to this exclusive run;
    # all shared geometry, Jet1/Jet2 and parameter reverse nodes stay intact.
    tree=ast.parse((V/'exports/potential_k128_resume_e1949_20261007/shared.py').read_text('utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_shared')
    for n in fn.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='run' for t in n.targets):n.value=ast.parse('public_run',mode='eval').body
    shared.public_run=out
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(shared.__file__),'exec'),shared.__dict__)
    cm,span,_=shared.load_shared('training',K,runtime_module=shared.RUNTIME)
    return shared,cm,span
def canonical_pipeline(entry,args,config,state,cp,out):
    # Use the same original full relative physical operator, global GS and
    # true N/T linear solve. Solver choice is an explicit installation input.
    implementation=V/'exports/projected_potential_loss_20261008/implementation'
    sys.path.insert(0,str(implementation))
    metric=load('metaball_metric',V/'exports/projected_potential_loss_20261008/math/projected_metric_objective.py')
    mapping=load('metaball_mapping',implementation/'source_curl_mapping.py');mapping.ASSETS=Path(args.physical)
    canonical=load('metaball_inverse',Path(args.canonical_module) if args.canonical_module else implementation/'canonical_lift.py')
    canonical.ASSETS=Path(args.physical);canonical.OLD_CERTIFICATE=Path(args.physical)/'complete.json';canonical.CERTIFICATE=Path(args.physical)/'complete.json'
    options=json.loads(Path(args.factor_options).read_text('utf-8')) if args.factor_options else {}
    protocol=dict(physical_volume=args.physical_volume,metric_backend='cupy',canonical_factor_options=options,
        canonical_lift_class=args.canonical_class,canonical_lift_module=str(canonical.__file__),
        source_curl_operator=args.source_operator,source_inverse_map=args.source_map,
        canonical_solve_options=(dict(rtol=1e-12,max_refinement=3,minimum_refinement=0,threads=1) if args.canonical_class=='TreeCanonicalLift' else dict(rtol=1e-12,maxiter=80)))
    original_loader=entry.module_from_path
    def modules(name,path):
        if 'canonical_lift' in name:return canonical
        if 'source_mapping' in name:return mapping
        if 'metric_objective' in name:return metric
        return original_loader(name,path)
    entry.module_from_path=modules
    pipeline=entry.MethodPipeline(protocol,config,state['source_points'],state['independent_points'],out/'operator',device_backend=cp)
    return pipeline
def bundle(model,epoch,K,out,config,digest,start,method,aux=None,validation_count=None):
    atomic=load('metaball_atomic',V/'exports/projected_potential_loss_20261008/support/atomic_checkpoint.py')
    return atomic.publish_bundle(out/'restart_bundles',epoch=epoch,K=K,model_writer=lambda p:model.save(p,digest),
        sampler_state=dict(next_epoch=epoch+1,seed=config['seed'],rule='seed+100000+epoch',validation_token_rule='10000000+index',validation_seed_rule='seed+9000000+index'),
        counts=dict(optimizer_steps=epoch,validations=epoch//25 if validation_count is None else validation_count,dual_updates=0),
        timing=dict(normal_training_seconds=time.perf_counter()-start,state_recovery_seconds=0.,discarded_attempt_seconds=0.,active_stage='normal_training'),
        bindings=json.loads((out/'checkpoint_bindings.json').read_text()),
        method_state=dict(method=method,source_points=103244,query_points=6144,all_modes=K,auxiliary_multipliers='not_applicable'),auxiliary_na_reason='Rayleigh-only methods have no AL multiplier/rho state.')
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--method',required=True,choices=['Potential','Velocity','Potential-AL','Potential-FG','NKB'])
    p.add_argument('--K',type=int,required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data-dir',default='data/metaball');p.add_argument('--samples');p.add_argument('--physical');p.add_argument('--source-operator');p.add_argument('--source-map')
    p.add_argument('--physical-volume',type=float,default=.7585406920021169);p.add_argument('--canonical-module');p.add_argument('--canonical-class',default='CanonicalLift');p.add_argument('--factor-options')
    p.add_argument('--finite-gauge-setting',choices=['matched','scalar-gauge'],default='matched')
    p.add_argument('--timing-only',action='store_true',help='Potential K1024: 25 updates, original 2000-update learning-rate horizon, zero validations')
    args=p.parse_args()
    if args.K<=0:raise ValueError('All positive K modes required')
    if args.timing_only and (args.method!='Potential' or args.K!=1024):p.error('--timing-only requires Potential K1024')
    config=json.loads(Path(args.config).read_text('utf-8'));config['modes']=args.K
    if config['epochs']!=2000 or config['points_per_step']!=6144 or config['validate_every']!=25:raise ValueError('Full2000 / 6144 / 80 schedule required')
    if args.method=='Potential-FG' and args.finite_gauge_setting=='scalar-gauge' and config['seed']!=0:raise ValueError('Scalar-gauge setting requires seed 0')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ['NEB_DATA_DIR']=str(Path(args.data_dir).resolve());os.environ['CUPY_CACHE_DIR']=str(out/'cupy_cache')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    sys.path.insert(0,str(V));case=V/'ICLR2027_campaign/cases/metaball';sys.path.insert(0,str(case))
    start=time.perf_counter();write(out/'config.json',config);bindings=source_binding(out,args,config);write(out/'checkpoint_bindings.json',bindings)
    digest=bindings['protocol_sha256'];c=SimpleNamespace(**config)
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=12*2**30);resource(cp,out)
    if args.method=='NKB':
        nkb(args,config,c,cp,out,start);return
    if args.method=='Velocity':
        velpath=case/'velocity_k1024_full2000_v1';sys.path.insert(0,str(velpath))
        core=load('core',velpath/'core.py');fused=load('fused',velpath/'fused.py');fused.install()
        fast=load('fast_path',velpath/'fast_path.py');fast.install()
        model=core.Model(c,args.K,config['seed']);geo=cp.asarray(c.fixed_training_geometry,'f4')
        forward=lambda epoch:core.compute(model,c,geo,core.fluid_points(c,geo,6144,c.seed+100000+epoch))
        validate=lambda epoch:[{'index':i,'mean_Rayleigh':float(cp.mean(core.compute(model,c,g,core.fluid_points(c,g,4096,c.seed+9000000+i),False)['losses']))} for i in range(8) for g in [cp.asarray(core.geometry_from_token(c,10000000+i),'f4')]]
        def step(epoch):
            result=forward(epoch);grads=result.pop('grads');model.adam(*grads,learning_rate(c,epoch),c.grad_clip)
            return float(cp.mean(result['losses'])),result['timings']
        pipeline=None
    else:
        if not args.samples:raise ValueError('Complete103244 source samples required')
        shared,cm,span=native_context(config,args.K,out,Path(args.samples));model=cm.train.Model(c,args.K,config['seed'])
        if args.method=='Potential-AL':
            augmented_lagrangian(args,config,cm,span,cp,out,start,digest);return
        entry=load('canonical_entry',V/'exports/projected_potential_loss_20261008/support/train_quotient_entry.py');entry.K=args.K
        cm.guard=lambda *a:resource(cp,out)
        geo=cp.asarray(c.fixed_training_geometry,'f8');source=np.load(args.samples,allow_pickle=False)['fit']
        if source.shape!=(103244,3):raise ValueError('Complete source point count/order required')
        ind=cm.rt.fluid_points(c,geo,16384,919000)
        state=dict(c=c,model=model,geometry=geo,sampling_geometry=geo.astype('f4'),source_points=source,independent_points=ind)
        if args.method=='Potential':
            if not all((args.physical,args.source_operator,args.source_map)):raise ValueError('Full physical complex and source operators required')
            pipeline=canonical_pipeline(entry,args,config,state,cp,out)
            def step(epoch):
                result,grads,phases=entry.full_forward_reverse(cm,pipeline,state,epoch)
                model.adam(*grads,learning_rate(c,epoch),c.grad_clip)
                return float(result['objective']['loss_mean']),phases
            validate=lambda epoch:entry.scheduled_validation(cm,pipeline,state,epoch)
        elif args.finite_gauge_setting=='matched':
            from finite_gauge import NativeFiniteGaugePipeline,full_epoch
            spaces=[cls.__dict__['__init__'].__globals__ for cls in cm.train.Pipeline.__mro__ if '__init__' in cls.__dict__ and 'Small' in getattr(cls.__dict__['__init__'],'__globals__',{})]
            if len(spaces)!=1:raise RuntimeError('Finite-gauge scalar factory closure is ambiguous')
            pipeline=NativeFiniteGaugePipeline(cm,c,geo,source,spaces[0]['Small'],spaces[0]['geo1'],lambda:resource(cp,out))
            def step(epoch):
                tick=time.perf_counter();result,_,_=full_epoch(pipeline,model,geo.astype('f4'),epoch)
                return float(cp.mean(result['obj']['losses'])),dict(finite_gauge_epoch_seconds=time.perf_counter()-tick)
            def validate(epoch):
                # Preserve eight shared geometry diagnostics and the fixed
                # finite-gauge objective on all 16384 independent points.
                original=V/'exports/projected_potential_loss_20261008/support/train_quotient_entry.py'
                tree=ast.parse(original.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scheduled_validation')
                end=next(i for i,n in enumerate(node.body) if isinstance(n,ast.FunctionDef) and n.name=='fixed')
                node.body=node.body[:end]+ast.parse("result=pipeline.compute(model,state['independent_points'],False)\nreturn dict(epoch=epoch,shared_geometry_diagnostics=rows,fixed_finite_gauge_mean=float(cm.cp.mean(result['obj']['losses'])),independent_points=16384)").body
                ns=dict(np=np,time=time,K=args.K,array_sha=lambda a:hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest())
                exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(original),'exec'),ns)
                return ns['scheduled_validation'](cm,pipeline,state,epoch)
        else:
            # Mirror the explicit source input expected by the original finite
            # scalar factory. This local copy contains samples, never a basis.
            sample_target=Path(args.data_dir)/'ICLR2027_campaign/cases/metaball/results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz'
            sample_target.parent.mkdir(parents=True,exist_ok=True)
            if Path(args.samples).resolve()!=sample_target.resolve():shutil.copyfile(args.samples,sample_target)
            pipeline=cm.train.Pipeline(c,geo)
            if pipeline.scalar.n!=441:raise ValueError('Original finite scalar441 space required')
            def step(epoch):
                result=pipeline.compute(model,cm.rt.fluid_points(c,geo,6144,908000+epoch),True)
                cm.train.adam_batched(model,*result['grads'],learning_rate(c,epoch))
                return float(cp.mean(result['obj']['losses'])),result['timings']
            validate=lambda epoch:dict(epoch=epoch,fixed_FG_mean=float(cp.mean(pipeline.compute(model,cm.rt.fluid_points(c,geo,4096,919000),False)['obj']['losses'])),source_points=103244,scalar_directions=441)
    model.save(out/'checkpoint_e0000.npz',digest)
    try:
        endpoint=25 if args.timing_only else 2000
        for epoch in range(1,endpoint+1):
            resource(cp,out);tick=time.perf_counter();loss,phases=step(epoch)
            if model.epoch!=epoch:raise RuntimeError('Adam epoch counter mismatch')
            if epoch%25==0 and not args.timing_only:write(out/f'validation_{epoch:04d}.json',validate(epoch))
            model.save(out/'checkpoint_latest.npz',digest)
            validation_count=0 if args.timing_only else epoch//25
            if epoch%100==0 or epoch==endpoint:bundle(model,epoch,args.K,out,config,digest,start,args.method,validation_count=validation_count)
            row=dict(epoch=epoch,validations=validation_count,loss=loss,phases=phases,actual_epoch_seconds=time.perf_counter()-tick,normal_training_seconds=time.perf_counter()-start)
            append(out/'steps.jsonl',row);write(out/'progress.json',row);cp.cuda.Stream.null.synchronize();gc.collect();cp.get_default_memory_pool().free_all_blocks()
        write(out/'training_complete.json',dict(method=args.method,K=args.K,optimizer_updates=endpoint,scheduled_validations=0 if args.timing_only else 80,normal_training_seconds=time.perf_counter()-start,state_recovery_seconds=0.))
    finally:
        if pipeline is not None and hasattr(pipeline,'close'):pipeline.close()
def learning_rate(c,epoch):return c.min_learning_rate+(c.learning_rate-c.min_learning_rate)*.5*(1+math.cos(math.pi*(epoch-1)/1999))

def augmented_lagrangian(args,config,cm,span,cp,out,start,digest):
    math_source=load('matched_coulomb_training',V/'matched_coulomb_training.py')
    state=math_source.make_state(cm);model=state['model'];model.save(out/'checkpoint_e0000.npz',digest)
    atomic=load('metaball_atomic',V/'exports/projected_potential_loss_20261008/support/atomic_checkpoint.py')
    for epoch in range(1,2001):
        resource(cp,out);row=math_source.full_epoch(cm,span,state,epoch)
        if epoch%10==0:append(out/'dual_updates.jsonl',math_source.full_dual_update(cm,span,state,epoch))
        if epoch%25==0:write(out/f'validation_{epoch:04d}.json',math_source.full_validation(cm,state,epoch))
        model.save(out/'checkpoint_latest.npz',digest)
        if epoch%100==0:
            values=dict(lambda_source=cp.asnumpy(state['lam']),eta=cp.asnumpy(state['eta']),rho=np.asarray(state['rho']),previous_residual=np.asarray(state['previous_residual']),scale=np.asarray(state['scale']),epoch=np.asarray(epoch),config_hash=np.asarray(digest))
            schema={k:dict(shape=list(v.shape),dtype=v.dtype.str) for k,v in values.items()}
            atomic.publish_bundle(out/'restart_bundles',epoch=epoch,K=args.K,model_writer=lambda p:model.save(p,digest),
                auxiliary_writer=lambda p:np.savez(p,**values),auxiliary_expected_arrays=schema,
                sampler_state=dict(seed=config['seed'],next_epoch=epoch+1,rule='seed+100000+epoch',validation_seed_rule='seed+9000000+index',validation_token_rule='10000000+index'),
                counts=dict(optimizer_steps=epoch,validations=epoch//25,dual_updates=epoch//10),
                timing=dict(normal_training_seconds=time.perf_counter()-start,state_recovery_seconds=0.,discarded_attempt_seconds=0.),
                bindings=json.loads((out/'checkpoint_bindings.json').read_text()),
                method_state=dict(method='Potential-AL',source_points=103244,query_points=6144,all_modes=args.K,dual_every=10,rho_cap=8.))
        row.update(normal_training_seconds=time.perf_counter()-start,validations=epoch//25,dual_updates=epoch//10)
        append(out/'steps.jsonl',row);write(out/'progress.json',row)
    write(out/'training_complete.json',dict(method='Potential-AL',K=args.K,optimizer_updates=2000,scheduled_validations=80,dual_updates=200,normal_training_seconds=time.perf_counter()-start,state_recovery_seconds=0.))
def nkb(args,config,c,cp,out,start):
    d=V/'ICLR2027_campaign/nkb_baseline_20260928';sys.path.insert(0,str(d))
    core=load('core',d/'core.py');domains=load('domains',d/'domains.py')
    domain=domains.Domain.__new__(domains.Domain);domain.name='metaball';domain.xp=cp;domain.periodic=False;domain.dim=3
    domain.base_config=config;domain.geo=np.asarray(c.fixed_training_geometry);domain.descriptor=domain.geo.ravel();domain.lo=np.zeros(3);domain.hi=np.ones(3);domain.bandwidth=.05
    _,features,_,_=domain.sample(2,c.seed);model=core.Model(cp,features.shape[1],3,args.K,c.seed)
    model.save(out/'checkpoint_e0000.npz')
    for epoch in range(1,2001):
        resource(cp,out);_,f,j,data=domain.sample(6144,c.seed+100000+epoch)
        v,jac,tape=model.forward(f,j,True);raw,gv,gj=core.losses(v,jac,*data,cp)
        grads=model.backward(tape,gv,gj);model.adam(grads,.0005*.96**(10*(epoch-1)/2000));loss=float(raw['loss'])
        del v,jac,tape,gv,gj,grads,raw,f,j,data
        if epoch%25==0:
            rows=[]
            for i,geo in enumerate(domain.validation_geometries()):
                _,f,j,data=domain.sample(4096,c.seed+9000000+i,geo);v,jac,_=model.forward(f,j)
                raw,_,_=core.losses(v,jac,*data,cp,False);rows.append({k:float(v) for k,v in raw.items()})
            write(out/f'validation_{epoch:04d}.json',dict(epoch=epoch,details=rows))
        model.save(out/'checkpoint_latest.npz')
        if epoch%100==0:
            b=out/'restart_bundles'/f'e{epoch:04d}';b.mkdir(parents=True,exist_ok=False);model.save(b/'model.npz')
            with np.load(b/'model.npz',allow_pickle=False) as z:
                if int(z['step'])!=epoch or len(z.files)!=49 or any(not np.isfinite(z[k]).all() for k in z.files):raise ValueError('Complete48 Adam/model NKB checkpoint required')
            with (b/'model.npz').open('rb') as f:os.fsync(f.fileno())
            manifest=dict(epoch=epoch,K=args.K,optimizer_steps=epoch,validations=epoch//25,next_epoch=epoch+1,sampler_seed=c.seed,sampler_rule='seed+100000+epoch',auxiliary='not_applicable',normal_training_seconds=time.perf_counter()-start,state_recovery_seconds=0.,config_sha256=sha(out/'config.json'),bindings=json.loads((out/'checkpoint_bindings.json').read_text()),model_sha256=sha(b/'model.npz'))
            write(b/'complete_manifest.json',manifest);write(out/'restart_bundles/CURRENT.json',dict(bundle_directory=str(b),complete_manifest=str(b/'complete_manifest.json')))
        row=dict(epoch=epoch,validations=epoch//25,loss=loss,normal_training_seconds=time.perf_counter()-start)
        append(out/'steps.jsonl',row);write(out/'progress.json',row);gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(out/'training_complete.json',dict(method='NKB',K=args.K,optimizer_updates=2000,scheduled_validations=80,normal_training_seconds=time.perf_counter()-start,state_recovery_seconds=0.))
if __name__=='__main__':main()
