




import gc
import hashlib
import json
import os
from _neb_paths import Path
import time
import numpy as np
from coefficient_rom import CoefficientROM
from streamed_fields import StreamedFields
from prepare_coefficient_fields import write

OUT=Path('_data/metaball/capacity2048_20260925')
RUN=OUT/'coefficient_k512'


def progress(phase,**values):
    row=dict(phase=phase,unix=time.time(),pid=os.getpid(),**values)
    write(RUN/'accelerated_progress.json',row)
    print(json.dumps(row),flush=True)


class SafeFields(StreamedFields):
    def blocks(self):

        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            for off in range(0,3*self.N,self.block_rows):
                count=min(self.block_rows,3*self.N-off)
                host=np.fromfile(stream,dtype='f4',count=count*self.K).reshape(count,self.K)
                block=self.xp.asarray(host,dtype='f8',blocking=True)
                yield off,block
                del block,host


def references(fields,flat,folder,tag,cp):
    target=RUN/f'reference_loads_{tag}.npz'
    if target.exists():
        with np.load(target) as z:
            return z['loads'].copy(),z['norms'].copy(),float(z['seconds'])
    start=time.perf_counter()
    ref=cp.empty((3*fields.N,30),dtype='f8')
    norms=cp.empty(30,dtype='f8')
    for frame in range(30):
        velocity=cp.asarray(np.load(folder/'velocity'/f'frame_{frame:03d}.npy'),blocking=True)
        fluid=velocity.reshape(-1,3)[flat].reshape(-1)
        ref[:,frame]=fluid
        norms[frame]=cp.sum(fluid*fluid)/128**3
        del velocity,fluid
    loads=cp.zeros((fields.K,30),dtype='f8')
    for off,b in fields.blocks():
        loads+=b.T@ref[off:off+len(b)]/128**3
    loads,norms=cp.asnumpy(loads),cp.asnumpy(norms)
    seconds=time.perf_counter()-start
    temp=target.with_suffix('.tmp.npz')
    np.savez(temp,loads=loads,norms=norms,seconds=seconds,source=str(folder))
    temp.replace(target)
    del ref
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
    progress('reference_loads_complete',reference=tag,seconds=seconds)
    return loads,norms,seconds


def ns_reference(grid,spectral,cp):

    folder=RUN/'gt_dt0.000625'
    folder.mkdir(exist_ok=True);(folder/'velocity').mkdir(exist_ok=True)
    receipt=folder/'complete.json'
    if receipt.exists():
        return folder,json.loads(receipt.read_text())
    sync=cp.cuda.get_current_stream().synchronize
    dt=.000625
    v=cp.asarray(np.load(grid.D/'original_fourier_initial.npy'),dtype='c16',blocking=True)
    sync();times=[];rows=[];output_seconds=0.

    for frame in range(30):
        sync();tick=time.perf_counter()
        if frame:
            for _ in range(64):
                v=spectral.step(v,dt)
        sync()
        if frame:times.append(time.perf_counter()-tick)
        tick=time.perf_counter();u=spectral.real(v)
        d=spectral.diagnostics(u)
        if not bool(cp.isfinite(u).all()) or d['max_speed']*dt*128>=1 or d['fluid_energy']>=10:
            raise FloatingPointError('NS reference failed original finite-state/CFL/energy guard')
        rows.append(dict(frame=frame,time=.04*frame,**d))
        temp=folder/'velocity'/f'frame_{frame:03d}.tmp.npy'
        np.save(temp,cp.asnumpy(u.transpose(1,2,3,0)))
        temp.replace(folder/'velocity'/f'frame_{frame:03d}.npy')
        del u
        sync();output_seconds+=time.perf_counter()-tick
        progress('matched_dt_ns_reference',frame=frame,frames=30,
                 integration_seconds=sum(times),output_seconds=output_seconds)
    write(folder/'diagnostics.json',rows)
    result=dict(dt=dt,frames=30,steps=1856,final_time=1.16,
                integration_seconds=sum(times),frame_integration_seconds=times,
                output_seconds=output_seconds,precision='float64',
                initial='Original unmasked Fourier checkpoint',
                integrator='Original NS split viscosity/normal penalty with RK4 convection')
    write(receipt,result)
    del v;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return folder,result


def run(old,k=512,run_folder=None):
    global RUN
    if run_folder is not None:RUN=Path(run_folder)
    if (RUN/'accelerated_complete.json').exists():return
    cp,grid=old.cp,old.grid
    sync=cp.cuda.get_current_stream().synchronize
    began=time.perf_counter()
    source=json.loads((RUN/'derived_fields_complete.json').read_text())
    assembled=json.loads((RUN/'quadratic_complete.json').read_text())
    assert assembled['contract']==source['contract'] and assembled['all_nodes'] and assembled['all_modes']
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(Path(grid.__file__))==source['contract']['core_sha256']
    assert digest(OUT/f'evaluation_k{k}/velocity/fields.json')==source['contract']['fields_receipt_sha256']
    fields=SafeFields(Path(source['original_fields']),cp,expected_points=source['points'],expected_modes=k,block_rows=8192)
    flat=cp.asarray(np.load(grid.OLD/'fluid_flat_indices.npy'))
    with np.load(OUT/f'evaluation_k{k}/velocity/initial_projection.npz') as z:
        mass=z['mass'].copy();c0=cp.asarray(z['coefficients'],dtype='f8')
    linear=np.load(RUN/'linear.npy')
    tick=time.perf_counter()
    if k==512:
        quadratic=cp.asarray(np.load(RUN/'quadratic.npy'),blocking=True)
        block_bytes=quadratic.nbytes
        storage=dict(backend='GPU resident FP64',tensor_bytes=quadratic.nbytes)
    else:
        from stream_tensor import load
        quadratic,block_bytes,storage=load(RUN,k,cp)
    rom=CoefficientROM(mass,linear,quadratic,cp,block_bytes=block_bytes)
    sync();load_seconds=time.perf_counter()-tick
    spectral=grid.Spectral('f8')
    class Direct(grid.Basis):
        def __init__(self):
            self.flat,self.N=flat,fields.N
            self.mass,self.L=rom.mass,rom.cholesky
        def fluid(self,c):return fields.fluid(c)
        def transpose_fluid(self,f):return fields.transpose_fluid(f)
    direct=Direct()
    progress('production_rhs_equivalence')
    tick=time.perf_counter()
    errors=[]
    rng=np.random.default_rng(926)
    random=cp.asarray(rng.normal(size=k))
    random*=cp.sqrt((c0@rom.mass@c0)/(random@rom.mass@random))
    for coefficients in (c0,random):
        actual=rom.rhs(coefficients)
        expected=direct.rhs(coefficients,spectral)
        errors.append(float(cp.linalg.norm(actual-expected)/cp.linalg.norm(expected)))

    stepped=rom.step(c0,.000625)
    sync();step_tick=time.perf_counter()
    direct_stepped=direct.step(c0,.000625,spectral)
    sync();old_step_seconds=time.perf_counter()-step_tick
    delta=stepped-direct_stepped
    rk_error=float(cp.sqrt((delta@rom.mass@delta)/(direct_stepped@rom.mass@direct_stepped)))
    if max(errors)>2e-9 or rk_error>2e-10:
        raise ValueError(f'Production equivalence failed: RHS={errors}, RK4={rk_error}')
    validation=dict(rhs_relative=errors,rk4_mass_relative=rk_error,
                    seconds=time.perf_counter()-tick,full_grid_step_seconds=old_step_seconds,
                    scope=f'Actual trained K{k}, all fluid nodes, complete nonlinear/viscous/normal-penalty generator')
    write(RUN/'production_equivalence.json',validation)
    del random,actual,expected,stepped,direct_stepped,delta
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
    progress('production_equivalence_passed',**validation)

    loads,norms,reference_seconds=references(fields,flat,grid.D/'ns_dt0.0025','original',cp)
    load_gpu=cp.asarray(loads)
    initial_error=rom.relative_error(c0,load_gpu[:,0],norms[0])
    initial_known=json.loads((OUT/f'evaluation_k{k}/velocity/initial_projection_metrics.json').read_text())['relative_velocity_L2']
    if abs(initial_error-initial_known)>1e-9:
        raise ValueError('Reference metric/initial projection mismatch')
    all_rows={};all_coefficients={};integrations={}
    for dt in (.00125,.000625):
        c=c0.copy();rows=[];coefficients=[];times=[];first_frame=0
        saved=RUN/f'coefficient_dt{dt}_checkpoint.npz'
        if saved.exists():
            with np.load(saved) as z:
                assert float(z['dt'])==dt and int(z['K'])==k
                coefficients=list(z['coefficients']);c=cp.asarray(coefficients[-1])
                times=list(z['times']);rows=json.loads(str(z['rows']));first_frame=len(rows)

        warm=rom.step(c0,dt);sync();del warm
        for frame in range(first_frame,30):
            if (RUN/'user_requested_stop.json').exists():raise RuntimeError('User stopped this coefficient task')
            sync();tick=time.perf_counter()
            if frame:
                for _ in range(round(.04/dt)):
                    c=rom.step(c,dt)
            sync();elapsed=time.perf_counter()-tick
            if frame:times.append(elapsed)
            if not bool(cp.isfinite(c).all()) or float(.5*c@rom.mass@c)>=10:
                raise FloatingPointError('Coefficient RK4 failed finite-state/energy guard')
            rows.append(dict(frame=frame,time=.04*frame,dt=dt,
                         relative_velocity_L2=rom.relative_error(c,load_gpu[:,frame],norms[frame]),
                         mass_energy=float(.5*c@rom.mass@c),integration_seconds=elapsed if frame else 0.))
            coefficients.append(cp.asnumpy(c))
            temp=saved.with_suffix('.tmp.npz')
            np.savez(temp,K=k,dt=dt,coefficients=np.stack(coefficients),times=np.array(times),rows=json.dumps(rows))
            temp.replace(saved)
            progress('coefficient_rk4',dt=dt,frame=frame,frames=30,
                     integration_seconds=sum(times),relative_velocity_L2=rows[-1]['relative_velocity_L2'])
        all_rows[str(dt)]=rows;all_coefficients[str(dt)]=np.stack(coefficients)
        integrations[str(dt)]=dict(steps=29*round(.04/dt),seconds=sum(times),frame_seconds=times)
        np.save(RUN/f'coefficients_dt{dt}.npy',all_coefficients[str(dt)])
        write(RUN/f'metrics_dt{dt}.json',rows)
    differences=[]
    for a,b in zip(all_coefficients['0.00125'],all_coefficients['0.000625']):
        d=a-b
        differences.append(float(np.sqrt(max(0.,d@mass@d)/(b@mass@b))))


    previous=OUT/f'evaluation_k{k}/velocity/dt0.00125'
    comparisons=[]
    if (previous/'complete.json').exists():
        for frame,a in enumerate(all_coefficients['0.00125']):
            with np.load(previous/f'coefficients_{frame:03d}.npz') as z:b=z['coefficients']
            d=a-b
            comparisons.append(float(np.sqrt(max(0.,d@mass@d)/(b@mass@b))))
        if max(comparisons)>1e-7:
            raise ValueError(f'Long-trajectory full-grid equivalence failed: {max(comparisons)}')
    write(RUN/'trajectory_equivalence.json',dict(time_step_refinement_mass_relative=differences,
          prior_full_grid_coarse_mass_relative=comparisons))

    gt_folder,gt_timing=ns_reference(grid,spectral,cp)
    fine_loads,fine_norms,fine_ref_seconds=references(fields,flat,gt_folder,'dt0000625',cp)
    fine_load_gpu=cp.asarray(fine_loads)
    for key,rows in all_rows.items():
        for frame,row in enumerate(rows):
            row['matched_dt_gt_relative_velocity_L2']=rom.relative_error(
                cp.asarray(all_coefficients[key][frame]),fine_load_gpu[:,frame],fine_norms[frame])
        write(RUN/f'metrics_dt{key}.json',rows)


    tick=time.perf_counter();selected=[0,14,29]
    cs=cp.asarray(all_coefficients['0.000625'][selected].T)
    recon=cp.empty((3*fields.N,len(selected)),dtype='f8')
    for off,block in fields.blocks():
        recon[off:off+len(block)]=block@cs
    identity_differences=[];gt_convergence=[];selected_diagnostics=[]
    for column,frame in enumerate(selected):
        fluid=recon[:,column].reshape(-1,3)
        original=cp.asarray(np.load(grid.D/f'ns_dt0.0025/velocity/frame_{frame:03d}.npy'),blocking=True)
        fine=cp.asarray(np.load(gt_folder/'velocity'/f'frame_{frame:03d}.npy'),blocking=True)
        actual_error=float(cp.linalg.norm(fluid-fine.reshape(-1,3)[flat])/cp.linalg.norm(fine.reshape(-1,3)[flat]))
        reported=all_rows['0.000625'][frame]['matched_dt_gt_relative_velocity_L2']
        identity_differences.append(abs(actual_error-reported))
        gt_convergence.append(float(cp.linalg.norm((fine-original).reshape(-1,3)[flat])/cp.linalg.norm(fine.reshape(-1,3)[flat])))
        np.save(RUN/f'velocity_frame_{frame:03d}.npy',cp.asnumpy(fluid))
        full=cp.zeros((128**3,3),dtype='f8');full[flat]=fluid
        full=full.reshape(128,128,128,3).transpose(3,0,1,2)
        diag=spectral.diagnostics(full)
        if diag['max_speed']*.000625*128>=1 or diag['fluid_energy']>=10:
            raise FloatingPointError('Selected ROM output failed original physical guard')
        selected_diagnostics.append(dict(frame=frame,**diag))
        del original,fine,full
    sync();output_seconds=time.perf_counter()-tick
    if max(identity_differences)>1e-9:
        raise ValueError('Full-field metric differs from coefficient identity')
    del recon,cs


    fine_time=integrations['0.000625']['seconds']
    summary=dict(K=k,precision='FP64 operators/accumulation; original FP32 sampled basis cache retained',
        grid=128,fluid_points=fields.N,all_modes=True,all_nodes=True,
        final_time=1.16,reference_original=str(grid.D/'ns_dt0.0025'),reference_matched_dt=str(gt_folder),
        quadratic_assembly_seconds=assembled['seconds'],operator_load_seconds=load_seconds,tensor_storage=storage,
        derived_field_preparation_seconds=source.get('preparation_seconds'),
        validation=validation,rom_integration=integrations,ns_integration=gt_timing,
        matched_dt_online_speedup=gt_timing['integration_seconds']/fine_time,
        streamed_to_coefficient_step_speedup=old_step_seconds/(fine_time/1856),
        original_reference_projection_seconds=reference_seconds,fine_reference_projection_seconds=fine_ref_seconds,
        selected_field_output_seconds=output_seconds,selected_frames=selected,
        selected_diagnostics=selected_diagnostics,selected_metric_identity_absolute=identity_differences,
        selected_gt_time_refinement_relative=gt_convergence,
        final_original_gt_error=all_rows['0.000625'][-1]['relative_velocity_L2'],
        final_matched_dt_gt_error=all_rows['0.000625'][-1]['matched_dt_gt_relative_velocity_L2'],
        max_rom_dt_refinement_relative=max(differences),
        max_prior_trajectory_difference=max(comparisons) if comparisons else None,
        evaluation_seconds=time.perf_counter()-began,
        timing_scope='Online integration includes GPU synchronization, excludes offline assembly/validation/reference projection/output. Both timings measured on same GPU in this run; NS split and ROM unsplit RK4.',
        future_reference_injection=False,per_step_field_reconstruction=False)
    write(RUN/'comparison.json',summary)
    from plot_coefficient import render
    render(RUN)
    write(RUN/'accelerated_complete.json',dict(status='complete',unix=time.time(),comparison=str(RUN/'comparison.json')))
    progress('accelerated_comparison_complete',**summary)
