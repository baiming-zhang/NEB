
import csv,shutil
from runtime import *
import original_potential_k32_evaluate as ev
from native_cupy_evaluate import save_npz
from canonical_poisson_component import shift

def gauge(model,c,geo,points,t):
    with timed(t,'weak_evaluation_setup'):pipe=Pipeline(c,geo)


    pipe.fit=NeuralPoints(cp.asnumpy(pipe.x),geo,c,'f8')
    with timed(t,'canonical_basis'):
        state=pipe.compute(model,points,False,'f8');T=state['obj']['T'];co=state['co'];amp=state['amp']
        cm=dict(mean_rayleigh=float(cp.mean(state['obj']['losses'])),min_rayleigh=float(cp.min(state['obj']['losses'])),
            max_rayleigh=float(cp.max(state['obj']['losses'])),gram_max=float(state['obj']['gram_max']),
            potential_rank=int(cp.linalg.matrix_rank(state['p'].T@state['p'])),velocity_rank=int(cp.linalg.matrix_rank(state['u'].T@state['u'])),
            canonical_note='Small resolution8 constrained weak projection and compatible harmonic; harmonic coefficient on fixed103244 source points, unlike original assessment16384; whole-pipeline comparison',
            stage_timings=state['timings'],solvers=state['solvers'])
        assert cm['potential_rank']==cm['velocity_rank']==32
        save_npz(D/'canonical_basis.npz',transform=T,scalar_coefficients=co,harmonic_scalar_coefficients=pipe.hco,
                 harmonic_amplitudes=amp,points=points,losses=state['obj']['losses'],order=state['obj']['order'])
        del state
    with timed(t,'independent_absolute_and_normalized_divergence'):
        x=fluid_points(c,geo,512,1919727);xx=cp.asarray(x,dtype='f8');raw=network_fields(model,NeuralPoints(x,geo,c,'f8'),'aud')
        G=pipe.scalar.operator(xx,'grad');L=pipe.scalar.operator(xx,'lap');_,dg,lg,_=lift_fields(xx,geo,c)
        h=dg.reshape(-1,1)-G@pipe.hco;hd=lg.reshape(-1,1)-L@pipe.hco
        p=(raw['a']-G@co-h@amp)@T;div=(raw['d']-L@co-hd@amp)@T
        cm['independent_canonical_divergence_per_potential']=cp.asnumpy(cp.linalg.norm(div,axis=0)/cp.maximum(cp.linalg.norm(p,axis=0),1e-30)).tolist()
        cm['absolute_canonical_divergence_rms']=cp.asnumpy(cp.sqrt(cp.mean(div*div,axis=0))).tolist()
        cm['absolute_raw_channel_canonical_divergence_rms']=cp.asnumpy(cp.sqrt(cp.mean((raw['d']-L@co-hd@amp)**2,axis=0))).tolist()
        save_npz(D/'canonical_diagnostic_fields.npz',points=x,potential=p,divergence=div)
        del raw,G,L,h,hd,p,div
    with timed(t,'gauge_stress_full_parameter_gradients'):
        pipe.fit=NeuralPoints(cp.asnumpy(pipe.x),geo,c,'f4')
        points=fluid_points(c,geo,256,908001);base=pipe.compute(model,points,True)
        data,G,h=pipe.batch(points);sh,_=shift(cp.asarray(points,dtype='f8'),geo,c)
        shf=cp.concatenate([shift(pipe.x[i:i+2048],geo,c)[0] for i in range(0,len(pipe.x),2048)]);dco=pipe.solve(shf,'gauge_shift');damp=pipe.hf.T@(shf-pipe.fit_field(dco))/pipe.hm
        delta=sh-G@dco-h@damp
        scale=cp.linalg.norm(base['a'],axis=0)/cp.linalg.norm(sh);sign=cp.where((base['p'].T@delta)[:,0]>=0,1.,-1.)
        shifted=potential_objective(base['p']+delta*(scale*sign)[None,:],base['u'],len(points))
        rt={};grads=pipe.reverse(model,data,G,h,shifted,rt)
        a=base['grads'][0]+base['grads'][1];b=grads[0]+grads[1]
        cm['gauge_stress']=dict(relative_loss_change=float(cp.sum(shifted['losses'])/cp.sum(base['obj']['losses'])-1),
            parameter_gradient_relative_change=float(cp.sqrt(sum(cp.sum((y-z)**2) for y,z in zip(a,b))/sum(cp.sum(y*y) for y in a))),
            remaining_gradient_fraction=float(cp.linalg.norm(delta)/cp.linalg.norm(sh)),training_gate=False,
            amplitude='RMS matched original admissible scalar shift; detached amplitude and sign; fixed-fit harmonic coefficient differentiated',
            scalar_rhs_precision='FP64 shift; actual FP32 network path',solvers=pipe.scalar.records)
    write(D/'canonical_metrics.json',cm)
    return T,cm
