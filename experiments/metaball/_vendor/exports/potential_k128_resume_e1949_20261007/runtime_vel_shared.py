







from _neb_paths import Path
import importlib.util

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'ICLR2027_campaign/cases/metaball/velocity_k1024_full2000_v1/fused.py'

def install(cm,variant='minimal'):
    if variant not in ('literal','minimal'):raise ValueError(variant)
    cp=cm.cp;ns=cm.train.Model.forward.__globals__;old=ns['old'];Jet1=ns['Jet1']
    prior_forward=old.sine_forward_jet1;prior_reverse=old.sine_backward_jet1
    spec=importlib.util.spec_from_file_location('historical_vel_fused_reference',SOURCE)
    vel=importlib.util.module_from_spec(spec);spec.loader.exec_module(vel)
    F=cp.ElementwiseKernel('T v, raw T g','T y, raw T yg',r'''
T s=sin(v), c=cos(v);y=s;
for(int a=0;a<3;a++)yg[3*i+a]=c*g[3*i+a];
''','matched_original_velocity_shared_sine')
    B=cp.ElementwiseKernel('T v, raw T g, T av, raw T ag','T bv, raw T bg',r'''
T s=sin(v),c=cos(v);T z=0;
for(int a=0;a<3;a++){z+=ag[3*i+a]*g[3*i+a];bg[3*i+a]=c*ag[3*i+a];}
bv=c*av-s*z;
''','matched_original_velocity_shared_sine_reverse')
    def sine(x):
        y=Jet1(cp.empty_like(x.value),cp.empty_like(x.grad))
        if variant=='literal':
            zeros=cp.zeros(x.value.shape+(6,),x.value.dtype);yh=cp.empty_like(zeros)
            vel.F(x.value,cp.ascontiguousarray(x.grad),zeros,y.value,y.grad,yh)
        else:F(x.value,cp.ascontiguousarray(x.grad),y.value,y.grad)
        return y
    def reverse(x,a):
        y=Jet1(cp.empty_like(x.value),cp.empty_like(x.grad))
        if variant=='literal':
            zeros=cp.zeros(x.value.shape+(6,),x.value.dtype);yh=cp.empty_like(zeros)
            vel.B(x.value,cp.ascontiguousarray(x.grad),zeros,a.value,cp.ascontiguousarray(a.grad),zeros,y.value,y.grad,yh)
        else:B(x.value,cp.ascontiguousarray(x.grad),a.value,cp.ascontiguousarray(a.grad),y.value,y.grad)
        return y
    old.sine_forward_jet1=sine;old.sine_backward_jet1=reverse
    spec=importlib.util.spec_from_file_location('matched_original_velocity_fast_path',HERE/'runtime_matched.py')
    fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast)


    metadata=fast.install(cm,variant='minimal')
    fast_restores=list(cm._selected_runtime_restore)
    def restore():
        for callback in reversed(fast_restores):callback()
        old.sine_forward_jet1=prior_forward;old.sine_backward_jet1=prior_reverse
    cm._selected_runtime_restore=[restore]
    metadata.update(name='Historical velocity shared sine/fast-path arithmetic, Jet1 required channels',
        sine_reference=str(SOURCE),shared_compiler_flags='Original velocity defaults unchanged',
        shared_kernel_reference=variant,additional_trig_or_pack_cache=False,
        shared_geometry_approximation=False,shared_new_optimization=False,
        inherent_derivative_order='Potential Jet1 versus velocity Jet2 as required by physical objectives',
        qualification='Requires actual original-kernel shared-channel comparison and complete-state preflight')
    return metadata
