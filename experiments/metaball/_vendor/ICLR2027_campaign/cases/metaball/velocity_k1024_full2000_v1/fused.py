
from velocity_trace_jet2 import cp,J2
F=cp.ElementwiseKernel('T v, raw T g, raw T h','T y, raw T yg, raw T yh',r'''
T s=sin(v), c=cos(v);y=s;
for(int a=0;a<3;a++)yg[3*i+a]=c*g[3*i+a];
const int p[6]={0,0,0,1,1,2};const int q[6]={0,1,2,1,2,2};
for(int a=0;a<6;a++)yh[6*i+a]=c*h[6*i+a]-s*g[3*i+p[a]]*g[3*i+q[a]];
''','velocity3d_sine_jet2')
B=cp.ElementwiseKernel('T v, raw T g, raw T h, T av, raw T ag, raw T ah','T bv, raw T bg, raw T bh',r'''
T s=sin(v),c=cos(v);T z=0;
for(int a=0;a<3;a++){z+=ag[3*i+a]*g[3*i+a];bg[3*i+a]=c*ag[3*i+a];}
bv=c*av-s*z;
const int p[6]={0,0,0,1,1,2};const int q[6]={0,1,2,1,2,2};
for(int a=0;a<6;a++){
 T t=ah[6*i+a];int l=p[a],r=q[a];
 bv+=t*(-s*h[6*i+a]-c*g[3*i+l]*g[3*i+r]);
 bg[3*i+l]-=s*t*g[3*i+r];bg[3*i+r]-=s*t*g[3*i+l];bh[6*i+a]=c*t;
}
''','velocity3d_sine_reverse_jet2')
def sine(x):
    y=J2(cp.empty(x.value.shape,dtype=x.value.dtype),cp.empty(x.grad.shape,dtype=x.grad.dtype),cp.empty(x.hess.shape,dtype=x.hess.dtype))
    F(x.value,cp.ascontiguousarray(x.grad),cp.ascontiguousarray(x.hess),y.value,y.grad,y.hess);return y
def sine_reverse(x,a):
    y=J2(cp.empty(x.value.shape,dtype=x.value.dtype),cp.empty(x.grad.shape,dtype=x.grad.dtype),cp.empty(x.hess.shape,dtype=x.hess.dtype))
    B(x.value,cp.ascontiguousarray(x.grad),cp.ascontiguousarray(x.hess),a.value,cp.ascontiguousarray(a.grad),cp.ascontiguousarray(a.hess),y.value,y.grad,y.hess);return y
def install():
    import velocity_trace_jet2 as jet
    jet.sine=sine;jet.sine_reverse=sine_reverse
