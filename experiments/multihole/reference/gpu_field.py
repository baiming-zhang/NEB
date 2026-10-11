"""Native CuPy evaluation of the unchanged preview's continuous scalar field.

The preview module is loaded in a separate namespace. Its Taylor formulas are
reused with a CuPy array backend; only polynomial multiplication and B-spline
evaluation are accelerated. No random draws occur when restoring sample00.
"""
from pathlib import Path
import importlib.util
import math
import numpy as np
import cupy as cp
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(__package__+'.random_preview_cupy', HERE/'random_fields.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
backend.np = cp

_mul = cp.RawKernel(r'''
extern "C" __global__ void mul(const double* a,const double* b,double* c,int N,int n,int m){
 int r=blockDim.x*blockIdx.x+threadIdx.x;if(r>=N*m)return;
 int k=r%m,row=r/m,d=0;while((d+1)*(d+2)/2<=k)++d;
 int i=k-d*(d+1)/2,j=d-i;double v=0;
 for(int p=0;p<=i;++p)for(int q=0;q<=j;++q){
  int aa=(p+q)*(p+q+1)/2+p,bb=(i+j-p-q)*(i+j-p-q+1)/2+i-p;
  v+=a[row*m+aa]*b[row*m+bb];
 } c[r]=v;
}''','mul',options=('--fmad=false',))

class Jet(backend.Jet):
    def __mul__(self, other):
        if not isinstance(other, Jet):
            return super().__mul__(other)
        assert self.n == other.n
        result = Jet(cp.zeros(len(self.a)), self.n)
        n, m = self.a.shape
        _mul(((n*m+255)//256,), (256,), (self.a, other.a, result.a, n, self.n, m))
        return result
    __rmul__ = __mul__

backend.Jet = Jet

def coordinates(points,case):
    x,y=(Jet.variable(points[:,i],i) for i in range(2))
    q=[1-((x-5)/4).power(12)-(y/4).power(12)]
    for obstacle in backend.geo.get_obstacles(case):
        c,s=math.cos(obstacle.rotation),math.sin(obstacle.rotation)
        xx=(x-obstacle.x)*c+(y-obstacle.y)*s
        yy=(x-obstacle.x)*(-s)+(y-obstacle.y)*c
        if obstacle.kind=='circle':
            value=(xx.power(2)+yy.power(2))/obstacle.size**2-1
        elif obstacle.kind=='square':
            value=(xx/obstacle.size).power(obstacle.power)+(yy/obstacle.size).power(obstacle.power)-1
        else:
            planes=[(xx*math.cos(a)+yy*math.sin(a))/obstacle.size-1
                    for a in np.pi/2+2*np.pi*np.arange(3)/3]
            shift=cp.max(cp.stack([p.a[:,0] for p in planes]),axis=0)
            value=sum(((p-shift)*obstacle.smooth_k).exp() for p in planes).log()/obstacle.smooth_k+shift
        q.append(value)
    return q

backend.coordinates=coordinates
_spline = cp.RawKernel(r'''
__device__ int span(const double*t,int len,int p,double x){
 int n=len-p-2;if(x>=t[n+1])return n;
 int l=p,h=n+1,m=(l+h)/2;
 while(x<t[m]||x>=t[m+1]){if(x<t[m])h=m;else l=m;m=(l+h)/2;}return m;
}
__device__ void basis(const double*t,int s,int p,double x,double*b){
 double l[6],r[6];b[0]=1.;
 for(int j=1;j<=p;++j){l[j]=x-t[s+1-j];r[j]=t[s+j]-x;double save=0;
  for(int k=0;k<j;++k){double temp=b[k]/(r[k+1]+l[j-k]);b[k]=save+r[k+1]*temp;save=l[j-k]*temp;}b[j]=save;}
}
extern "C" __global__ void spline(const double*xy,int n,const double*tx,int nx,const double*ty,int ny,
 const double*c,int px,int py,double*o){
 int r=blockDim.x*blockIdx.x+threadIdx.x;if(r>=n)return;
 double x=xy[2*r],y=xy[2*r+1],bx[6],by[6];
 int sx=span(tx,nx,px,x),sy=span(ty,ny,py,y);basis(tx,sx,px,x,bx);basis(ty,sy,py,y,by);
 double v=0.;int cy=ny-py-1;
 for(int i=0;i<=px;++i)for(int j=0;j<=py;++j)v+=bx[i]*by[j]*c[(sx-px+i)*cy+sy-py+j];o[r]=v;
}''','spline',options=('--fmad=false',))

class GPUField(backend.RandomField):
    def __init__(self, source):
        self.case, self.sample, self.seed = source.case, source.sample, source.seed
        self.scale = source.scale
        self.constants = list(source.constants)
        self.tcks = []
        for total in range(5):
            for i in range(total+1):
                j=total-i
                s=source.spline if total==0 else source.spline.partial_derivative(i,j)
                tx,ty,c=s.tck
                self.tcks.append((cp.asarray(tx),cp.asarray(ty),cp.asarray(c),5-i,5-j,math.factorial(i)*math.factorial(j)))

    def spline_jet(self, points):
        result = Jet(cp.zeros(len(points)),4)
        pts = cp.ascontiguousarray(points,dtype='f8')
        for k,(tx,ty,c,px,py,factor) in enumerate(self.tcks):
            values=cp.empty(len(points),dtype='f8')
            _spline(((len(points)+255)//256,),(256,),
                (pts,len(points),tx,len(tx),ty,len(ty),c,px,py,values))
            result.a[:,k]=values/factor
        return result

    def evaluate(self, points, chunk=16384):
        points=cp.asarray(points,dtype='f8')
        result=cp.empty((len(points),4),dtype='f8')
        for start in range(0,len(points),chunk):
            result[start:start+chunk]=self.evaluate_block(points[start:start+chunk])
        return result
