"""Native CuPy type2 Gaussian NUFFT, with exact Fourier sums for qualification.

For oversampled grid M, tau=pi*w/(M*sqrt(M*(M-N))), Gaussian interpolation
undoes multiplication of each Fourier coefficient by exp(tau*|k|^2).
Periodic-image alias and stencil truncation are both exponentially small.
This is only an evaluator of the represented Fourier series, not an FD solver.
"""
import cupy as cp
import numpy as np
from scipy.fft import next_fast_len

_kernel=cp.RawKernel(r'''
extern "C" __global__ void interp(const double2* g,const double* x,const double* y,const double* z,
 double2* out,int m,int count,double beta) {
 int p=blockIdx.x,t=threadIdx.x;if(p>=count)return;
 __shared__ double w[84];__shared__ int idx[84];__shared__ double sr[128],si[128];
 if(t<84){int axis=t/28,j=t%28;double u=(axis==0?x[p]:(axis==1?y[p]:z[p]));
  u=u-floor(u/m)*m;int l=(int)floor(u)-13+j;double d=u-l;w[t]=exp(-beta*d*d);
  idx[t]=(l%m+m)%m;}
 __syncthreads();double r=0,i=0;
 for(int j=t;j<21952;j+=128){int a=j/784,b=(j/28)%28,c=j%28;
  long long offset=((long long)idx[a]*m+idx[28+b])*m+idx[56+c];
  double v=w[a]*w[28+b]*w[56+c];double2 q=g[offset];r+=v*q.x;i+=v*q.y;}
 sr[t]=r;si[t]=i;__syncthreads();
 for(int stride=64;stride;stride/=2){if(t<stride){sr[t]+=sr[t+stride];si[t]+=si[t+stride];}__syncthreads();}
 if(t==0){double factor=pow(beta/3.14159265358979323846,1.5);out[p]=make_double2(sr[0]*factor,si[0]*factor);}
}
''','interp')


def evaluate(coeff,angles):
    n=coeff.shape[0];m=next_fast_len(2*n);w=14
    tau=np.pi*w/(m*np.sqrt(m*(m-n)));beta=np.pi**2/(tau*m*m)
    k=cp.arange(-(n//2),n//2+1,dtype='f8');kk=k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2
    ids=cp.asarray(k,dtype='i4')%m
    padded=cp.zeros((m,m,m),dtype='c16')
    padded[ids[:,None,None],ids[None,:,None],ids[None,None,:]]=coeff*cp.exp(tau*kk)
    del kk
    cp.fft.config.get_plan_cache().set_size(1)
    grid=cp.fft.ifftn(padded);del padded;grid*=m**3
    coordinates=[cp.ascontiguousarray(angles[:,j]*(m/(2*np.pi))) for j in range(3)]
    result=cp.empty(len(angles),dtype='c16')
    _kernel((len(angles),),(128,),(grid,*coordinates,result,m,len(angles),beta))
    return result


def direct(coeff,angles):
    n=coeff.shape[0];k=cp.arange(-(n//2),n//2+1,dtype='f8');out=[]
    for off in range(0,len(angles),64):
        xx=angles[off:off+64];e=[cp.exp(1j*xx[:,j,None]*k[None,:]) for j in range(3)]
        q=(e[0]@coeff.reshape(n,-1)).reshape(-1,n,n)
        q=cp.einsum('nij,ni->nj',q,e[1]);out.append(cp.sum(q*e[2],axis=1))
    return cp.concatenate(out)


_spread=cp.RawKernel(r'''
extern "C" __global__ void spread(double2* g,const double* x,const double* y,const double* z,
 const double2* cot,int m,int count,double beta) {
 int p=blockIdx.x,t=threadIdx.x;if(p>=count)return;
 __shared__ double w[84];__shared__ int idx[84];
 if(t<84){int axis=t/28,j=t%28;double u=(axis==0?x[p]:(axis==1?y[p]:z[p]));
  u=u-floor(u/m)*m;int l=(int)floor(u)-13+j;double d=u-l;w[t]=exp(-beta*d*d);
  idx[t]=(l%m+m)%m;}
 __syncthreads();double factor=pow(beta/3.14159265358979323846,1.5);double2 q=cot[p];
 for(int j=t;j<21952;j+=128){int a=j/784,b=(j/28)%28,c=j%28;
  long long offset=((long long)idx[a]*m+idx[28+b])*m+idx[56+c];
  double v=factor*w[a]*w[28+b]*w[56+c];
  atomicAdd(&g[offset].x,v*q.x);atomicAdd(&g[offset].y,v*q.y);}
}
''','spread')


def adjoint(cot,angles,n):
    """Hermitian transpose of evaluate, including its actual interpolation."""
    m=next_fast_len(2*n);w=14
    tau=np.pi*w/(m*np.sqrt(m*(m-n)));beta=np.pi**2/(tau*m*m)
    grid=cp.zeros((m,m,m),dtype='c16')
    coordinates=[cp.ascontiguousarray(angles[:,j]*(m/(2*np.pi))) for j in range(3)]
    cot=cp.ascontiguousarray(cot,dtype='c16')
    _spread((len(angles),),(128,),(grid,*coordinates,cot,m,len(angles),beta))
    cp.fft.config.get_plan_cache().set_size(1)
    transformed=cp.fft.fftn(grid);del grid
    k=cp.arange(-(n//2),n//2+1,dtype='f8');ids=k.astype('i4')%m
    out=transformed[ids[:,None,None],ids[None,:,None],ids[None,None,:]];del transformed
    out*=cp.exp(tau*(k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2))
    return out
