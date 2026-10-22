"""Paper large-eddy vector-potential ensemble and analytic curl."""
import numpy as np

def guard():
    from run import guard as check
    return check()

def wavevectors():
    values=[]
    for x in range(-2,3):
        for y in range(-2,3):
            for z in range(-2,3):
                k=(x,y,z);norm=x*x+y*y+z*z
                if 1<=norm<=4 and next(v for v in k if v)!=abs(next(v for v in k if v)):continue
                if 1<=norm<=4:values.append(k)
    return np.array(values,dtype='i8')
WAVEVECTORS=wavevectors()

def draw(key):
    rng=np.random.Generator(np.random.Philox(key=int(key)))
    return rng.standard_normal((len(WAVEVECTORS),2,3))/np.linalg.norm(WAVEVECTORS,axis=1)[:,None,None]

def curl_points(raw,points):
    raw=np.asarray(raw,dtype='f8');points=np.asarray(points,dtype='f8')
    assert raw.shape==(len(WAVEVECTORS),2,3) and np.isfinite(raw).all() and points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
    out=np.zeros((len(points),3),'f8')
    for k,a in zip(WAVEVECTORS,raw):
        phase=2*np.pi*(points@k)
        out+=2*np.pi*(np.cos(phase)[:,None]*np.cross(k,a[1])[None,:]-np.sin(phase)[:,None]*np.cross(k,a[0])[None,:])
    return out

def raw_curl(raw,resolution):
    coordinate=(np.arange(resolution)+.5)/resolution
    out=np.zeros((3,resolution,resolution,resolution),dtype='f8')
    # Complete original grid, fixed ZYX order, bounded one XY slab at a time.
    yy,xx=np.meshgrid(coordinate,coordinate,indexing='ij')
    for z,c in enumerate(coordinate):
        guard()
        points=np.stack((xx.ravel(),yy.ravel(),np.full(xx.size,c)),axis=1)
        out[:,z]=curl_points(raw,points).T.reshape(3,resolution,resolution)
    return out
