
from original_potential_k32_core import cp,NeuralPoints
from canonical_poisson_polynomial import POWERS
from cupyx.scipy.sparse import csr_matrix,hstack

def fields(x,geo,c):
    q=(x-.5)/c.outer_half_extent
    out=1-cp.sum(q**c.outer_superellipsoid_power,axis=1)
    go=-c.outer_superellipsoid_power/c.outer_half_extent*q**(c.outer_superellipsoid_power-1)
    delta=x[:,None]-geo[None,:,:3];length=cp.linalg.norm(delta,axis=2);d=length-geo[None,:,3]
    low=cp.min(d,axis=1);exp=cp.exp(-(d-low[:,None])/c.metaball_softmin_tau);total=cp.sum(exp,axis=1)
    inner=low-c.metaball_softmin_tau*cp.log(total)
    gi=cp.sum((exp/total[:,None])[:,:,None]*delta/length[:,:,None],axis=1)
    fo=out/(c.outer_envelope_transition+out);fi=inner/(c.obstacle_envelope_transition+inner)
    dfo=c.outer_envelope_transition*go/(c.outer_envelope_transition+out)[:,None]**2
    dfi=c.obstacle_envelope_transition*gi/(c.obstacle_envelope_transition+inner)[:,None]**2
    e=fo*fi;de=dfo*fi[:,None]+dfi*fo[:,None]
    do=out*(c.outer_half_extent/c.outer_superellipsoid_power);ddo=go*(c.outer_half_extent/c.outer_superellipsoid_power)
    den=do+inner;lift=do/den;dlift=(ddo*inner[:,None]-do[:,None]*gi)/den[:,None]**2
    return e,de,lift,dlift

def gradient(base,x,geo,c):
    e,de,_,dlift=fields(x,geo,c)
    vals=[];rows=[];cols=[]
    for first in range(0,len(x),512):
        xx=x[first:first+512]

        ijk=cp.floor(xx/base.dx-.5).astype('i4')[:,None,:]+base.offsets[None,:,:]
        valid=cp.all((ijk>=0)&(ijk<base.nside),axis=2)
        flat=(ijk[:,:,0]*base.nside+ijk[:,:,1])*base.nside+ijk[:,:,2]
        col=base.lookup[cp.clip(flat,0,base.nside**3-1)];valid&=col>=0
        delta=xx[:,None,:]-base.centers[cp.maximum(col,0)];r=cp.linalg.norm(delta,axis=2)/base.radius
        valid&=r<1;row,slot=cp.nonzero(valid);col=col[row,slot];delta=delta[row,slot];r=r[row,slot]
        t=1-r;f=t**8*(32*r**3+25*r**2+8*r+1);g=-22*t**7*(16*r**2+7*r+1)
        gf=g[:,None]*delta/base.radius**2
        value=(de[first+row]*f[:,None]+e[first+row,None]*gf).reshape(-1)
        rr=(3*(row+first)[:,None]+cp.arange(3)).reshape(-1);cc=cp.repeat(col,3)
        vals.append(value/base.center_e[cc]);rows.append(rr);cols.append(cc)
    G=csr_matrix((cp.concatenate(vals),(cp.concatenate(rows),cp.concatenate(cols))),shape=(3*len(x),base.n))
    y=2*x-1
    def mono(a):return cp.prod(y**cp.asarray(a)[None,:],axis=1)
    p=cp.stack([mono(a) for a in POWERS],axis=1);dp=cp.zeros((len(x),10,3),dtype='f8')
    for j,a in enumerate(POWERS):
        for k in range(3):
            if a[k]:
                b=list(a);b[k]-=1;dp[:,j,k]=2*a[k]*mono(b)
    poly=(de[:,None,:]*p[:,:,None]+e[:,None,None]*dp).transpose(0,2,1).reshape(-1,10)
    return hstack([G,csr_matrix(poly)],format='csr'),dlift.reshape(-1,1)

def batch(pipe,points,dtype='f4'):
    data=NeuralPoints(points,pipe.geo,pipe.c,dtype);x=cp.asarray(points,dtype='f8')
    G,dlift=gradient(pipe.scalar.base,x,pipe.geo,pipe.c)
    return data,G,dlift-G@pipe.hco
