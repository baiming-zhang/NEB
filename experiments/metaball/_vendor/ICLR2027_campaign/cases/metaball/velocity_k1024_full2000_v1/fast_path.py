
import velocity_trace_jet2 as j
from native_cupy_core import cp

def forward(model,data,start=0,stop=32,cache=False):
    x=data.features;hist=[];dtype=x.value.dtype
    for layer in range(len(model.W)):
        W=model.W[layer][start:stop].astype(dtype,copy=False)
        b=model.b[layer][start:stop].astype(dtype,copy=False)
        if layer==len(model.W)-1 and cache:
            hist.append((x,None,1,W))
            return None,hist
        pre=j.linear(x,W,b)
        if layer<len(model.W)-1:
            omega=model.c.siren_first_omega0 if layer==0 else model.c.siren_hidden_omega0
            pre=j.scaled(pre,omega)
            if cache:hist.append((x,pre,omega,W))
            x=j.sine(pre)
        else:x=pre
    return j.product(data.envelope,x),hist

def backward(model,data,hist,adj):
    a=j.product_reverse(data.envelope,adj);gw=[];gb=[]
    for index in range(len(hist)-1,-1,-1):
        source,pre,omega,W=hist[index]
        if pre is not None:a=j.scaled(j.sine_reverse(pre,a),omega)
        if index==0:
            ap=j.pack(a);xp=j.pack(source)
            w=ap.transpose(0,2,1)@xp;b=a.value.sum(axis=1)
        else:a,w,b=j.linear_reverse(source,a,W)
        gw.append(w);gb.append(b)
    return gw[::-1],gb[::-1]

def install():
    import core
    core.forward=forward;core.backward=backward
