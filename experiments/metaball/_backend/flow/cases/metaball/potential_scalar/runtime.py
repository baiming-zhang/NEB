
from runtime_base import *
import runtime_base as base
import grouped8
import value_reverse
base.network_fields=grouped8.network_fields
base.network_backward=grouped8.network_backward
from small import Pipeline as SmallPipeline
class Pipeline(SmallPipeline):
    def reverse(self,model,data,G,h,obj,t):
        with timed(t,'harmonic_and_coefficient_reverse'):
            beta=h.T@obj['GA']/self.hm;rhs=-G.T@obj['GA']+self.Gfh@beta
        with timed(t,'poisson_transpose'):adj=self.solve(rhs,'transpose',True)
        with timed(t,'harmonic_reverse_fit'):adj-=self.hf@beta
        with timed(t,'parameter_reverse_query'):grad=grouped8.network_backward(model,data,value=obj['GA'],curl=obj['GU'])
        with timed(t,'parameter_reverse_source'):
            grad=value_reverse.network_backward_value(model,self.fit,adj,grads=grad) if self.fit.x.dtype==cp.float32 else grouped8.network_backward(model,self.fit,value=adj,grads=grad)
        return grad
network_fields=grouped8.network_fields
network_backward=grouped8.network_backward
