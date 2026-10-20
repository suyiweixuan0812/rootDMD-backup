"""Choose bridge strength from PAST queries, before drawing the next source.

The policy may depend on known stop J and history. It must never be fitted to
the current reference or candidate weights inside a posterior update. Conditional
on history and J, the proposal is fixed and its exact importance correction and
conditional-iSIR invariance remain valid.
"""
import math
import torch
from .path_ratio_smc import cps_mean


class PastKLBridge:
    def __init__(self,nodes=4,kl_budget=.25,decay=.95):
        if nodes<1 or kl_budget<=0 or not 0<=decay<=1:raise ValueError('Invalid past-KL bridge policy')
        self.proxy=[0.]*(nodes-1)
        self.budget=float(kl_budget)
        self.decay=float(decay)

    def estimated_kl(self,stop):
        if not 1<=stop<=len(self.proxy)+1:raise ValueError('Invalid known stopping node')
        return sum(self.proxy[:stop-1])

    def strength(self,stop):
        kl=self.estimated_kl(stop)
        if kl<=self.budget:return 0.
        return 1-math.sqrt(self.budget/kl)

    def observe(self,conditional_kls):
        if len(conditional_kls)>len(self.proxy):raise ValueError('Too many observed transitions')
        for k,v in enumerate(conditional_kls):
            v=float(v)
            if not math.isfinite(v) or v<0:raise ValueError('KL proxy must be finite and nonnegative')
            # Conservative decayed envelope, updated only AFTER the query.
            self.proxy[k]=max(self.decay*self.proxy[k],v)


@torch.no_grad()
def source_teacher_kernel_kls(states,positive_sigmas,eta,student,teacher):
    """Continuous-kernel KL at actual Student parents; not posterior KL.

    For BF16 this is a Gaussian pre-rounding KL proxy; data processing supplies
    an upper bound per conditional kernel, not equality to discrete KL.
    """
    out=[]
    for k in range(len(states)-1):
        x=states[k].float();s,t=float(positive_sigmas[k]),float(positive_sigmas[k+1])
        a=cps_mean(x,student(states[k],s).float(),s,t,eta)
        b=cps_mean(x,teacher(states[k],s).float(),s,t,eta)
        variance=(t*math.sin(eta*math.pi/2))**2
        out.append(float((a.double()-b.double()).square().sum()/(2*variance)))
    return out
