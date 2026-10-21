"""Exact finite-state check of a NEXT-ROUND stopping-point RB proposal.

Not imported by any SD3 trainer. No GPU/image-quality result is produced.
Teacher/Student importance ratios stay in conditional i-SIR; a virtual Gibbs
update of J follows it. All probabilities and variance identities are enumerated.
"""
from itertools import product
from pathlib import Path
import math,json

states=[(0.,0.),(0.,1.),(1.,0.),(1.,1.)]
student=[.32,.18,.12,.38];teacher=[.45,.10,.35,.10];prior=[.4,.6]
sigma=.75;alpha=1-sigma;y=.35;n=3
likelihood=[[math.exp(-.5*((y-alpha*f)/sigma)**2) for f in x] for x in states]
z=sum(student[x]*prior[j]*likelihood[x][j] for x in range(4) for j in range(2))
pi=[student[x]*prior[j]*likelihood[x][j]/z for x in range(4) for j in range(2)]
f=[states[x][j] for x in range(4) for j in range(2)]
mean=sum(p*v for p,v in zip(pi,f));variance=sum(p*(v-mean)**2 for p,v in zip(pi,f))
transition=[[0.]*8 for _ in range(8)]
means=[0.,0.];second=[0.,0.];removed=[0.,0.]
for x0 in range(4):
 for j0 in range(2):
  start=2*x0+j0
  for proposals in product(range(4),repeat=n-1):
   probability=pi[start]*math.prod(teacher[x] for x in proposals)
   pool=[x0,*proposals]
   raw=[student[x]/teacher[x]*likelihood[x][j0] for x in pool]
   w=[r/sum(raw) for r in raw]
   jweights=[]
   for x in pool:
    q=[prior[j]*likelihood[x][j] for j in range(2)];jweights.append([r/sum(q) for r in q])
   old=sum(p*states[x][j0] for p,x in zip(w,pool))
   rb=sum(p*sum(q*states[x][j] for j,q in enumerate(qj)) for p,x,qj in zip(w,pool,jweights))
   for k,m in enumerate([old,rb]):means[k]+=probability*m;second[k]+=probability*m*m
   removed[0]+=probability*sum(p*(states[x][j0]-old)**2 for p,x in zip(w,pool))
   removed[1]+=probability*sum(p*q*(states[x][j]-rb)**2 for p,x,qj in zip(w,pool,jweights) for j,q in enumerate(qj))
   for p,x,qj in zip(w,pool,jweights):
    for j,q in enumerate(qj):
     transition[start][2*x+j]+=math.prod(teacher[t] for t in proposals)*p*q
stationary=[sum(pi[i]*transition[i][j] for i in range(8)) for j in range(8)]
variances=[s-m*m for s,m in zip(second,means)]
assert max(abs(p-q) for p,q in zip(stationary,pi))<1e-13
assert all(abs(m-mean)<1e-13 for m in means)
assert all(abs(variance-v-r)<1e-13 for v,r in zip(variances,removed))
assert all(v<=variance+1e-13 for v in variances)
result=dict(scope='Exact8-state,3-candidate mathematical oracle only; proposed virtual stopping-point Gibbs/Rao-Blackwell composition. Not an SD3 training feature or quality result.',reference_posterior_mean=mean,single_reference_variance=variance,ordinary_iSIR_RB_mean=means[0],ordinary_iSIR_RB_variance=variances[0],stopping_RB_mean=means[1],stopping_RB_variance=variances[1],stationarity_max_error=max(abs(p-q) for p,q in zip(stationary,pi)),variance_decomposition_max_error=max(abs(variance-v-r) for v,r in zip(variances,removed)),configuration=dict(student=student,teacher=teacher,stop_prior=prior,observation=y,sigma=sigma,candidates=n),limitation='Variance bounded by one posterior reference in this oracle and general RB proof; no general dominance over the existing RB estimator and no guarantee after clipping/Adam.')
root=Path(__file__).resolve().parents[1];out=root/'logs/path-ratio-round1/research-stop-rb-oracle.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
