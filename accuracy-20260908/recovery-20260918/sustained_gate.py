"""Offline development prototype. Does not alter production or heldout settings."""
from collections import deque
class SustainedEvidence:
    def __init__(self, threshold=.65, window=2., fraction=.8, max_gap=.4):
        self.threshold,self.window,self.fraction,self.max_gap=threshold,window,fraction,max_gap
        self.samples=deque()
    def observe(self,t,score):
        if self.samples and (t<=self.samples[-1][0] or t-self.samples[-1][0]>self.max_gap):
            self.samples.clear()
        self.samples.append((t,score))
        while len(self.samples)>1 and self.samples[1][0]<=t-self.window:
            self.samples.popleft()
        if t-self.samples[0][0]<self.window-1e-6:return False
        # Time weighted evidence: duplicate/uneven samples cannot inflate votes.
        credit=sum(max(0,b[0]-max(a[0],t-self.window)) for a,b in zip(self.samples,list(self.samples)[1:]) if a[1]>=self.threshold and b[1]>=self.threshold)
        return score>=self.threshold and credit>=self.window*self.fraction-1e-6
if __name__=='__main__':
    import json,pathlib
    root=pathlib.Path(__file__).parent;results=[]
    for row in json.loads((root/'development-diagnostic.json').read_text()):
        gate=SustainedEvidence();hits=[x['t'] for x in row['trace'] if gate.observe(x['t'],x['score'])]
        baseline=next(g['triggered'] for g in row['gates'] if g['threshold']==.8)
        results.append({'id':row['reference']['id'],'label':row['reference']['label'],'roi080_trigger':baseline,'sustained_trigger':bool(hits),'combined_trigger':baseline or bool(hits),'first_sustained':hits[0] if hits else None})
    for name,series,expected in [('short_spike',[(i*.2,.9 if 5<=i<8 else .1) for i in range(30)],False),('sustained',[(i*.2,.7) for i in range(30)],True),('gaps',[(i,.9) for i in range(10)],False),('duplicates',[(0,.9)]*30,False)]:
        g=SustainedEvidence();actual=any([g.observe(t,s) for t,s in series]);assert actual==expected,name
    (root/'sustained-results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));print('4 temporal edge checks passed')
