import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from nn.dataset import make_az_split
PASS=0; NEG=-1e30
train, val = make_az_split("data/az_opp_gen0.parquet","opp",0.1,0,1.0)
m=torch.jit.load("models/az_opp_gen0.pt").eval()
def pass_calib(ds, tag):
    P=[];Y=[]
    with torch.no_grad():
        for hand,opp,trick,osz,usz,value,mask,tgt in DataLoader(ds,batch_size=4096):
            _,lg=m(hand,opp,trick,osz,usz)
            pi=F.softmax(lg.masked_fill(~mask,NEG),-1).numpy()
            sel=mask[:,PASS].numpy().astype(bool)
            P+=list(pi[sel,PASS]); Y+=list((tgt.numpy()[sel]==PASS).astype(float))
    P=np.array(P);Y=np.array(Y)
    print(f"\n{tag}: n(pass-possible)={len(Y)} base={Y.mean():.3f}")
    for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,.7),(.7,1.01)]:
        b=(P>=lo)&(P<hi)
        if b.sum()==0: continue
        am=Y[b].mean(); pm=P[b].mean(); se=np.sqrt(max(am*(1-am),1e-9)/b.sum())
        print(f"  [{lo:.1f},{hi:.1f}) pred={pm:.3f} actual={am:.3f}±{1.96*se:.3f} (n={int(b.sum())})")
pass_calib(train,"TRAIN"); pass_calib(val,"VAL")
