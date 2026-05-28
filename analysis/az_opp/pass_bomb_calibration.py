import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from nn.dataset import make_az_split
from nn.model_az import OPP_HEAD_DIM

# head slots: pass = 0; bombs = engine ids [kBOMB_START,kSTRAIGHT5_START)=[169,325) (identity)
PASS_SLOT = 0
BOMB_SLOTS = list(range(169, 325))
NEG = -1e30

_, val_ds = make_az_split("data/az_opp_gen0.parquet", "opp", 0.1, 0, 1.0)
m = torch.jit.load("models/az_opp_gen0.pt").eval()

P_pass=[]; Y_pass=[]; P_bomb=[]; Y_bomb=[]
dl = DataLoader(val_ds, batch_size=4096)
with torch.no_grad():
    for hand,opp,trick,osz,usz,value,mask,tgt in dl:
        mv,logits = m(hand,opp,trick,osz,usz)
        pi = F.softmax(logits.masked_fill(~mask, NEG), -1).numpy()
        mask=mask.numpy(); tgt=tgt.numpy()
        # PASS: only rows where pass is possible (responding positions)
        sel = mask[:,PASS_SLOT]
        P_pass += list(pi[sel,PASS_SLOT]); Y_pass += list((tgt[sel]==PASS_SLOT).astype(float))
        # BOMB: rows where >=1 bomb slot is possible
        bmask = mask[:,BOMB_SLOTS].any(1)
        pb = pi[:,BOMB_SLOTS].sum(1)
        yb = np.isin(tgt, BOMB_SLOTS).astype(float)
        P_bomb += list(pb[bmask]); Y_bomb += list(yb[bmask])

def report(name, P, Y):
    P=np.array(P); Y=np.array(Y); n=len(Y)
    base=Y.mean(); pred=P.mean()
    eps=1e-9
    ll=-(Y*np.log(P+eps)+(1-Y)*np.log(1-P+eps)).mean()
    ll_base=-(Y*np.log(base+eps)+(1-Y)*np.log(1-base+eps)).mean()
    brier=((P-Y)**2).mean()
    # discrimination: AUC via rank
    order=np.argsort(P); r=np.empty(n); r[order]=np.arange(1,n+1)
    npos=Y.sum(); nneg=n-npos
    auc=(r[Y==1].sum()-npos*(npos+1)/2)/(npos*nneg) if npos>0 and nneg>0 else float('nan')
    print(f"\n=== {name}  (n={n}, action possible) ===")
    print(f"actual rate      = {base:.3f}")
    print(f"mean predicted   = {pred:.3f}   (calibration: pred vs actual)")
    print(f"log-loss model   = {ll:.3f}   vs base-rate {ll_base:.3f}")
    print(f"Brier            = {brier:.4f}")
    print(f"AUC (discrim.)   = {auc:.3f}")
    # reliability bins
    print("  reliability  [pred-bin] -> actual (count):")
    for lo in (0,.1,.3,.5,.7,.9):
        hi=lo+ (0.1 if lo==0 else (0.2 if lo<0.9 else 0.1))
        b=(P>=lo)&(P<hi)
        if b.sum(): print(f"    [{lo:.1f},{hi:.1f}) -> {Y[b].mean():.3f}  (n={int(b.sum())})")

report("PASS", P_pass, Y_pass)
report("BOMB (any)", P_bomb, Y_bomb)
