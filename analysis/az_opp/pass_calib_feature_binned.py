"""Feature-binned calibration of the opp behavior head's PASS prediction.

The earlier high-pass over-prediction was found by binning on the model's OWN
predicted P(pass) — which inflates the top bin via regression-to-the-mean whenever
the model has estimation error (a partial artifact, not fixable by data). This
script instead bins on a FEATURE of the position: whether a bomb sits in the
opponent's legal/plausible set (read straight from the legal mask). A bias of the
aggregate over a feature-defined group CANNOT be a prediction-binning artifact, so
this isolates the genuine under-fit component.

For each group we report mean predicted P(pass) vs actual pass rate (calibration of
the aggregate) and mean predicted bomb-mass vs actual bomb rate.

    uv run python -m analysis.az_opp.pass_calib_feature_binned [model.pt] [data.parquet]
"""
import sys
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from nn.dataset import make_az_split

PASS_SLOT = 0
BOMB_SLOTS = list(range(169, 325))  # engine bomb ids (identity in opp head)
NEG = -1e30

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/az_opp_gen0.pt"
data_path = sys.argv[2] if len(sys.argv) > 2 else "data/az_opp_gen0.parquet"

_, val_ds = make_az_split(data_path, "opp", 0.1, 0, 1.0)
m = torch.jit.load(model_path).eval()

# slots that are a NON-bomb, non-pass response (singles/doubles/triples/FH +
# straights/sisters/triple-straights): everything except pass(0) and bombs.
NONBOMB_SLOTS = [
    i for i in range(len(val_ds[0][6])) if i != PASS_SLOT and i not in set(BOMB_SLOTS)
]

Ppass, Ypass, Pbomb, Ybomb, bomb_possible, only_bomb = [], [], [], [], [], []
dl = DataLoader(val_ds, batch_size=4096)
with torch.no_grad():
    for hand, opp, trick, osz, usz, value, mask, tgt in dl:
        _, logits = m(hand, opp, trick, osz, usz)
        pi = F.softmax(logits.masked_fill(~mask, NEG), -1).numpy()
        mask = mask.numpy()
        tgt = tgt.numpy()
        sel = mask[:, PASS_SLOT]  # responding positions only
        bposs = mask[:, BOMB_SLOTS].any(1)
        # only pass-or-bomb available: a bomb is legal but no non-bomb response is.
        onlyb = bposs & ~mask[:, NONBOMB_SLOTS].any(1)
        Ppass += list(pi[sel, PASS_SLOT])
        Ypass += list((tgt[sel] == PASS_SLOT).astype(float))
        Pbomb += list(pi[sel][:, BOMB_SLOTS].sum(1))
        Ybomb += list(np.isin(tgt[sel], BOMB_SLOTS).astype(float))
        bomb_possible += list(bposs[sel].astype(bool))
        only_bomb += list(onlyb[sel].astype(bool))

Ppass = np.array(Ppass)
Ypass = np.array(Ypass)
Pbomb = np.array(Pbomb)
Ybomb = np.array(Ybomb)
bp = np.array(bomb_possible)
ob = np.array(only_bomb)


def grp(name, m):
    n = int(m.sum())
    if n == 0:
        print(f"  {name:28s}  n=0")
        return
    pp, yp = Ppass[m].mean(), Ypass[m].mean()
    pb, yb = Pbomb[m].mean(), Ybomb[m].mean()
    # 95% CI on the actual pass rate (binomial normal approx)
    se = (yp * (1 - yp) / n) ** 0.5
    print(
        f"  {name:28s}  n={n:6d}  "
        f"pass: pred={pp:.3f} actual={yp:.3f} [{yp-1.96*se:.3f},{yp+1.96*se:.3f}]   "
        f"bomb: pred={pb:.3f} actual={yb:.3f}"
    )


print(f"model={model_path}  data={data_path}")
print(f"responding positions: n={len(Ppass)}")
print("feature-binned (NOT prediction-binned) — bias here is genuine under-fit:")
grp("all responding", np.ones_like(bp, bool))
grp("bomb possible", bp)
grp("no bomb possible", ~bp)
grp("ONLY pass-or-bomb", ob)  # sharp feature for the high-pass region

# Within those, split by the high-pass prediction region to compare directly with
# the prediction-binned finding (pred pass in [0.7,0.9]).
grp("only-bomb & pred-pass .7-.9", ob & (Ppass >= 0.7) & (Ppass < 0.9))
grp("bomb-poss & pred-pass .7-.9", bp & (Ppass >= 0.7) & (Ppass < 0.9))
