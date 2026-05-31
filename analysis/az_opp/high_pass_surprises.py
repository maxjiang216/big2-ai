import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from nn.dataset import make_az_split

PASS = 0
NEG = -1e30
train, _ = make_az_split("data/az_opp_gen0.parquet", "opp", 0.1, 0, 1.0)
m = torch.jit.load("models/az_opp_gen0.pt").eval()
ppass = []
played_p = []
nonpass_mass = []
n_legal = []
played_moves = []
with torch.no_grad():
    for hand, opp, trick, osz, usz, value, mask, tgt in DataLoader(
        train, batch_size=4096
    ):
        _, lg = m(hand, opp, trick, osz, usz)
        pi = F.softmax(lg.masked_fill(~mask, NEG), -1)
        sel = (mask[:, PASS]) & (pi[:, PASS] >= 0.7)  # model very confident pass
        if sel.sum() == 0:
            continue
        s = pi[sel]
        tg = tgt[sel]
        mk = mask[sel]
        for i in range(len(tg)):
            ppass.append(float(s[i, PASS]))
            if int(tg[i]) != PASS:  # opponent did NOT pass (the surprises)
                played_p.append(float(s[i, int(tg[i])]))
                nonpass_mass.append(float(1 - s[i, PASS]))
                n_legal.append(int(mk[i].sum()))
                played_moves.append(int(tg[i]))
played_p = np.array(played_p)
print(
    f"high-confidence-pass positions (train, pi_pass>=0.7): mean pi(pass)={np.mean(ppass):.3f}"
)
print(f"  of these, the opponent did NOT pass in {len(played_p)} cases")
print(
    f"  when they played: model's mean prob on the move they ACTUALLY played = {played_p.mean():.3f}"
)
print(
    f"     median = {np.median(played_p):.3f}   (so the 'surprise' was deemed very unlikely)"
)
print(
    f"  model's total non-pass mass in these spots = {np.mean(nonpass_mass):.3f} spread over ~{np.mean(n_legal):.0f} legal moves"
)
import collections

c = collections.Counter(played_moves)
print(
    f"  distinct surprise moves played = {len(c)} (spread, not one move): top5 {c.most_common(5)}"
)
