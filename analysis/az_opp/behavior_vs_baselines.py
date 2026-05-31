import numpy as np, torch, torch.nn.functional as F
from nn.dataset import Big2AZOppDataset, make_az_split
from nn.model_az import Big2NetOpp, OPP_HEAD_DIM, az_opp_head_index

# Same split as training (val_frac=0.1, seed 0).
train_ds, val_ds = make_az_split("data/az_opp_gen0.parquet", "opp", 0.1, 0, 1.0)
print(f"train={len(train_ds)} val={len(val_ds)}")

# ---- Baselines computed on the val split ----
NEG = -1e30
# marginal over head slots from TRAIN played moves
marg = np.zeros(OPP_HEAD_DIM)
for i in range(len(train_ds)):
    *_, tgt = train_ds[i]
    marg[int(tgt)] += 1
marg = marg / marg.sum()
log_marg = np.log(np.clip(marg, 1e-12, None))

uni_ce, marg_ce, n_legal_sum = 0.0, 0.0, 0
N = len(val_ds)
for i in range(N):
    *_, mask, tgt = val_ds[i]
    legal = mask.numpy().astype(bool)
    k = int(legal.sum())
    n_legal_sum += k
    uni_ce += np.log(k)  # uniform over legal
    lp = log_marg.copy()
    lp[~legal] = NEG
    lp = lp - np.log(np.exp(lp - lp.max()).sum()) - lp.max() * 0  # renorm in logspace
    # proper masked renorm:
    z = (
        np.log(np.exp(log_marg[legal] - log_marg[legal].max()).sum())
        + log_marg[legal].max()
    )
    marg_ce += -(log_marg[int(tgt)] - z)
uni_ce /= N
marg_ce /= N
print(f"avg legal-set size  = {n_legal_sum/N:.1f}")
print(f"uniform-over-legal CE = {uni_ce:.3f}  (perplexity {np.exp(uni_ce):.1f})")
print(f"marginal-freq    CE = {marg_ce:.3f}  (perplexity {np.exp(marg_ce):.1f})")

# ---- Model behavior CE + top-k accuracy on val ----
m = torch.jit.load("models/az_opp_gen0.pt").eval()
bs = 2048
ce = 0.0
top1 = top3 = tot = 0
from torch.utils.data import DataLoader

dl = DataLoader(val_ds, batch_size=bs)
with torch.no_grad():
    for hand, opp, trick, osz, usz, value, mask, tgt in dl:
        mv, logits = m(hand, opp, trick, osz, usz)
        logits = logits.masked_fill(~mask, NEG)
        lp = F.log_softmax(logits, -1)
        ce += F.nll_loss(lp, tgt, reduction="sum").item()
        pred = lp.argmax(-1)
        top1 += (pred == tgt).sum().item()
        t3 = lp.topk(3, -1).indices
        top3 += (t3 == tgt.unsqueeze(1)).any(1).sum().item()
        tot += len(tgt)
print(f"MODEL behavior   CE = {ce/tot:.3f}  (perplexity {np.exp(ce/tot):.1f})")
print(f"MODEL top-1 acc = {top1/tot:.3f}   top-3 acc = {top3/tot:.3f}")
