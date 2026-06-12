"""Diagnose whether az_s1_player_gen1's policy head is collapsed vs az_player_gen0."""
import numpy as np
import torch
import pyarrow.parquet as pq
from nn.model_az import load_compose_matrix, LEGAL_MOVES

torch.manual_seed(0)
np.random.seed(0)

C = load_compose_matrix()  # [NUM_MOVES, 138]
NUM_MOVES = C.shape[0]
print(f"compose matrix C: {tuple(C.shape)} (NUM_MOVES={NUM_MOVES})")

gen0 = torch.jit.load("models/az_player_gen0.pt").eval()
gen1 = torch.jit.load("models/az_s1_player_gen1.pt").eval()

f = pq.ParquetFile("data/az_s1_player_gen1.parquet")
tbl = f.read_row_group(0)
N_SCAN = 30000
rows = tbl.slice(0, N_SCAN).to_pylist()

# Use a 2000-row sample. Keep rows with >=2 legal moves so argmax is meaningful;
# split into a general sample and a with-visits sample for target agreement.
import random
random.seed(0)
random.shuffle(rows)
sample = [r for r in rows if len(r["legal"]) >= 2][:2000]
print(f"sample rows: {len(sample)} (>=2 legal moves)")

def to_tensors(batch):
    enc = np.array([r["enc"] for r in batch], dtype=np.float32)  # [B,144]
    hand = torch.from_numpy(enc[:, 0:48])
    opp = torch.from_numpy(enc[:, 48:96])
    trick = torch.from_numpy(enc[:, 96:144])
    opp_size = torch.tensor([r["opp_size"] / 16.0 for r in batch], dtype=torch.float32)
    our_size = torch.tensor([r["our_size"] / 16.0 for r in batch], dtype=torch.float32)
    return hand, opp, trick, opp_size, our_size

hand, opp, trick, opp_size, our_size = to_tensors(sample)

# legal masks over NUM_MOVES move-id space
mask = torch.zeros(len(sample), NUM_MOVES, dtype=torch.bool)
for i, r in enumerate(sample):
    ids = [m for m in r["legal"] if m < NUM_MOVES]
    mask[i, ids] = True

def composed_policy(net):
    with torch.no_grad():
        value, head = net(hand, opp, trick, opp_size, our_size)  # head [B,138]
    logits = head @ C.T  # [B, NUM_MOVES]
    neg = torch.finfo(logits.dtype).min / 2
    logits = logits.masked_fill(~mask, neg)
    probs = torch.softmax(logits, dim=-1)
    return value.numpy(), logits, probs

v0, l0, p0 = composed_policy(gen0)
v1, l1, p1 = composed_policy(gen1)

arg0 = p0.argmax(dim=-1).numpy()
arg1 = p1.argmax(dim=-1).numpy()

def entropy(probs):
    pl = probs.clamp_min(1e-12)
    return float((-(probs * pl.log()).sum(dim=-1)).mean())

def pass_frac(arg):
    return float(np.mean(arg == 0))

print("\n=== ARGMAX move distribution ===")
print(f"gen0 argmax==pass(0): {pass_frac(arg0):.3f}   gen1 argmax==pass(0): {pass_frac(arg1):.3f}")
# How concentrated is the argmax across distinct moves?
u0, c0 = np.unique(arg0, return_counts=True)
u1, c1 = np.unique(arg1, return_counts=True)
print(f"gen0 distinct argmax moves: {len(u0)}; top move {u0[c0.argmax()]} chosen {c0.max()/len(sample):.3f}")
print(f"gen1 distinct argmax moves: {len(u1)}; top move {u1[c1.argmax()]} chosen {c1.max()/len(sample):.3f}")
print("gen1 top-5 argmax moves (id:frac):",
      [(int(u1[i]), round(float(c1[i]/len(sample)),3)) for i in np.argsort(-c1)[:5]])

print("\n=== Agreement ===")
print(f"gen1 argmax == gen0 argmax: {np.mean(arg1==arg0):.3f}")

# Agreement with training target (visit_moves argmax) on rows that have visits.
vis_idx = [i for i, r in enumerate(sample) if len(r["visit_moves"]) > 0]
if vis_idx:
    tgt = np.array([sample[i]["visit_moves"][int(np.argmax(sample[i]["visit_counts"]))] for i in vis_idx])
    a1 = arg1[vis_idx]; a0 = arg0[vis_idx]
    print(f"rows with visit targets: {len(vis_idx)}")
    print(f"gen1 argmax == target: {np.mean(a1==tgt):.3f}")
    print(f"gen0 argmax == target: {np.mean(a0==tgt):.3f}")
    print(f"target==pass frac: {np.mean(tgt==0):.3f}")

print("\n=== Policy entropy (over legal subset) ===")
print(f"gen0 mean entropy: {entropy(p0):.4f} nats")
print(f"gen1 mean entropy: {entropy(p1):.4f} nats")
# also max possible entropy given legal-move counts
nlegal = mask.sum(dim=-1).float()
print(f"mean log(#legal) [uniform entropy ceiling]: {float(nlegal.log().mean()):.4f} nats")

print("\n=== Value head ===")
print(f"gen0 value: mean {v0.mean():.4f} std {v0.std():.4f} min {v0.min():.4f} max {v0.max():.4f}")
print(f"gen1 value: mean {v1.mean():.4f} std {v1.std():.4f} min {v1.min():.4f} max {v1.max():.4f}")

# Raw 138-d head stats to check for dead/saturated head
with torch.no_grad():
    _, h0 = gen0(hand, opp, trick, opp_size, our_size)
    _, h1 = gen1(hand, opp, trick, opp_size, our_size)
print("\n=== Raw 138-d head logits ===")
print(f"gen0 head: mean {h0.mean():.3f} std {h0.std():.3f} per-dim-std(min/mean/max) "
      f"{h0.std(0).min():.3f}/{h0.std(0).mean():.3f}/{h0.std(0).max():.3f}")
print(f"gen1 head: mean {h1.mean():.3f} std {h1.std():.3f} per-dim-std(min/mean/max) "
      f"{h1.std(0).min():.3f}/{h1.std(0).mean():.3f}/{h1.std(0).max():.3f}")
