"""Python mirror of research/az_nn_check.cpp — prints value / first logits /
argmax per head for the same fixed (history, side-input) cases, to cross-check
C++ seq-net inference (both the KV-cache and full-recompute C++ paths must
match this reference forward).

    uv run python -m nn.az_nn_check models/az_seq.pt
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from nn.dataset import encode_exact_np, encode_upper_bound_np

# Fixed test cases (mirrored in research/az_nn_check.cpp):
# (prefix, path, hand, opp_max, trick(unused by seq net), opp_size, our_size, otm)
CASES = [
    (
        [],
        [],
        [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
        16,
        16,
        1.0,
    ),
    (
        [3, 7, 0],
        [5, 0, 2],
        [1] * 13,
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0],
        11,
        13,
        0.0,
    ),
    (
        [1, 14, 0, 26, 0, 4, 8, 12, 0],
        [2, 6],
        [0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
        8,
        6,
        1.0,
    ),
]


def main():
    net = torch.jit.load(sys.argv[1]).eval()
    for i, (prefix, path, hand, oppmax, osz, usz, otm) in enumerate(CASES):
        seq = prefix + path
        tokens = torch.tensor([seq + [0]], dtype=torch.long)  # pad 1 to allow empty
        h48 = torch.from_numpy(encode_exact_np(np.array([hand], np.int64)))
        o48 = torch.from_numpy(encode_upper_bound_np(np.array([oppmax], np.int64)))
        with torch.no_grad():
            H = net.forward_full(tokens)
            hsel = H[:, len(seq)]
            v, pol, beh, qa = net.readout(
                hsel,
                h48,
                o48,
                torch.tensor([osz / 16.0]),
                torch.tensor([usz / 16.0]),
                torch.tensor([otm]),
                torch.tensor([0.0]),  # my_pts/50
                torch.tensor([0.0]),  # opp_pts/50
            )
        p, b, q = pol[0], beh[0], qa[0]
        print(
            f"  py   case{i} value={float(v[0]):.6f} policy[0..4]="
            + " ".join(f"{float(p[j]):.6f}" for j in range(5))
            + f" pamax={int(p.argmax())} behavior[0..2]="
            + " ".join(f"{float(b[j]):.6f}" for j in range(3))
            + f" bamax={int(b.argmax())} qa[0..2]="
            + " ".join(f"{float(q[j]):.6f}" for j in range(3))
        )


if __name__ == "__main__":
    main()
