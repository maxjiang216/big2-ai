"""Python mirror of research/az_nn_check.cpp — prints value + first logits +
argmax for the same fixed positions, to cross-check C++ inference.

    uv run python -m nn.az_nn_check models/az_player.pt models/az_opp.pt
"""

from __future__ import annotations

import sys

import torch

from nn.dataset import encode_exact_np, encode_upper_bound_np
import numpy as np

# Fixed test positions (mirrored in research/az_nn_check.cpp).
PLAYER = [
    (
        [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        8,
        6,
    ),
    ([0] * 13, [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0] * 13, 3, 0),
    (
        [1] * 13,
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        11,
        13,
    ),
]
OPP = [
    (
        [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        8,
        6,
    ),
    ([0] * 13, [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0] * 13, 3, 0),
    (
        [1] * 13,
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        11,
        13,
    ),
]


def main():
    player_pt, opp_pt = sys.argv[1], sys.argv[2]
    pnet = torch.jit.load(player_pt).eval()
    onet = torch.jit.load(opp_pt).eval()

    print("PLAYER")
    for i, (hand, opp, trick, osz, usz) in enumerate(PLAYER):
        h = torch.from_numpy(encode_exact_np(np.array([hand], np.int32)))
        o = torch.from_numpy(encode_upper_bound_np(np.array([opp], np.int32)))
        t = torch.from_numpy(encode_exact_np(np.array([trick], np.int32)))
        with torch.no_grad():
            v, logits = pnet(
                h, o, t, torch.tensor([osz / 16.0]), torch.tensor([usz / 16.0])
            )
        lg = logits[0]
        print(
            f"  pos{i} value={float(v[0]):.6f} logits[0..4]="
            + " ".join(f"{float(lg[j]):.6f}" for j in range(5))
            + f" argmax={int(lg.argmax())}"
        )

    print("OPP")
    for i, (hand, opp, trick, osz, usz) in enumerate(OPP):
        h = torch.from_numpy(encode_exact_np(np.array([hand], np.int32)))
        o = torch.from_numpy(encode_upper_bound_np(np.array([opp], np.int32)))
        t = torch.from_numpy(encode_exact_np(np.array([trick], np.int32)))
        with torch.no_grad():
            mv, logits = onet(
                h, o, t, torch.tensor([osz / 16.0]), torch.tensor([usz / 16.0])
            )
        mvr, lg = mv[0], logits[0]
        print(
            f"  pos{i} move_value[0..4]="
            + " ".join(f"{float(mvr[j]):.6f}" for j in range(5))
            + " logits[0..4]="
            + " ".join(f"{float(lg[j]):.6f}" for j in range(5))
            + f" argmax={int(lg.argmax())}"
        )


if __name__ == "__main__":
    main()
