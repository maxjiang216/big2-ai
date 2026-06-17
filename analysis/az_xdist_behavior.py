"""Cross-distribution diagnostic: score a trained seq model's per-head val
losses on a DIFFERENT generation's data without training on it.

Answers: does the behavior head generalize to an unseen opponent policy
(e.g. teacher-trained gen0 scored on gen1 NN-self-play games)?

    uv run python analysis/az_xdist_behavior.py models/az_seq_gen0.pt \
        data/az_games_gen1.parquet data/az_player_gen1.parquet data/az_opp_gen1.parquet
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
from nn.dataset_seq import make_seq_split  # noqa: E402
from nn.model_az import load_compose_matrix  # noqa: E402
from nn.train_az import _device  # noqa: E402
from nn.train_az_seq import _step  # noqa: E402


def main() -> None:
    model_path, games, player, opp = sys.argv[1:5]
    device = _device("auto")
    model = torch.jit.load(model_path, map_location=device).eval()
    C = load_compose_matrix().to(device)
    # val_frac tiny: score the big (train) side too — we never train here.
    tl, vl = make_seq_split(
        [(games, player, opp)], 512, device, val_frac=0.05, seed=0, validate=False
    )
    sums = [0.0] * 4
    nps = nos = 0
    with torch.no_grad():
        for loader in (tl, vl):
            for batch in loader:
                _, parts, (n_p, n_o) = _step(model, C, batch)
                lv, lp, lb, lq = (float(x) for x in parts)
                sums[0] += lv * n_p
                sums[1] += lp * n_p
                sums[2] += lb * n_o
                sums[3] += lq * n_o
                nps += n_p
                nos += n_o
    print(
        f"{model_path} on {games}: "
        f"v={sums[0] / nps:.3f} p={sums[1] / nps:.3f} "
        f"b={sums[2] / nos:.3f} q={sums[3] / nos:.3f} "
        f"({nps:,} player / {nos:,} opp rows)"
    )


if __name__ == "__main__":
    main()
