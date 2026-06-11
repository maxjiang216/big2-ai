"""Convert a pre-series Big2NetSeqAZ checkpoint to the series-aware layout.

The only architectural change is size_embed: Linear(3, 16) -> Linear(5, 16),
adding two input columns for (my_pts/50, opp_pts/50). Those columns are
zero-initialised, so the converted net is bit-identical to the original on any
input with zero series points. This lets the current champion seed both the
Markov-chain estimation and the first series-regime generation without retrain.

    uv run python scripts/convert_az_seq_series.py \\
        --in models/az_seq_champion.pt --out models/az_seq_series.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from nn.az_nn_check import CASES
from nn.dataset import encode_exact_np, encode_upper_bound_np
from nn.model_az_seq import Big2NetSeqAZ, load_token_feats


def _infer_and_build(old, sd: dict) -> Big2NetSeqAZ:
    n_layers, n_heads, _head_dim, d_model, seq_cap = old.config()
    d_ff = sd["blocks.0.ff1.weight"].shape[0]
    return Big2NetSeqAZ(
        load_token_feats(),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        seq_cap=seq_cap,
    )


def _parity_check(old, new) -> None:
    old.eval()
    new.eval()
    worst = 0.0
    for prefix, path, hand, oppmax, osz, usz, otm in CASES:
        seq = prefix + path
        tokens = torch.tensor([seq + [0]], dtype=torch.long)
        h48 = torch.from_numpy(encode_exact_np(np.array([hand], np.int64)))
        o48 = torch.from_numpy(encode_upper_bound_np(np.array([oppmax], np.int64)))
        osz_t = torch.tensor([osz / 16.0])
        usz_t = torch.tensor([usz / 16.0])
        otm_t = torch.tensor([otm])
        z = torch.tensor([0.0])
        with torch.no_grad():
            Ho = old.forward_full(tokens)[:, len(seq)]
            vo, po, bo, qo = old.readout(Ho, h48, o48, osz_t, usz_t, otm_t)
            Hn = new.forward_full(tokens)[:, len(seq)]
            vn, pn, bn, qn = new.readout(Hn, h48, o48, osz_t, usz_t, otm_t, z, z)
        for a, b in ((vo, vn), (po, pn), (bo, bn), (qo, qn)):
            worst = max(worst, float((a - b).abs().max()))
    print(f"parity (zero series points) max|Δ| = {worst:.3e}")
    assert worst < 1e-5, "converted net diverges from original at zero series points"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old = torch.jit.load(args.inp, map_location="cpu")
    sd = dict(old.state_dict())

    model = _infer_and_build(old, sd)
    new_sd = model.state_dict()
    for k in new_sd:
        if k == "size_embed.weight":
            w = torch.zeros_like(new_sd[k])
            w[:, :3] = sd[k]  # old 3 columns; new columns 3,4 stay zero
            new_sd[k] = w
        elif k in sd:
            new_sd[k] = sd[k]
        else:
            raise KeyError(f"missing key in source checkpoint: {k}")
    model.load_state_dict(new_sd)

    _parity_check(old, model)

    scripted = torch.jit.script(model.eval())
    scripted.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
