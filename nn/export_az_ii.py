"""Export a trained Big2NetII checkpoint to TorchScript for the C++ az_ii player.

    uv run python -m nn.export_az_ii --ckpt models/az_ii_gen1.pt \\
        --out models/az_ii_gen1.ts.pt

The scripted module exposes forward(tokens, hist_idx, hand, oppmax, osz, usz,
otm, mpts, opts) -> (value, policy, behavior). AR / bomb / outcome heads are
training-only (jit-ignored) and not part of the scripted surface.
"""

from __future__ import annotations

import argparse

import torch

from nn.model_az_ii import Big2NetII
from nn.model_az_seq import load_token_feats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    cfg = p.parse_args()

    ck = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    model = Big2NetII(
        load_token_feats(),
        d_model=arch["d_model"], n_layers=arch["n_layers"],
        n_heads=arch["n_heads"], d_ff=arch["d_ff"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(cfg.out)
    print(f"✓ scripted {cfg.ckpt} -> {cfg.out}")


if __name__ == "__main__":
    main()
