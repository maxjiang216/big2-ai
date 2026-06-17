"""Probe: how much do net outputs move when ONLY the series state changes?

Take real positions from a generation's parquet, hold every other input fixed,
sweep (my_pts, opp_pts) over a grid, and measure how far each head's output
moves vs the (0,0) baseline:

  value    — mean |Δvalue| over player rows
  policy   — mean KL(policy@state || policy@(0,0)) over legal moves, player rows
  behavior — mean KL over opp rows
  q_a      — mean |Δqa| at played slot, opp rows

If these are ~0 the series objective is inert: the net plays one fixed policy
regardless of score. A converted (zero-pts-column) champion is the control —
it MUST read exactly 0.0 everywhere.

Usage:
  uv run python -m analysis.az_series_sensitivity --model models/az_seq_gen5.pt \
      --games data/az_games_gen5.parquet --player data/az_player_gen5.parquet \
      --opp data/az_opp_gen5.parquet --batch-games 256
"""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F

from nn.dataset_seq import make_seq_split
from nn.model_az import load_compose_matrix
from nn.train_az import _device, masked_log_softmax
from nn.train_az_seq import _gather_h


def _kl_rows(logp_a, logp_b):
    # KL(a||b) per row over the masked simplex; logp_* are log-probs [n,K]
    pa = logp_a.exp()
    return (pa * (logp_a - logp_b)).sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--player", nargs="+", required=True)
    p.add_argument("--opp", nargs="+", required=True)
    p.add_argument("--batch-games", type=int, default=256)
    p.add_argument("--device", default="auto")
    cfg = p.parse_args()
    dev = _device(cfg.device)

    model = torch.jit.load(cfg.model, map_location=dev).eval()
    C = load_compose_matrix().to(dev)

    triples = list(zip(cfg.games, cfg.player, cfg.opp))
    tl, _ = make_seq_split(
        triples,
        cfg.batch_games,
        dev,
        val_frac=0.0,
        seed=0,
        validate=False,
        series_v=None,
    )
    batch = next(iter(tl))

    with torch.no_grad():
        H = model.forward_full(batch["tokens"])
        hp = _gather_h(H, batch["p_local"], batch["p_hist"])
        ho = _gather_h(H, batch["o_local"], batch["o_hist"])
        n_p, n_o = hp.shape[0], ho.shape[0]
        h = torch.cat([hp, ho])
        hand = torch.cat([batch["p_hand"], batch["o_hand"]])
        oppm = torch.cat([batch["p_opp"], batch["o_opp"]])
        osz = torch.cat([batch["p_osz"], batch["o_osz"]])
        usz = torch.cat([batch["p_usz"], batch["o_usz"]])
        otm = torch.cat([torch.ones(n_p, device=dev), torch.zeros(n_o, device=dev)])
        pmask = batch["p_mask"]
        omask = batch["o_mask"]
        otgt = batch["o_target"]

        def run(a, b):
            mpts = torch.full((n_p + n_o,), a / 50.0, device=dev)
            opts = torch.full((n_p + n_o,), b / 50.0, device=dev)
            val, pol, beh, qa = model.readout(h, hand, oppm, osz, usz, otm, mpts, opts)
            logp = masked_log_softmax(pol[:n_p] @ C.t(), pmask)
            logb = masked_log_softmax(beh[n_p:], omask)
            q = qa[n_p:].gather(1, otgt.unsqueeze(1)).squeeze(1)
            return val[:n_p], logp, logb, q

        v0, lp0, lb0, q0 = run(0, 0)
        print(f"probed {n_p} player rows, {n_o} opp rows from {cfg.model}\n")
        grid = [
            (0, 0),
            (10, 0),
            (25, 0),
            (40, 0),
            (49, 0),
            (0, 40),
            (40, 40),
            (45, 5),
            (5, 45),
            (49, 1),
            (1, 49),
        ]
        hdr = f"{'(a,b)':>9}  {'|Δvalue|':>9}  {'policyKL':>9}  {'behavKL':>9}  {'|Δqa|':>9}  {'meanV':>7}"
        print(hdr)
        print("-" * len(hdr))
        for a, b in grid:
            v, lp, lb, q = run(a, b)
            dv = (v - v0).abs().mean().item()
            pkl = _kl_rows(lp, lp0).mean().item()
            bkl = _kl_rows(lb, lb0).mean().item()
            dq = (q - q0).abs().mean().item()
            print(
                f"{f'({a},{b})':>9}  {dv:9.5f}  {pkl:9.5f}  {bkl:9.5f}  {dq:9.5f}  {v.mean().item():7.4f}"
            )
        # full-range value swing as a ceiling reference
        vlo = run(0, 49)[0].mean().item()
        vhi = run(49, 0)[0].mean().item()
        print(
            f"\nmean value (0,49) -> (49,0): {vlo:.4f} -> {vhi:.4f}  (swing {vhi-vlo:+.4f})"
        )


if __name__ == "__main__":
    main()
