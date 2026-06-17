"""Estimate the series Markov chain from single-game outcomes.

A series state is (a, b) = (points of the player leading the next game, points
of the follower). The leader is the previous game's winner. After a game the
winner scores p = points_for_cards(loser_cards); if winner_total >= 50 the
series ends, else the winner leads the next game.

Given empirical per-state outcome distributions P(leader_won, p | a, b)
(shrunk toward the pooled global distribution), we solve

  V(a, b) = Σ_p P(win, p) · [a+p>=50 ? 1 : V(a+p, b)]
          + Σ_p P(lose, p) · [b+p>=50 ? 0 : 1 - V(b+p, a)]

by backward induction on a+b (scores strictly increase -> DAG), and the natural
state-visit frequencies by a forward pass from (0, 0).

Sources:
  --games    az_games_genN.parquet (game_id, first_player, winner, moves[,
             pts0, pts1]); loser_cards derived by replaying the move tokens.
  --outcomes eval_az_series --dump-outcomes CSVs
             (a, b, first_player, winner, loser_cards[, model_seat]).

Output: a 50x50 CSV `a,b,v,natural_freq` consumable by C++ (load_series_table)
and Python training (sample weights).

Usage:
  uv run python analysis/series_markov.py --games data/az_games_gen3.parquet \
      --out data/series_v_gen3.csv [--shrink-k 100] [--bootstrap-global]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq

TARGET = 50
PVALS = list(range(1, 13)) + [20, 30, 40, 50]  # 16 distinct point awards
NP = len(PVALS)
NOUT = 2 * NP  # outcome cells: [win x PVALS, lose x PVALS]
_P_TO_IDX = {p: i for i, p in enumerate(PVALS)}


def points_for_cards(cards: np.ndarray) -> np.ndarray:
    """Vectorised twin of src/core/series.cpp:points_for_cards."""
    cards = np.asarray(cards, dtype=np.int64)
    out = np.where(cards <= 12, cards, 10 * (cards - 11))
    return np.clip(out, 0, None)


def _move_cards(path: str = "nn/az_token_cards.json") -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    return np.asarray(d["cards"], dtype=np.int64)  # [NUM_MOVES, 13]


def _loser_cards_from_moves(first_player, winner, mflat, moff, mc) -> np.ndarray:
    """16 - cards played by the loser, replaying the token sequence per game."""
    totals = mc.sum(-1)  # cards per move id
    G = len(first_player)
    loser_cards = np.empty(G, dtype=np.int64)
    for g in range(G):
        seq = mflat[moff[g] : moff[g + 1]]
        loser = 1 - winner[g]
        # seat of move k = (first_player + k) % 2
        seats = (first_player[g] + np.arange(len(seq))) % 2
        played = totals[seq][seats == loser].sum()
        loser_cards[g] = 16 - played
    return loser_cards


def _outcome_idx(leader_won: np.ndarray, p: np.ndarray) -> np.ndarray:
    pidx = np.array([_P_TO_IDX[int(x)] for x in p], dtype=np.int64)
    return np.where(leader_won, pidx, NP + pidx)


def _records_from_games(path: str, mc: np.ndarray):
    """(a, b, outcome_idx) per game. a,b are -1 if pts columns are absent."""
    t = pq.read_table(path).combine_chunks()
    cols = set(t.schema.names)
    first = t.column("first_player").chunk(0).to_numpy()
    winner = t.column("winner").chunk(0).to_numpy()
    mcol = t.column("moves").chunk(0)
    moff = mcol.offsets.to_numpy().astype(np.int64)
    mflat = mcol.values.to_numpy().astype(np.int64)
    if "loser_cards" in cols:
        loser_cards = t.column("loser_cards").chunk(0).to_numpy()
    else:
        loser_cards = _loser_cards_from_moves(first, winner, mflat, moff, mc)
    leader_won = winner == first
    p = points_for_cards(loser_cards)
    if "pts0" in cols and "pts1" in cols:
        pts = np.stack(
            [
                t.column("pts0").chunk(0).to_numpy(),
                t.column("pts1").chunk(0).to_numpy(),
            ],
            axis=1,
        )
        a = pts[np.arange(len(first)), first]
        b = pts[np.arange(len(first)), 1 - first]
    else:
        a = np.full(len(first), -1, dtype=np.int64)
        b = np.full(len(first), -1, dtype=np.int64)
    return a, b, _outcome_idx(leader_won, p)


def _records_from_outcomes(path: str):
    import csv

    a, b, lw, lc = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            a.append(int(row["a"]))
            b.append(int(row["b"]))
            lw.append(int(row["winner"]) == int(row["first_player"]))
            lc.append(int(row["loser_cards"]))
    p = points_for_cards(np.asarray(lc))
    return (
        np.asarray(a, dtype=np.int64),
        np.asarray(b, dtype=np.int64),
        _outcome_idx(np.asarray(lw), p),
    )


def estimate(a, b, oidx, shrink_k: float, bootstrap_global: bool):
    counts = np.zeros((TARGET, TARGET, NOUT), dtype=np.float64)
    glob = np.zeros(NOUT, dtype=np.float64)
    for ai, bi, oi in zip(a, b, oidx):
        glob[oi] += 1.0
        if 0 <= ai < TARGET and 0 <= bi < TARGET:
            counts[ai, bi, oi] += 1.0
    if glob.sum() == 0:
        raise SystemExit("no usable game records")
    pg = glob / glob.sum()

    P = np.zeros((TARGET, TARGET, NOUT), dtype=np.float64)
    for ai in range(TARGET):
        for bi in range(TARGET):
            ns = counts[ai, bi].sum()
            if bootstrap_global or ns == 0:
                P[ai, bi] = pg
            else:
                P[ai, bi] = (counts[ai, bi] + shrink_k * pg) / (ns + shrink_k)

    pvals = np.asarray(PVALS, dtype=np.int64)
    V = np.zeros((TARGET, TARGET), dtype=np.float64)
    # Backward induction on a+b descending.
    for s in range(2 * TARGET - 2, -1, -1):
        for ai in range(max(0, s - TARGET + 1), min(s, TARGET - 1) + 1):
            bi = s - ai
            if bi < 0 or bi >= TARGET:
                continue
            pw, pl = P[ai, bi, :NP], P[ai, bi, NP:]
            v = 0.0
            for k, p in enumerate(pvals):
                na = ai + p
                v += pw[k] * (1.0 if na >= TARGET else V[na, bi])
                nb = bi + p
                v += pl[k] * (0.0 if nb >= TARGET else 1.0 - V[nb, ai])
            V[ai, bi] = v

    # Forward pass for natural visit frequencies, a+b ascending.
    natural = np.zeros((TARGET, TARGET), dtype=np.float64)
    natural[0, 0] = 1.0
    for s in range(0, 2 * TARGET - 1):
        for ai in range(max(0, s - TARGET + 1), min(s, TARGET - 1) + 1):
            bi = s - ai
            if bi < 0 or bi >= TARGET:
                continue
            f = natural[ai, bi]
            if f == 0.0:
                continue
            pw, pl = P[ai, bi, :NP], P[ai, bi, NP:]
            for k, p in enumerate(pvals):
                na = ai + p
                if na < TARGET:
                    natural[na, bi] += f * pw[k]
                nb = bi + p
                if nb < TARGET:
                    natural[nb, ai] += f * pl[k]
    return V, natural, pg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=[], help="az_games parquet files")
    ap.add_argument("--outcomes", nargs="*", default=[], help="eval outcome CSVs")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--shrink-k", type=float, default=100.0)
    ap.add_argument(
        "--bootstrap-global",
        action="store_true",
        help="use one pooled outcome distribution for every state (gen0 seed)",
    )
    ap.add_argument("--gen", default="?", help="provenance tag for the CSV header")
    args = ap.parse_args()

    if not args.games and not args.outcomes:
        raise SystemExit("provide --games and/or --outcomes")

    mc = _move_cards()
    a_all, b_all, o_all = [], [], []
    for gp in args.games:
        a, b, o = _records_from_games(gp, mc)
        a_all.append(a)
        b_all.append(b)
        o_all.append(o)
    for op in args.outcomes:
        a, b, o = _records_from_outcomes(op)
        a_all.append(a)
        b_all.append(b)
        o_all.append(o)
    a = np.concatenate(a_all)
    b = np.concatenate(b_all)
    oidx = np.concatenate(o_all)
    n_state = int(((a >= 0) & (b >= 0)).sum())

    V, natural, pg = estimate(a, b, oidx, args.shrink_k, args.bootstrap_global)

    with open(args.out, "w") as f:
        f.write(
            f"# gen={args.gen} games={len(a)} state_labeled={n_state} "
            f"shrink_k={args.shrink_k} bootstrap_global={args.bootstrap_global}\n"
        )
        f.write("a,b,v,natural_freq\n")
        for ai in range(TARGET):
            for bi in range(TARGET):
                f.write(f"{ai},{bi},{V[ai, bi]:.6f},{natural[ai, bi]:.6f}\n")

    exp_games = float(natural.sum())
    # Monotonicity spot-checks (should hold for a sane chain).
    mono_a = bool(np.all(np.diff(V[:, 0]) >= -1e-6))
    mono_b = bool(np.all(np.diff(V[0, :]) <= 1e-6))
    print(f"wrote {args.out}")
    print(f"  V(0,0)              = {V[0, 0]:.4f}")
    print(f"  expected games/series = {exp_games:.3f}")
    print(f"  V increasing in a (b=0): {mono_a}   decreasing in b (a=0): {mono_b}")
    print(f"  pooled P(leader win)  = {pg[:NP].sum():.4f}")


if __name__ == "__main__":
    main()
