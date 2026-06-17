"""Render sample az_pi self-play games as a self-contained HTML page.

Each game shows every recorded decision: both exact hands (perfect info), the
trick faced, the move actually played (reconstructed from successive hand
states), the search's visit distribution ("thinking"), and optionally the raw
net value when --model is given.

Usage:
    uv run python scripts/sample_games_pi.py data/az_pi_gen9.parquet \
        --games 10 --model models/az_pi.pt --out samples/az_pi_gen9.html
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import sample_games as sg  # noqa: E402  (R, RM, _mtc, fmt_move, kPASS)

MTC = sg._mtc()


def decode_exact(vec: np.ndarray) -> list[int]:
    """Invert encode_exact: per rank r the RM[r] slots one-hot the count."""
    counts, off = [], 0
    for mx in sg.RM:
        sl = vec[off : off + mx]
        counts.append(int(sl.argmax()) + 1 if sl.max() > 0.5 else 0)
        off += mx
    return counts


def move_label(mid: int) -> str:
    if mid == sg.kPASS:
        return "PASS"
    return sg.fmt_move(MTC[mid][:13], False)


def hand_chips(counts: list[int], cls: str = "") -> str:
    chips = []
    for r in range(13):
        for _ in range(counts[r]):
            rank_cls = {"2": "r2", "A": "rA", "K": "rK"}.get(sg.R[r], "")
            chips.append(f'<span class="card {rank_cls}">{sg.R[r]}</span>')
    return f'<span class="hand {cls}">{"".join(chips) or "&mdash;"}</span>'


def match_move(before: list[int], after: list[int]) -> str:
    """Identify the move played between two hand states of the same player."""
    diff = [b - a for b, a in zip(before, after)]
    if all(d == 0 for d in diff):
        return "PASS"
    for mid in range(1, sg.LEGAL_MOVES_SIZE):
        if MTC[mid][:13] == diff:
            return move_label(mid)
    return "+".join(sg.R[r] * diff[r] for r in range(13) if diff[r] > 0)


def visit_bars(moves: list[int], counts: list[int], top: int = 5) -> str:
    total = sum(counts) or 1
    order = np.argsort(counts)[::-1][:top]
    rows = []
    for i in order:
        pct = 100.0 * counts[i] / total
        rows.append(
            f'<div class="vrow"><span class="vmove">{html.escape(move_label(moves[i]))}</span>'
            f'<span class="vbar"><span class="vfill" style="width:{pct:.0f}%"></span></span>'
            f'<span class="vn">{counts[i]} ({pct:.0f}%)</span></div>'
        )
    return "".join(rows)


CSS = """
body { background:#11151c; color:#d6dde8; font:14px/1.45 'SF Mono',Consolas,monospace; margin:24px; }
h1 { font-size:18px; color:#8ecdf7; }
.meta { color:#7a8699; margin-bottom:18px; }
details { margin:10px 0; border:1px solid #26303f; border-radius:8px; background:#161b24; }
summary { cursor:pointer; padding:10px 14px; font-weight:bold; color:#8ecdf7; }
summary .w0 { color:#7ee08a; } summary .w1 { color:#f2a366; }
table { border-collapse:collapse; width:100%; }
td, th { padding:6px 10px; border-top:1px solid #222b38; vertical-align:top; text-align:left; }
th { color:#7a8699; font-weight:normal; font-size:12px; }
.mover0 { color:#7ee08a; font-weight:bold; } .mover1 { color:#f2a366; font-weight:bold; }
.card { display:inline-block; min-width:13px; text-align:center; background:#222b38; border-radius:3px;
        margin:1px; padding:1px 3px; color:#d6dde8; }
.card.r2 { color:#ff7a7a; } .card.rA { color:#ffd479; } .card.rK { color:#b7a6ff; }
.hand.dim .card { opacity:.45; }
.play { color:#ffd479; font-weight:bold; white-space:nowrap; }
.vrow { display:flex; align-items:center; gap:6px; margin:1px 0; }
.vmove { width:90px; color:#9fb3cc; white-space:nowrap; overflow:hidden; }
.vbar { flex:0 0 120px; height:8px; background:#222b38; border-radius:4px; overflow:hidden; }
.vfill { display:block; height:100%; background:#3f7fbf; }
.vn { color:#7a8699; font-size:12px; }
.val { white-space:nowrap; }
.valbar { display:inline-block; width:60px; height:8px; background:#222b38; border-radius:4px;
          overflow:hidden; vertical-align:middle; margin-right:6px; }
.valfill { display:block; height:100%; background:linear-gradient(90deg,#bf4040,#7ee08a); }
"""


def net_values(df: pd.DataFrame, model_path: str, device: str) -> np.ndarray:
    import torch

    m = torch.jit.load(model_path, map_location=device)
    m.eval()
    enc = np.stack(df["enc"].to_numpy())
    hand = torch.tensor(enc[:, :48], dtype=torch.float32)
    opp = torch.tensor(enc[:, 48:96], dtype=torch.float32)
    trick = torch.tensor(enc[:, 96:144], dtype=torch.float32)
    series_net = any("pts_embed" in k for k in m.state_dict().keys())
    if series_net and "my_pts" in df.columns:  # score inputs (/50)
        s0 = torch.tensor(df["my_pts"].to_numpy(), dtype=torch.float32) / 50.0
        s1 = torch.tensor(df["opp_pts"].to_numpy(), dtype=torch.float32) / 50.0
    else:  # legacy score-blind net: hand sizes (/16)
        s0 = torch.tensor(df["opp_size"].to_numpy(), dtype=torch.float32) / 16.0
        s1 = torch.tensor(df["our_size"].to_numpy(), dtype=torch.float32) / 16.0
    with torch.no_grad():
        v, _ = m(
            hand.to(device),
            opp.to(device),
            trick.to(device),
            s0.to(device),
            s1.to(device),
        )
    return v.cpu().numpy().reshape(-1)


def render_game(gdf: pd.DataFrame, net_v: np.ndarray | None) -> str:
    rows = []
    recs = gdf.sort_values("turn_idx").to_dict("records")
    # Mover parity: recorded rows alternate by actual turn_idx parity per game
    # (seat 0 always leads). value==1 rows belong to the winner.
    winner = 0 if (recs[0]["value"] == 1.0) == (recs[0]["turn_idx"] % 2 == 0) else 1
    # Reconstruct played moves: diff successive hands of the same mover.
    hands = [decode_exact(np.asarray(r["enc"][:48])) for r in recs]
    opps = [decode_exact(np.asarray(r["enc"][48:96])) for r in recs]
    tricks = [decode_exact(np.asarray(r["enc"][96:144])) for r in recs]
    movers = [r["turn_idx"] % 2 for r in recs]

    # Both exact hands appear on every row (PI), so each player's hand is
    # known at every recorded ply — as mover or as opponent. A decision's
    # played move is the hand diff to the next ply where that player's hand
    # is visible, provided the player moved only once in between.
    timeline: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}
    for k, r in enumerate(recs):
        ply = r["turn_idx"]
        timeline[movers[k]][ply] = hands[k]
        timeline[1 - movers[k]][ply] = opps[k]

    for k, r in enumerate(recs):
        m, ply = movers[k], r["turn_idx"]
        played = "&hellip;"
        later = sorted(p for p in timeline[m] if p > ply)
        if later:
            j = later[0]
            # m moves at plies of parity m; exactly one such ply in [ply, j)?
            n_moves = sum(1 for p in range(ply, j) if p % 2 == m)
            if n_moves == 1:
                played = html.escape(match_move(hands[k], timeline[m][j]))
            else:
                played = (
                    html.escape(match_move(hands[k], timeline[m][j]))
                    + f' <span class="vn">(+{n_moves - 1} forced)</span>'
                )
        elif m == winner:
            # Last recorded decision of the winner: the remaining cards go out
            # over this + subsequent FORCED moves (not one combined move).
            rest = html.escape(match_move(hands[k], [0] * 13))
            played = (
                f"<span class='vn'>sheds rest:</span> {rest} "
                f"<span class='play'>wins (forced finish)</span>"
            )
        trick_s = (
            hand_chips(tricks[k]) if any(tricks[k]) else '<span class="vn">lead</span>'
        )

        def val_html(v: float, proven: bool = False) -> str:
            tag = ' <span class="play">proven</span>' if proven else ""
            return (
                f'<span class="val"><span class="valbar">'
                f'<span class="valfill" style="width:{v*100:.0f}%"></span></span>'
                f"{v:.2f}{tag}</span>"
            )

        val_s = val_html(float(net_v[r["_row"]])) if net_v is not None else ""
        sv_s = ""
        if (
            "root_value" in r
            and r["root_value"] is not None
            and not pd.isna(r["root_value"])
        ):
            sv = float(r["root_value"])
            sv_s = val_html(sv, proven=sv in (0.0, 1.0))
        rows.append(
            f'<tr><td>{r["turn_idx"]}</td>'
            f'<td class="mover{movers[k]}">P{movers[k]}</td>'
            f"<td>{hand_chips(hands[k])}</td>"
            f'<td>{hand_chips(opps[k], "dim")}</td>'
            f"<td>{trick_s}</td>"
            f'<td class="play">{played}</td>'
            f"<td>{visit_bars(list(r['visit_moves']), list(r['visit_counts']))}</td>"
            f"<td>{val_s}</td><td>{sv_s}</td></tr>"
        )
    gid = recs[0]["game_id"]
    score = ""
    if "my_pts" in recs[0]:
        # Seat-relative: recs[0]'s mover holds my_pts. Map back to seats.
        m0 = recs[0]["turn_idx"] % 2  # seat 0 leads; parity gives the mover
        a, b = recs[0]["my_pts"], recs[0]["opp_pts"]
        s0, s1 = (a, b) if m0 == 0 else (b, a)
        score = f" &mdash; series score {s0}&ndash;{s1}"
    head = (
        f'Game {gid} &mdash; <span class="w{winner}">P{winner} wins</span> '
        f"({len(recs)} recorded decisions){score}"
    )
    cols = (
        "<tr><th>ply</th><th>mover</th><th>mover hand</th><th>opp hand</th>"
        "<th>trick</th><th>played</th><th>search visits</th><th>net value</th>"
        "<th>search value</th></tr>"
    )
    return (
        f"<details{' open' if gid == 0 else ''}><summary>{head}</summary>"
        f"<table>{cols}{''.join(rows)}</table></details>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    gids = sorted(df["game_id"].unique())[: args.games]
    df = df[df["game_id"].isin(gids)].reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    net_v = net_values(df, args.model, args.device) if args.model else None

    body = "".join(render_game(df[df["game_id"] == g], net_v) for g in gids)
    title = f"az_pi sample games &mdash; {Path(args.parquet).name}"
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}</style></head><body>"
        f"<h1>{title}</h1>"
        f"<div class='meta'>{len(gids)} games &middot; perfect information "
        f"&middot; visit bars = MCTS thinking at each recorded decision "
        f"(forced moves are not recorded)</div>{body}</body></html>"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({len(gids)} games, {len(df)} decisions)")


if __name__ == "__main__":
    main()
