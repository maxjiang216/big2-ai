#!/usr/bin/env python3
"""
Analyze Parquet exports from bin/generate_data (game_*.parquet + turn_*.parquet).

Produces a multi-panel report (PNG) and stdout summaries. Full move histories for
anomaly games are produced by generate_data --samples-md (not by this script).

Example:
  python analysis/analysis.py \\
    --game-parquet data/foo_game.parquet --turn-parquet data/foo_turn.parquet \\
    --output-dir analysis/
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_parquet(game_file: Path | None, turn_file: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame()
    turns = pd.DataFrame()
    if game_file is not None and game_file.is_file():
        games = pd.read_parquet(game_file)
        print(f"Loaded game features: {game_file}  shape={games.shape}")
    else:
        print(f"Game parquet not found or not given: {game_file}")
    if turn_file is not None and turn_file.is_file():
        turns = pd.read_parquet(turn_file)
        print(f"Loaded turn features: {turn_file}  shape={turns.shape}")
    else:
        print(f"Turn parquet not found or not given: {turn_file}")
    return games, turns


def prepare_games_df(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games
    df = games.copy()
    if "game_index" in df.columns:
        df.insert(0, "game_id", df["game_index"].astype(int))
    else:
        df.insert(0, "game_id", np.arange(len(df), dtype=int))
    return df


def format_game_sample(row: pd.Series) -> str:
    parts: list[str] = [f"game_id={int(row['game_id'])}"]
    if "length" in row.index and pd.notna(row["length"]):
        parts.append(f"length={int(row['length'])}")
    if "outcome" in row.index and pd.notna(row["outcome"]):
        parts.append(f"winner=P{int(row['outcome'])}")
    if "start_legal_moves" in row.index and pd.notna(row["start_legal_moves"]):
        parts.append(f"start_legal_moves={int(row['start_legal_moves'])}")
    return " | ".join(parts)


def print_anomaly_samples(games: pd.DataFrame, samples_md_hint: str | None) -> None:
    if games.empty:
        return
    print("\n--- Anomalous sample games ---")
    if "length" in games.columns:
        L = games["length"]
        print(f"  Longest game:   {format_game_sample(games.loc[L.idxmax()])}")
        print(f"  Shortest game:  {format_game_sample(games.loc[L.idxmin()])}")
    if "start_legal_moves" in games.columns:
        S = games["start_legal_moves"]
        print(f"  Most start legal moves:   {format_game_sample(games.loc[S.idxmax()])}")
        print(f"  Fewest start legal moves: {format_game_sample(games.loc[S.idxmin()])}")
    if samples_md_hint:
        print(f"\n  (Full hands + histories: {samples_md_hint})")


def print_text_summary(
    games: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    samples_md_hint: str | None,
) -> None:
    if games.empty:
        print("No game rows.")
        return

    print("\n--- Game-level (per-game) ---")
    for col in sorted(c for c in games.columns if c not in ("game_id", "game_index")):
        vals = games[col].tolist()
        if col == "outcome":
            c = Counter(vals)
            parts = ", ".join(f"P{w} wins: {c[w]}" for w in sorted(c))
            print(f"  {col}: {parts}")
            continue
        arr = np.array(vals, dtype=float)
        print(
            f"  {col}: mean={arr.mean():.4f}  "
            f"stdev={arr.std(ddof=1) if len(arr) > 1 else 0:.4f}  "
            f"min={arr.min():.0f}  max={arr.max():.0f}"
        )

    print_anomaly_samples(games, samples_md_hint)

    if not turns.empty and "player_hand_size" in turns.columns:
        ph = turns["player_hand_size"]
        print("\n--- Turn-level (all turns x perspectives) ---")
        print(
            f"  player_hand_size: mean={ph.mean():.4f}  "
            f"stdev={ph.std(ddof=1) if len(ph) > 1 else 0:.4f}  "
            f"min={ph.min()}  max={ph.max()}"
        )
        if "possible_moves" in turns.columns:
            pm = turns["possible_moves"]
            print(
                f"  possible_moves: mean={pm.mean():.4f}  "
                f"stdev={pm.std(ddof=1) if len(pm) > 1 else 0:.4f}  "
                f"min={pm.min()}  max={pm.max()}"
            )


def build_report_figure(
    games: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    title_suffix: str,
    samples_md_hint: str | None,
) -> plt.Figure:
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.42], hspace=0.32, wspace=0.28)
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
    ax_note = fig.add_subplot(gs[2, :])
    fig.suptitle(
        f"Self-play game statistics{title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0, 0]
    if not games.empty and "outcome" in games.columns:
        vc = games["outcome"].value_counts().sort_index()
        labels = [f"P{int(i)} wins" for i in vc.index]
        bars = ax.bar(labels, vc.values, color=["steelblue", "coral"], alpha=0.85)
        total = len(games)
        for bar, count in zip(bars, vc.values):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + total * 0.01,
                f"{int(count)}\n({100.0 * count / total:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylabel("Games")
        ax.set_title("Win rate (P0 vs P1)")
    else:
        ax.set_title("Win rate")
        ax.text(0.5, 0.5, "No outcome data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[0, 1]
    if not games.empty and "length" in games.columns:
        L = games["length"]
        ax.hist(
            L,
            bins=min(40, max(10, int(L.max() - L.min() + 1))),
            color="seagreen",
            alpha=0.75,
            edgecolor="black",
        )
        ax.axvline(L.mean(), color="red", linestyle="--", label=f"mean={L.mean():.2f}")
        if len(L) > 0 and int(L.min()) != int(L.max()):
            ax.axvline(
                games.loc[L.idxmax(), "length"],
                color="darkviolet",
                linestyle="-",
                linewidth=1.2,
                label="longest sample",
            )
            ax.axvline(
                games.loc[L.idxmin(), "length"],
                color="orange",
                linestyle="-",
                linewidth=1.2,
                label="shortest sample",
            )
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Turns per game")
        ax.set_ylabel("Count")
        ax.set_title(f"Game length (n={len(L)})")
    else:
        ax.set_title("Game length")
        ax.text(0.5, 0.5, "No length data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[0, 2]
    if not games.empty and "start_legal_moves" in games.columns:
        s = games["start_legal_moves"]
        ax.hist(
            s,
            bins=range(int(s.min()), int(s.max()) + 2),
            color="mediumpurple",
            alpha=0.75,
            edgecolor="black",
        )
        ax.set_xlabel("Legal moves (opening player, turn 0)")
        ax.set_ylabel("Games")
        ax.set_title("Starting legal moves")
    else:
        ax.set_title("Starting legal moves")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 0]
    if not turns.empty and "turn_idx" in turns.columns and "player_hand_size" in turns.columns:
        g = turns.groupby("turn_idx")["player_hand_size"].mean()
        ax.plot(g.index, g.values, color="darkgreen", linewidth=1.5)
        ax.fill_between(g.index, g.values, alpha=0.2, color="darkgreen")
        ax.set_xlabel("Turn index (within game)")
        ax.set_ylabel("Mean hand size")
        ax.set_title("Mean hand size over game progress")
    else:
        ax.set_title("Hand size over turns")
        ax.text(0.5, 0.5, "No turn data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    if not turns.empty and "possible_moves" in turns.columns:
        pm = turns["possible_moves"]
        hi = min(80, int(pm.max()) + 2)
        lo = max(0, int(pm.min()))
        bins = min(50, hi - lo + 1) if hi > lo else 10
        ax.hist(
            pm,
            bins=bins,
            range=(lo, hi) if hi > lo else None,
            color="teal",
            alpha=0.75,
            edgecolor="black",
        )
        ax.set_xlabel("Possible moves")
        ax.set_ylabel("Turn-perspectives")
        ax.set_title("Possible moves per position")
    else:
        ax.set_title("Possible moves")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 2]
    if not turns.empty and "last_move_card_count" in turns.columns:
        c = turns["last_move_card_count"]
        ax.hist(
            c,
            bins=range(int(c.min()), int(c.max()) + 2),
            color="goldenrod",
            alpha=0.8,
            edgecolor="black",
        )
        ax.set_xlabel("Cards in last combo (0 = pass)")
        ax.set_ylabel("Turn-perspectives")
        ax.set_title("Last move size")
    else:
        ax.set_title("Last move card count")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    ax_note.axis("off")
    note_lines = ["Anomalous sample games", ""]
    if not games.empty and "length" in games.columns and len(games) > 0:
        L = games["length"]
        note_lines.append(f"Longest:   {format_game_sample(games.loc[L.idxmax()])}")
        note_lines.append(f"Shortest:  {format_game_sample(games.loc[L.idxmin()])}")
    if not games.empty and "start_legal_moves" in games.columns and len(games) > 0:
        S = games["start_legal_moves"]
        note_lines.append(f"Most start legal moves:   {format_game_sample(games.loc[S.idxmax()])}")
        note_lines.append(f"Fewest start legal moves: {format_game_sample(games.loc[S.idxmin()])}")
    if samples_md_hint:
        note_lines.append("")
        note_lines.append(f"Full histories: {samples_md_hint}")

    ax_note.text(
        0.02,
        0.98,
        "\n".join(note_lines),
        transform=ax_note.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f5f5f5", "edgecolor": "#888888"},
    )

    fig.subplots_adjust(top=0.93, bottom=0.06)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--game-parquet",
        type=Path,
        default=None,
        help="Path to *_game.parquet (default: cwd game_features.parquet)",
    )
    ap.add_argument(
        "--turn-parquet",
        type=Path,
        default=None,
        help="Path to *_turn.parquet (default: cwd turn_features.parquet)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for report PNG and optional CSV exports",
    )
    ap.add_argument("--report-name", type=str, default="report.png", help="Report image filename")
    ap.add_argument(
        "--samples-md-hint",
        type=str,
        default=None,
        help="Shown in report/footer (path to sample_games.md from generate_data)",
    )
    ap.add_argument("--title", type=str, default="", help="Extra title suffix")
    args = ap.parse_args()

    game_path = args.game_parquet or Path("game_features.parquet")
    turn_path = args.turn_parquet or Path("turn_features.parquet")

    if not game_path.exists() and not turn_path.exists():
        print(f"Error: neither {game_path} nor {turn_path} found.", file=sys.stderr)
        sys.exit(1)

    games_raw, turns = load_parquet(
        game_path if game_path.exists() else None,
        turn_path if turn_path.exists() else None,
    )
    games = prepare_games_df(games_raw)

    n_games = len(games) if not games.empty else 0
    title_suffix = args.title or (f" (n={n_games} games)" if n_games else "")

    print_text_summary(games, turns, samples_md_hint=args.samples_md_hint)

    fig = build_report_figure(
        games,
        turns,
        title_suffix=title_suffix,
        samples_md_hint=args.samples_md_hint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.report_name
    fig.savefig(report_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote report: {report_path}")

    if not games.empty:
        csv_g = args.output_dir / "game_features_summary.csv"
        games.to_csv(csv_g, index=False)
        print(f"Wrote {csv_g}")
    if not turns.empty:
        csv_t = args.output_dir / "turn_features_summary.csv"
        turns.to_csv(csv_t, index=False)
        print(f"Wrote {csv_t}")


if __name__ == "__main__":
    main()
