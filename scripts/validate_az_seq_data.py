"""Validate an az seq-net parquet triple (games / player / opp).

Replays each game's move list and asserts the invariants the trainer relies
on: hist_idx strictly increasing per game and within bounds, history length
<= 64, stored hand sizes matching the replay, hand counts summing to our_size,
and the recomputed opp_max matching max_in_deck - hand - cum_discard.

    uv run python scripts/validate_az_seq_data.py \
        data/az_games_gen0.parquet data/az_player_gen0.parquet data/az_opp_gen0.parquet
"""

from __future__ import annotations

import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from nn.dataset_seq import MAX_HIST, SeqData  # noqa: E402


def main() -> None:
    games, player, opp = sys.argv[1:4]
    g = pq.read_table(games).to_pandas()
    lens = g["moves"].map(len)
    assert (lens <= MAX_HIST).all(), f"game longer than {MAX_HIST}"
    assert g["winner"].isin([0, 1]).all()
    assert g["first_player"].isin([0, 1]).all()

    for name, path in (("player", player), ("opp", opp)):
        df = pq.read_table(path).to_pandas()
        for gid, grp in df.groupby("game_id"):
            h = grp["hist_idx"].to_numpy()
            assert (np.diff(h) > 0).all(), f"{name} game {gid}: hist_idx not increasing"
        print(f"{name}: {len(df):,} rows, hist_idx monotone per game")

    # SeqData's loader runs the full replay cross-checks (sizes vs move list,
    # opp_max recompute) with validate=True — loading IS the deep check.
    data = SeqData([(games, player, opp)], validate=True)
    print(
        f"SeqData replay validation passed: {data.G:,} games, "
        f"{len(data.p['value']):,} player rows, {len(data.o['value']):,} opp rows, "
        f"max len {int(data.glen.max())}"
    )


if __name__ == "__main__":
    main()
