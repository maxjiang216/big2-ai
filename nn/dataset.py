"""Dataset for Big 2 DNN training.

Loads the Parquet produced by `bin/generate_data` with the dnn_gen0.json
config, computes trick-winner labels, and encodes rank counts as 48-bit
tensors.

Column layout expected from Parquet (turn-level table):
  game_index, turn_idx, perspective   - always present (added by export)
  turn_outcome                        - 0/1 game winner for this perspective
  next_player                         - 1 if this perspective is the mover
  tb_case                             - -1 for normal play
  n_3 .. n_2                          - 13 cols, player hand rank counts (before move)
  move_3 .. move_2                    - 13 cols, move rank counts
  opp_max_3 .. opp_max_2             - 13 cols, opponent max-possible rank counts
  hint                                - int (divide by 10000 → float)
  opp_cannot_respond                  - 0/1 flag
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Rank name suffix ordering (matches C++ RankFeature naming).
RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k", "a", "2"]
RANK_MAX_COUNTS = [4] * 11 + [3, 1]  # max cards per rank index
ENCODING_DIM = sum(RANK_MAX_COUNTS)   # 48


def _rank_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{r}" for r in RANK_NAMES]


HAND_COLS = _rank_cols("n")
MOVE_COLS = _rank_cols("move")
OPP_MAX_COLS = _rank_cols("opp_max")


def encode_exact_np(rank_counts: np.ndarray) -> np.ndarray:
    """Exact-count one-hot encoding.

    rank_counts: int array [N, 13]
    Returns float32 array [N, 48].
    """
    N = rank_counts.shape[0]
    out = np.zeros((N, ENCODING_DIM), dtype=np.float32)
    offset = 0
    for r, max_k in enumerate(RANK_MAX_COUNTS):
        c = rank_counts[:, r]
        for k in range(1, max_k + 1):
            out[:, offset + k - 1] = (c == k).astype(np.float32)
        offset += max_k
    return out


def encode_upper_bound_np(rank_counts: np.ndarray) -> np.ndarray:
    """Thermometer (≥k) encoding for opponent max-possible.

    rank_counts: int array [N, 13]
    Returns float32 array [N, 48].
    """
    N = rank_counts.shape[0]
    out = np.zeros((N, ENCODING_DIM), dtype=np.float32)
    offset = 0
    for r, max_k in enumerate(RANK_MAX_COUNTS):
        c = np.clip(rank_counts[:, r], 0, max_k)
        for k in range(1, max_k + 1):
            out[:, offset + k - 1] = (c >= k).astype(np.float32)
        offset += max_k
    return out


def compute_trick_winners(df: pd.DataFrame) -> pd.Series:
    """Label each mover row with whether that player won the current trick.

    A trick ends when a player passes; the last non-passer wins.  The game's
    final trick (ending with a player emptying their hand) is won by the player
    who played last.

    df must be the full turn table (both perspectives), sorted by
    (game_index, turn_idx).  Returns a Series aligned to df.index,
    with NaN for observer rows (next_player == 0).
    """
    move_sum = df[MOVE_COLS].sum(axis=1)
    is_pass = (move_sum == 0)

    result = pd.Series(np.nan, index=df.index, dtype=float)

    for game_id, gdf in df.groupby("game_index", sort=False):
        # Only mover rows needed for trick labelling; keep sort order.
        mover_idx = gdf.index[gdf["next_player"] == 1]
        if len(mover_idx) == 0:
            continue
        mdf = gdf.loc[mover_idx].sort_values("turn_idx")
        perspectives = mdf["perspective"].to_numpy()
        passes = is_pass.loc[mdf.index].to_numpy()
        n = len(mdf)

        trick_winner = np.full(n, -1, dtype=np.int8)
        trick_start = 0

        for i in range(n):
            if passes[i]:
                # Player perspectives[i] passed → last non-passer wins trick.
                # The non-passer is the other player (passes don't start tricks).
                winner = perspectives[i - 1] if i > trick_start else -1
                for j in range(trick_start, i + 1):
                    trick_winner[j] = int(perspectives[j] == winner)
                trick_start = i + 1

        # Final trick: game ended with a non-pass play.
        if trick_start < n:
            last_mover = perspectives[n - 1]
            for j in range(trick_start, n):
                trick_winner[j] = int(perspectives[j] == last_mover)

        result.loc[mdf.index] = trick_winner.astype(float)

    return result


class Big2Dataset(Dataset):
    """PyTorch Dataset for Big 2 DNN training.

    Filters to:
      - Mover rows (next_player == 1)
      - Non-tablebase turns (tb_case == -1)

    Each item is a tuple:
      hand      float32[48]   player hand AFTER playing move M (exact one-hot)
      opp       float32[48]   opponent max-possible hand (thermometer one-hot)
      move      float32[48]   move M (exact one-hot, all-zero for pass)
      hint      float32       response probability hint (from hint / 10000)
      flag      bool          opponent cannot respond
      pass_     bool          move is pass
      y_trick   float32       1 if player won this trick
      y_win     float32       1 if player won the game
    """

    def __init__(self, parquet_path: str,
                 val_game_ids: Optional[set] = None,
                 train: bool = True) -> None:
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(["game_index", "turn_idx"]).reset_index(drop=True)

        # Compute trick winners on full df before filtering.
        trick_labels = compute_trick_winners(df)
        df["p_win_trick"] = trick_labels

        # Filter to mover, non-tablebase rows.
        mask = (df["next_player"] == 1) & (df["tb_case"] == -1)
        df = df[mask].copy()

        # Train / val split by game_index.
        if val_game_ids is not None:
            if train:
                df = df[~df["game_index"].isin(val_game_ids)]
            else:
                df = df[df["game_index"].isin(val_game_ids)]

        df = df.dropna(subset=["p_win_trick"]).reset_index(drop=True)

        hand_before = df[HAND_COLS].to_numpy(dtype=np.int32)
        move_cards = df[MOVE_COLS].to_numpy(dtype=np.int32)
        opp_max = df[OPP_MAX_COLS].to_numpy(dtype=np.int32)

        hand_after = hand_before - move_cards  # player's hand after playing M
        hand_after = np.clip(hand_after, 0, None)

        # 48-bit encodings
        self.hand_enc = torch.from_numpy(encode_exact_np(hand_after))
        self.opp_enc = torch.from_numpy(encode_upper_bound_np(opp_max))
        self.move_enc = torch.from_numpy(encode_exact_np(move_cards))

        self.hint = torch.tensor(
            df["hint"].to_numpy(dtype=np.float32) / 10000.0, dtype=torch.float32)
        self.flag = torch.tensor(
            df["opp_cannot_respond"].to_numpy(dtype=bool), dtype=torch.bool)
        self.pass_ = torch.tensor(
            (move_cards.sum(axis=1) == 0), dtype=torch.bool)

        self.y_trick = torch.tensor(
            df["p_win_trick"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y_win = torch.tensor(
            df["turn_outcome"].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_win)

    def __getitem__(self, idx: int):
        return (
            self.hand_enc[idx],
            self.opp_enc[idx],
            self.move_enc[idx],
            self.hint[idx],
            self.flag[idx],
            self.pass_[idx],
            self.y_trick[idx],
            self.y_win[idx],
        )


def decode_handbits_np(at1: np.ndarray, at2: np.ndarray,
                        at3: np.ndarray, at4: np.ndarray) -> np.ndarray:
    """Decode HandBits thermometer bitfields → rank counts [N, 13].

    Bit r in atK is set iff count[r] >= K (thermometer encoding).
    """
    counts = np.zeros((len(at1), 13), dtype=np.int32)
    for r in range(13):
        bit = np.int32(1 << r)
        counts[:, r] = (
            ((at1 & bit) != 0).astype(np.int32) +
            ((at2 & bit) != 0).astype(np.int32) +
            ((at3 & bit) != 0).astype(np.int32) +
            ((at4 & bit) != 0).astype(np.int32)
        )
    return counts


def compute_trick_winners_selfplay(df: pd.DataFrame) -> pd.Series:
    """Trick winner labels for nn_selfplay format (all rows are movers).

    Uses game_id, turn_idx, current_player, pass_ columns.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, gdf in df.groupby("game_id", sort=False):
        gdf = gdf.sort_values("turn_idx")
        players = gdf["current_player"].to_numpy()
        passes = gdf["pass_"].to_numpy()
        n = len(gdf)
        trick_winner = np.full(n, -1, dtype=np.int8)
        trick_start = 0
        for i in range(n):
            if passes[i]:
                winner = players[i - 1] if i > trick_start else -1
                for j in range(trick_start, i + 1):
                    trick_winner[j] = int(players[j] == winner)
                trick_start = i + 1
        if trick_start < n:
            last_mover = players[n - 1]
            for j in range(trick_start, n):
                trick_winner[j] = int(players[j] == last_mover)
        result.loc[gdf.index] = trick_winner.astype(float)
    return result


class Big2SelfPlayDataset(Dataset):
    """Dataset for generate_nn_selfplay parquet output.

    Columns: game_id, turn_idx, current_player, move_at0..12,
             hand_at1..4, opp_at1..4, hint (float), flag (bool),
             pass_ (bool), winner.
    """

    def __init__(self, parquet_path: str,
                 val_game_ids: Optional[set] = None,
                 train: bool = True) -> None:
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(["game_id", "turn_idx"]).reset_index(drop=True)

        trick_labels = compute_trick_winners_selfplay(df)
        df["p_win_trick"] = trick_labels

        if val_game_ids is not None:
            if train:
                df = df[~df["game_id"].isin(val_game_ids)]
            else:
                df = df[df["game_id"].isin(val_game_ids)]

        df = df.dropna(subset=["p_win_trick"]).reset_index(drop=True)

        hand_counts = decode_handbits_np(
            df["hand_at1"].to_numpy(dtype=np.int32),
            df["hand_at2"].to_numpy(dtype=np.int32),
            df["hand_at3"].to_numpy(dtype=np.int32),
            df["hand_at4"].to_numpy(dtype=np.int32),
        )
        opp_counts = decode_handbits_np(
            df["opp_at1"].to_numpy(dtype=np.int32),
            df["opp_at2"].to_numpy(dtype=np.int32),
            df["opp_at3"].to_numpy(dtype=np.int32),
            df["opp_at4"].to_numpy(dtype=np.int32),
        )
        move_counts = df[[f"move_at{r}" for r in range(13)]].to_numpy(dtype=np.int32)

        self.hand_enc = torch.from_numpy(encode_exact_np(hand_counts))
        self.opp_enc  = torch.from_numpy(encode_upper_bound_np(opp_counts))
        self.move_enc = torch.from_numpy(encode_exact_np(move_counts))

        self.hint  = torch.tensor(df["hint"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.flag  = torch.tensor(df["flag"].to_numpy(dtype=bool), dtype=torch.bool)
        self.pass_ = torch.tensor(df["pass_"].to_numpy(dtype=bool), dtype=torch.bool)

        y_win = (df["winner"].to_numpy(dtype=np.int32) ==
                 df["current_player"].to_numpy(dtype=np.int32)).astype(np.float32)
        self.y_trick = torch.tensor(
            df["p_win_trick"].to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y_win = torch.tensor(y_win, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_win)

    def __getitem__(self, idx: int):
        return (
            self.hand_enc[idx],
            self.opp_enc[idx],
            self.move_enc[idx],
            self.hint[idx],
            self.flag[idx],
            self.pass_[idx],
            self.y_trick[idx],
            self.y_win[idx],
        )


def make_train_val_split(parquet_path: str,
                         val_frac: float = 0.1,
                         seed: int = 0) -> tuple[Big2Dataset, Big2Dataset]:
    """Return (train, val) datasets split by game ID, auto-detecting format."""
    import pyarrow.parquet as pq
    schema_names = set(pq.read_schema(parquet_path).names)
    is_selfplay = "game_id" in schema_names

    id_col = "game_id" if is_selfplay else "game_index"
    game_ids = pd.read_parquet(parquet_path, columns=[id_col])[id_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    n_val = max(1, int(len(game_ids) * val_frac))
    val_ids = set(game_ids[:n_val].tolist())

    DatasetClass = Big2SelfPlayDataset if is_selfplay else Big2Dataset
    train_ds = DatasetClass(parquet_path, val_game_ids=val_ids, train=True)
    val_ds   = DatasetClass(parquet_path, val_game_ids=val_ids, train=False)
    return train_ds, val_ds
