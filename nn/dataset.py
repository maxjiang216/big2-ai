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
ENCODING_DIM = sum(RANK_MAX_COUNTS)  # 48


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
    is_pass = move_sum == 0

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

    def __init__(
        self, parquet_path: str, val_game_ids: Optional[set] = None, train: bool = True
    ) -> None:
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
            df["hint"].to_numpy(dtype=np.float32) / 10000.0, dtype=torch.float32
        )
        self.flag = torch.tensor(
            df["opp_cannot_respond"].to_numpy(dtype=bool), dtype=torch.bool
        )
        self.pass_ = torch.tensor((move_cards.sum(axis=1) == 0), dtype=torch.bool)

        self.y_trick = torch.tensor(
            df["p_win_trick"].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.y_win = torch.tensor(
            df["turn_outcome"].to_numpy(dtype=np.float32), dtype=torch.float32
        )

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


def decode_handbits_np(
    at1: np.ndarray, at2: np.ndarray, at3: np.ndarray, at4: np.ndarray
) -> np.ndarray:
    """Decode HandBits thermometer bitfields → rank counts [N, 13].

    Bit r in atK is set iff count[r] >= K (thermometer encoding).
    """
    counts = np.zeros((len(at1), 13), dtype=np.int32)
    for r in range(13):
        bit = np.int32(1 << r)
        counts[:, r] = (
            ((at1 & bit) != 0).astype(np.int32)
            + ((at2 & bit) != 0).astype(np.int32)
            + ((at3 & bit) != 0).astype(np.int32)
            + ((at4 & bit) != 0).astype(np.int32)
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

    New format (hand_after_0..12 / opp_cnt_0..12 / trick_winner columns):
      C++ pre-computed rank counts and trick labels — no Python processing.

    Old format fallback (hand_at1..4 / opp_at1..4):
      Decodes HandBits and computes trick labels in Python.
    """

    def __init__(
        self,
        parquet_path: str,
        val_game_ids: Optional[set] = None,
        train: bool = True,
        subsample_frac: float = 1.0,
        seed: int = 0,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(["game_id", "turn_idx"]).reset_index(drop=True)

        new_format = "hand_after_0" in df.columns

        if new_format:
            # C++ already computed trick_winner — use directly, no Python loop needed.
            df["p_win_trick"] = df["trick_winner"].astype(float)
        else:
            trick_labels = compute_trick_winners_selfplay(df)
            df["p_win_trick"] = trick_labels

        if val_game_ids is not None:
            if train:
                df = df[~df["game_id"].isin(val_game_ids)]
            else:
                df = df[df["game_id"].isin(val_game_ids)]

        df = df.dropna(subset=["p_win_trick"]).reset_index(drop=True)

        if subsample_frac < 1.0:
            df = df.sample(frac=subsample_frac, random_state=seed).reset_index(
                drop=True
            )

        if new_format:
            hand_counts = df[[f"hand_after_{r}" for r in range(13)]].to_numpy(
                dtype=np.int32
            )
            opp_counts = df[[f"opp_cnt_{r}" for r in range(13)]].to_numpy(
                dtype=np.int32
            )
        else:
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
        self.opp_enc = torch.from_numpy(encode_upper_bound_np(opp_counts))
        self.move_enc = torch.from_numpy(encode_exact_np(move_counts))

        self.hint = torch.tensor(
            df["hint"].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.flag = torch.tensor(df["flag"].to_numpy(dtype=bool), dtype=torch.bool)
        self.pass_ = torch.tensor(df["pass_"].to_numpy(dtype=bool), dtype=torch.bool)

        if "vi_forced" in df.columns:
            vi_forced = df["vi_forced"].to_numpy(dtype=bool)
        else:
            vi_forced = np.zeros(len(df), dtype=bool)
        self.vi_forced = torch.tensor(vi_forced, dtype=torch.bool)

        if "vn_forced" in df.columns:
            vn_forced = df["vn_forced"].to_numpy(dtype=bool)
        else:
            vn_forced = np.zeros(len(df), dtype=bool)
        self.vn_forced = torch.tensor(vn_forced, dtype=torch.bool)

        y_win = (
            df["winner"].to_numpy(dtype=np.int32)
            == df["current_player"].to_numpy(dtype=np.int32)
        ).astype(np.float32)
        self.y_trick = torch.tensor(
            df["p_win_trick"].to_numpy(dtype=np.float32), dtype=torch.float32
        )
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
            self.vi_forced[idx],
            self.vn_forced[idx],
            self.y_trick[idx],
            self.y_win[idx],
        )


# ===========================================================================
# AlphaZero (az_search) datasets
#
# Two Parquet schemas, one per net (emitted by src/datagen/az_selfplay.cpp):
#
# Player file (az_player_genN.parquet) — one row per searched our-turn decision
# (tablebase turns excluded):
#   game_id, turn_idx                int32
#   hand_0..hand_12                  int32  our hand at the decision (pre-move)
#   opp_max_0..opp_max_12            int32  opponent max-cards-per-rank (thermo)
#   trick_0..trick_12                int32  current trick rank counts (0 == lead)
#   opp_size, our_size               int32  hand sizes
#   value                            float  1.0 if we won the game else 0.0
#   legal_moves                      list<int32>  legal engine move ids (mask)
#   visit_moves, visit_counts        list<int32>  MCTS root visits (policy target)
#
# Opponent file (az_opp_genN.parquet) — one row per real opponent decision:
#   game_id, turn_idx                int32
#   opp_max_0..opp_max_12            int32  mover max-cards-per-rank (thermo)
#   trick_0..trick_12                int32  current trick rank counts
#   opp_size, our_size               int32  mover / observer hand sizes
#   value                            float  1.0 if the observer (searcher) won
#   legal_moves                      list<int32>  plausible opp move ids (mask)
#   move_id                          int32  the move actually played (one-hot tgt)
# ===========================================================================

from nn.model_az import (  # noqa: E402
    OPP_HEAD_DIM,
    PLAYER_HEAD_DIM,
    az_opp_head_index,
    az_player_head_index,
)


def _counts(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_{r}" for r in range(13)]].to_numpy(dtype=np.int32)


def _as_int_list(v) -> np.ndarray:
    return np.asarray(v, dtype=np.int64)


class Big2AZPlayerDataset(Dataset):
    """Player-net samples: value target + MCTS visit-distribution policy target.

    Densifies the legal mask and visit distribution onto the PLAYER_HEAD_DIM
    head per item (collapsing TS5; DS8 ids never appear as player targets).
    Returns (hand, opp, trick, opp_size, our_size, value, mask, policy).
    """

    def __init__(
        self,
        parquet_path: str,
        val_game_ids: Optional[set] = None,
        train: bool = True,
        subsample_frac: float = 1.0,
        seed: int = 0,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        if val_game_ids is not None:
            df = df[~df["game_id"].isin(val_game_ids)] if train else df[
                df["game_id"].isin(val_game_ids)
            ]
        if subsample_frac < 1.0:
            df = df.sample(frac=subsample_frac, random_state=seed)
        df = df.reset_index(drop=True)

        self.hand_enc = torch.from_numpy(encode_exact_np(_counts(df, "hand")))
        self.opp_enc = torch.from_numpy(encode_upper_bound_np(_counts(df, "opp_max")))
        self.trick_enc = torch.from_numpy(encode_exact_np(_counts(df, "trick")))
        self.opp_size = torch.tensor(
            df["opp_size"].to_numpy(np.float32) / 16.0, dtype=torch.float32
        )
        self.our_size = torch.tensor(
            df["our_size"].to_numpy(np.float32) / 16.0, dtype=torch.float32
        )
        self.value = torch.tensor(df["value"].to_numpy(np.float32), dtype=torch.float32)
        self.legal = [_as_int_list(v) for v in df["legal_moves"].to_list()]
        self.vmoves = [_as_int_list(v) for v in df["visit_moves"].to_list()]
        self.vcounts = [_as_int_list(v) for v in df["visit_counts"].to_list()]

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, idx: int):
        mask = torch.zeros(PLAYER_HEAD_DIM, dtype=torch.bool)
        for m in self.legal[idx]:
            hi = az_player_head_index(int(m))
            if hi >= 0:
                mask[hi] = True
        policy = torch.zeros(PLAYER_HEAD_DIM, dtype=torch.float32)
        for m, c in zip(self.vmoves[idx], self.vcounts[idx]):
            hi = az_player_head_index(int(m))
            if hi >= 0:
                policy[hi] += float(c)
        total = policy.sum()
        if total > 0:
            policy /= total
        return (
            self.hand_enc[idx],
            self.opp_enc[idx],
            self.trick_enc[idx],
            self.opp_size[idx],
            self.our_size[idx],
            self.value[idx],
            mask,
            policy,
        )


class Big2AZOppDataset(Dataset):
    """Opponent-net samples: value target + one-hot behavior (imitation) target.

    Returns (opp, trick, opp_size, our_size, value, mask, target_idx).
    """

    def __init__(
        self,
        parquet_path: str,
        val_game_ids: Optional[set] = None,
        train: bool = True,
        subsample_frac: float = 1.0,
        seed: int = 0,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        if val_game_ids is not None:
            df = df[~df["game_id"].isin(val_game_ids)] if train else df[
                df["game_id"].isin(val_game_ids)
            ]
        if subsample_frac < 1.0:
            df = df.sample(frac=subsample_frac, random_state=seed)
        df = df.reset_index(drop=True)

        self.opp_enc = torch.from_numpy(encode_upper_bound_np(_counts(df, "opp_max")))
        self.trick_enc = torch.from_numpy(encode_exact_np(_counts(df, "trick")))
        self.opp_size = torch.tensor(
            df["opp_size"].to_numpy(np.float32) / 16.0, dtype=torch.float32
        )
        self.our_size = torch.tensor(
            df["our_size"].to_numpy(np.float32) / 16.0, dtype=torch.float32
        )
        self.value = torch.tensor(df["value"].to_numpy(np.float32), dtype=torch.float32)
        self.legal = [_as_int_list(v) for v in df["legal_moves"].to_list()]
        self.target_idx = torch.tensor(
            [az_opp_head_index(int(m)) for m in df["move_id"].to_numpy(np.int64)],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, idx: int):
        mask = torch.zeros(OPP_HEAD_DIM, dtype=torch.bool)
        for m in self.legal[idx]:
            mask[az_opp_head_index(int(m))] = True
        # The played move is always legal; guarantee it is unmasked.
        mask[self.target_idx[idx]] = True
        return (
            self.opp_enc[idx],
            self.trick_enc[idx],
            self.opp_size[idx],
            self.our_size[idx],
            self.value[idx],
            mask,
            self.target_idx[idx],
        )


def make_az_split(
    parquet_paths: "str | list[str]",
    kind: str,
    val_frac: float = 0.1,
    seed: int = 0,
    mix_decay: float = 1.0,
) -> "tuple[Dataset, Dataset]":
    """(train, val) split by game_id for an AZ player ('player') or opp ('opp')
    dataset, with mix_decay subsampling of older-generation files."""
    from torch.utils.data import ConcatDataset

    if isinstance(parquet_paths, str):
        parquet_paths = [parquet_paths]
    Cls = Big2AZPlayerDataset if kind == "player" else Big2AZOppDataset

    train_parts, val_parts = [], []
    for i, p in enumerate(parquet_paths):
        gids = pd.read_parquet(p, columns=["game_id"])["game_id"].unique()
        rng = np.random.default_rng(seed + i)
        rng.shuffle(gids)
        n_val = max(1, int(len(gids) * val_frac))
        val_ids = set(gids[:n_val].tolist())
        frac = mix_decay**i
        train_parts.append(Cls(p, val_ids, True, frac, seed + i))
        val_parts.append(Cls(p, val_ids, False, frac, seed + i))

    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return train_ds, val_ds


def _split_one(
    parquet_path: str, val_frac: float, seed: int, subsample_frac: float = 1.0
) -> tuple[Dataset, Dataset]:
    """Train/val split for a single parquet file."""
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
    kwargs = {"subsample_frac": subsample_frac, "seed": seed} if is_selfplay else {}
    return (
        DatasetClass(parquet_path, val_game_ids=val_ids, train=True, **kwargs),
        DatasetClass(parquet_path, val_game_ids=val_ids, train=False, **kwargs),
    )


def make_train_val_split(
    parquet_paths: "str | list[str]",
    val_frac: float = 0.1,
    seed: int = 0,
    mix_decay: float = 1.0,
) -> "tuple[Dataset, Dataset]":
    """Return (train, val) datasets split by game ID, auto-detecting format.

    parquet_paths may be a single path or a list. When mix_decay < 1, each
    subsequent file is subsampled by mix_decay^i (file 0 = full, file 1 = mix_decay,
    file 2 = mix_decay^2, etc.) to blend older-generation data.
    """
    from torch.utils.data import ConcatDataset

    if isinstance(parquet_paths, str):
        parquet_paths = [parquet_paths]

    fracs = [mix_decay**i for i in range(len(parquet_paths))]
    splits = [
        _split_one(p, val_frac, seed + i, subsample_frac=fracs[i])
        for i, p in enumerate(parquet_paths)
    ]
    train_parts, val_parts = zip(*splits)
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return train_ds, val_ds
