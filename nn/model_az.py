"""AlphaZero-style networks for the `az_search` player.

Two separate nets, both reusing Big2Net's embedding style (tied 48->64 card
embedding, role-specialised refinement, SELU ResBlock trunk, LayerNorm at the
junction). Both emit RAW policy/behavior LOGITS — the legal-move mask and the
softmax over the legal subset are applied by the caller (C++ search / training
loss), never inside the net.

Encodings (see src/datagen/nn_encode.h for the C++ twins):
  * hand / trick  -> encode_exact      : exact per-rank count one-hot   (48)
  * opp max-cards -> encode_upper_bound : thermometer per-rank upper bound (48)
Scalar hand sizes are passed in normalised by 16.

Head dimensions come from src/players/az_search/considered_moves.h and are
mirrored here by `az_player_head_index` / `az_opp_head_index` so the Parquet
move-id targets map onto the right logit. The mapping rule:
  identity for ids [0, TS5_BEGIN); the 7 TS5 ids collapse to one slot in both
  heads; the 5 DS8 ids are dropped from the player head and collapse to one slot
  in the opponent head.

Net I/O
  Big2NetAZ.forward(hand, opp, trick, opp_size, our_size)
      -> value[B] (sigmoid, P(player-to-move wins)),
         policy_logits[B, PLAYER_HEAD_DIM]
  Big2NetOpp.forward(hand, opp, trick, opp_size, our_size)
      -> move_value[B, OPP_HEAD_DIM] (sigmoid; per slot, P(searcher wins after
         the opponent plays that move)),
         behavior_logits[B, OPP_HEAD_DIM]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.model import ENCODING_DIM, ResBlock, _init_selu, _init_swish

# ---------------------------------------------------------------------------
# Head dimensions — must stay in sync with considered_moves.h.
#
# Engine move-id layout (see src/core/util.h):
#   identity region : [0, TS5_BEGIN)
#   TS5 ids         : [TS5_BEGIN, DS8_BEGIN)   (7 ids -> 1 slot, both heads)
#   DS8 ids         : [DS8_BEGIN, LEGAL_MOVES) (5 ids -> dropped/1 slot)
# ---------------------------------------------------------------------------
TS5_BEGIN = 456   # kTRIPLESTRAIGHT5_START
DS8_BEGIN = 463   # kDOUBLESTRAIGHT8_START
LEGAL_MOVES = 468

TS5_SLOT = TS5_BEGIN
OPP_DS8_SLOT = TS5_BEGIN + 1

PLAYER_HEAD_DIM = TS5_BEGIN + 1  # 457
OPP_HEAD_DIM = TS5_BEGIN + 2     # 458


def az_player_head_index(move_id: int) -> int:
    """Engine move id -> player policy head index, or -1 if dropped (DS8)."""
    if move_id < TS5_BEGIN:
        return move_id
    if move_id < DS8_BEGIN:
        return TS5_SLOT
    return -1


def az_opp_head_index(move_id: int) -> int:
    """Engine move id -> opponent behavior head index (always valid)."""
    if move_id < TS5_BEGIN:
        return move_id
    if move_id < DS8_BEGIN:
        return TS5_SLOT
    return OPP_DS8_SLOT


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Scalar value in [0, 1]: Linear -> LeakyReLU -> Linear -> Sigmoid."""

    def __init__(self, in_dim: int = 256, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.kaiming_normal_(
            self.fc1.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01
        )
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x), 0.01)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


class PolicyHead(nn.Module):
    """Raw logits over the head's move slots: Linear -> SELU -> Linear (no act.)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = _init_selu(nn.Linear(in_dim, hidden))
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.selu(self.fc1(x))
        return self.fc2(x)  # raw logits


class PerMoveValueHead(nn.Module):
    """Per-slot value q_a in [0,1]: Linear -> SELU -> Linear -> Sigmoid.

    One value per move slot — q_a = P(searcher wins after the opponent plays a).
    The search reads the played/representative move's slot; the node's scalar
    value is derived outside the net as Sum_a prior(a)*q_a.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = _init_selu(nn.Linear(in_dim, hidden))
        self.fc2 = nn.Linear(hidden, out_dim)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.selu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# ---------------------------------------------------------------------------
# Player net — position-conditioned dual head (true AlphaZero).
# ---------------------------------------------------------------------------


class Big2NetAZ(nn.Module):
    """Inputs our hand (exact), opp max-cards (thermo), current trick (exact),
    and the two hand sizes; outputs a value scalar and policy logits."""

    def __init__(self) -> None:
        super().__init__()
        # Tied 48->64 card-literacy embedding, shared across the three 48-dim
        # count inputs (hand / opp / trick).
        self.layer_a = _init_swish(nn.Linear(ENCODING_DIM, 64))
        # Role refinement.
        self.layer_b_hands = _init_swish(nn.Linear(64, 64))
        self.layer_b_trick = _init_swish(nn.Linear(64, 32))  # trick is info-light
        self.layer_c_player = _init_swish(nn.Linear(64, 128))
        self.layer_c_opp = _init_swish(nn.Linear(64, 96))
        # Scalar hand sizes (opp_size, our_size) -> small embedding.
        self.size_embed = _init_swish(nn.Linear(2, 16))

        # Junction: [player128, opp96, trick32, size16] = 272 -> 256.
        self.junction = _init_selu(nn.Linear(128 + 96 + 32 + 16, 256))
        self.junction_norm = nn.LayerNorm(256)
        self.trunk = nn.Sequential(ResBlock(256), ResBlock(256))

        self.value_head = ValueHead(256)
        self.policy_head = PolicyHead(256, PLAYER_HEAD_DIM)

    def forward(
        self,
        hand: torch.Tensor,  # [B, 48] exact
        opp: torch.Tensor,  # [B, 48] thermo
        trick: torch.Tensor,  # [B, 48] exact (all-zero == we hold initiative)
        opp_size: torch.Tensor,  # [B] float (normalised /16)
        our_size: torch.Tensor,  # [B] float (normalised /16)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.mish(self.layer_a(hand))
        h = F.mish(self.layer_b_hands(h))
        h = F.mish(self.layer_c_player(h))  # [B, 128]

        o = F.mish(self.layer_a(opp))
        o = F.mish(self.layer_b_hands(o))
        o = F.mish(self.layer_c_opp(o))  # [B, 96]

        t = F.mish(self.layer_a(trick))
        t = F.mish(self.layer_b_trick(t))  # [B, 32]

        s = F.mish(self.size_embed(torch.stack([opp_size, our_size], dim=-1)))  # [B,16]

        x = torch.cat([h, o, t, s], dim=-1)  # [B, 272]
        x = self.junction_norm(F.selu(self.junction(x)))
        x = self.trunk(x)

        value = self.value_head(x)
        policy_logits = self.policy_head(x)
        return value, policy_logits

    def embedding_param_ids(self) -> set:
        ids = set()
        for layer in (self.layer_a, self.layer_b_hands, self.layer_b_trick):
            for p in layer.parameters():
                ids.add(id(p))
        return ids


# ---------------------------------------------------------------------------
# Opponent behavior net — public information only.
# ---------------------------------------------------------------------------


class Big2NetOpp(nn.Module):
    """Inputs our hand (exact), opp max-cards (thermo), current trick (exact),
    and the two hand sizes — same inputs as the player net. Outputs a PER-MOVE
    value q_a (one per opp head slot, P(searcher wins after that opp move)) and
    behavior logits. Our hand is needed because the post-move value depends on
    whether/how we can respond and what we do next; the behavior priors riding
    along on it are a better posterior (deck correlation), not a bias."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_a = _init_swish(nn.Linear(ENCODING_DIM, 64))
        self.layer_b_hands = _init_swish(nn.Linear(64, 64))
        self.layer_b_trick = _init_swish(nn.Linear(64, 32))
        self.layer_c_player = _init_swish(nn.Linear(64, 128))
        self.layer_c_opp = _init_swish(nn.Linear(64, 96))
        self.size_embed = _init_swish(nn.Linear(2, 16))

        # Junction: [hand128, opp96, trick32, size16] = 272 -> 256 (== player net).
        self.junction = _init_selu(nn.Linear(128 + 96 + 32 + 16, 256))
        self.junction_norm = nn.LayerNorm(256)
        self.trunk = nn.Sequential(ResBlock(256), ResBlock(256))

        self.value_head = PerMoveValueHead(256, OPP_HEAD_DIM)
        self.behavior_head = PolicyHead(256, OPP_HEAD_DIM)

    def forward(
        self,
        hand: torch.Tensor,  # [B, 48] exact (our hand)
        opp: torch.Tensor,  # [B, 48] thermo
        trick: torch.Tensor,  # [B, 48] exact
        opp_size: torch.Tensor,  # [B] float (normalised /16)
        our_size: torch.Tensor,  # [B] float (normalised /16)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.mish(self.layer_a(hand))
        h = F.mish(self.layer_b_hands(h))
        h = F.mish(self.layer_c_player(h))  # [B, 128]

        o = F.mish(self.layer_a(opp))
        o = F.mish(self.layer_b_hands(o))
        o = F.mish(self.layer_c_opp(o))  # [B, 96]

        t = F.mish(self.layer_a(trick))
        t = F.mish(self.layer_b_trick(t))  # [B, 32]

        s = F.mish(self.size_embed(torch.stack([opp_size, our_size], dim=-1)))  # [B,16]

        x = torch.cat([h, o, t, s], dim=-1)  # [B, 272]
        x = self.junction_norm(F.selu(self.junction(x)))
        x = self.trunk(x)

        move_value = self.value_head(x)  # [B, OPP_HEAD_DIM] in [0,1]
        behavior_logits = self.behavior_head(x)
        return move_value, behavior_logits

    def embedding_param_ids(self) -> set:
        ids = set()
        for layer in (self.layer_a, self.layer_b_hands, self.layer_b_trick):
            for p in layer.parameters():
                ids.add(id(p))
        return ids
