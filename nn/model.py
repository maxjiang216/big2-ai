"""Big 2 strategic DNN.

Input shape per sample:
  hand   : float32[48] - exact-count one-hots for player's hand after playing M
  opp    : float32[48] - upper-bound one-hots for opponent's max-possible hand
  move   : float32[48] - exact-count one-hots for move M (all-zero for pass)
  hint   : float32     - MC estimate of opponent response probability
  flag   : bool        - opponent is definitely locked (cannot respond to M)
  pass_  : bool        - M is a pass

Outputs (all probabilities in [0, 1]):
  v_init    - win rate given player wins this trick
  v_no_init - win rate given player loses this trick
  p_trick   - probability player wins this trick
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Input encoding dimensions
# ---------------------------------------------------------------------------

# For each rank index 0..12: number of one-hot bits (= max cards of that rank).
# Ranks 0-10 (3-K): max 4.  Rank 11 (A): max 3.  Rank 12 (2): max 1.
RANK_MAX_COUNTS = [4] * 11 + [3, 1]
ENCODING_DIM = sum(RANK_MAX_COUNTS)  # 48


def encode_exact(rank_counts: torch.Tensor) -> torch.Tensor:
    """Exact-count one-hot encoding for a hand or move.

    rank_counts: int tensor [..., 13] with card counts per rank.
    Returns float tensor [..., 48].
    """
    *batch, _ = rank_counts.shape
    out = torch.zeros(*batch, ENCODING_DIM, dtype=torch.float32,
                      device=rank_counts.device)
    offset = 0
    for r, max_k in enumerate(RANK_MAX_COUNTS):
        c = rank_counts[..., r].long()
        for k in range(1, max_k + 1):
            out[..., offset + k - 1] = (c == k).float()
        offset += max_k
    return out


def encode_upper_bound(rank_counts: torch.Tensor) -> torch.Tensor:
    """Cumulative (thermometer) encoding for opponent max-possible hand.

    rank_counts: int tensor [..., 13] with max possible counts per rank.
    Returns float tensor [..., 48].
    """
    *batch, _ = rank_counts.shape
    out = torch.zeros(*batch, ENCODING_DIM, dtype=torch.float32,
                      device=rank_counts.device)
    offset = 0
    for r, max_k in enumerate(RANK_MAX_COUNTS):
        c = rank_counts[..., r].clamp(0, max_k).long()
        for k in range(1, max_k + 1):
            out[..., offset + k - 1] = (c >= k).float()
        offset += max_k
    return out


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _init_mish(layer: nn.Linear) -> nn.Linear:
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(layer.bias)
    return layer


def _init_selu(layer: nn.Linear) -> nn.Linear:
    # LeCun Normal: std = 1 / sqrt(fan_in)
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")
    nn.init.zeros_(layer.bias)
    return layer


class ResBlock(nn.Module):
    """SELU residual block: Linear → SELU → Linear → SELU(x + residual)."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.fc1 = _init_selu(nn.Linear(dim, dim))
        self.fc2 = _init_selu(nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.selu(self.fc1(x))
        return F.selu(self.fc2(x) + residual)


class Head(nn.Module):
    """Output head: Linear(in_dim, 128) → LeakyReLU → Linear(128, 1) → Sigmoid."""

    def __init__(self, in_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in",
                                nonlinearity="leaky_relu", a=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x), 0.01)
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Big2Net(nn.Module):
    """Big 2 strategic engine DNN.

    Architecture:
      A  (48→64, Mish, shared across hand/opp/move)
      B  (64→64, Mish, shared across player/opp hands)
      B' (64→64, Mish, independent for move)
      C  (64→128, Mish, independent, player hand only)
      C' (64→64,  Mish, independent, opponent only)
      move keeps 64-dim output of B'

      Junction: cat([C:128, C':64, B':64]) → 256
      Gating: sigmoid(Linear(1,256)(hint)) element-wise scale
      Hard mask: zero move dims if pass or flag
      LayerNorm(256)
      Trunk: 2 × ResBlock(256, SELU)

      E1  256      → Head → v_init    (sigmoid)
      E2  256      → Head → v_no_init (sigmoid)
      E3  256+hint → Head → p_trick   (sigmoid, post-processed for pass/flag)
    """

    def __init__(self) -> None:
        super().__init__()

        # Shared embedding layers
        self.layer_a = _init_mish(nn.Linear(ENCODING_DIM, 64))
        self.layer_b_hands = _init_mish(nn.Linear(64, 64))
        self.layer_b_move = _init_mish(nn.Linear(64, 64))

        # Per-role refinement layers
        self.layer_c_player = _init_mish(nn.Linear(64, 128))
        self.layer_c_opp = _init_mish(nn.Linear(64, 64))

        # Hint gate: projects scalar hint to 256-dim element-wise gate
        self.hint_gate = nn.Linear(1, 256)
        nn.init.normal_(self.hint_gate.weight, 0.0, 0.01)
        nn.init.zeros_(self.hint_gate.bias)

        self.junction_norm = nn.LayerNorm(256)
        self.trunk = nn.Sequential(ResBlock(256), ResBlock(256))

        # Output heads
        self.head_v_init = Head(256)
        self.head_v_no_init = Head(256)
        self.head_p_trick = Head(257)  # trunk + hint scalar

    def forward(
        self,
        hand: torch.Tensor,    # [B, 48] float
        opp: torch.Tensor,     # [B, 48] float
        move: torch.Tensor,    # [B, 48] float
        hint: torch.Tensor,    # [B]     float  (0-1)
        flag: torch.Tensor,    # [B]     bool   (opponent locked)
        pass_: torch.Tensor,   # [B]     bool   (move is pass)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (v_init, v_no_init, p_trick), each shape [B]."""
        # --- Shared Layer A ---
        h = F.mish(self.layer_a(hand))
        o = F.mish(self.layer_a(opp))
        m = F.mish(self.layer_a(move))

        # --- Shared Layer B (hands only) ---
        h = F.mish(self.layer_b_hands(h))
        o = F.mish(self.layer_b_hands(o))

        # --- Independent Layer B' (move) ---
        m = F.mish(self.layer_b_move(m))

        # --- Independent refinement ---
        h = F.mish(self.layer_c_player(h))   # [B, 128]
        o = F.mish(self.layer_c_opp(o))       # [B, 64]

        # Hard mask: zero move embedding when pass or flag
        active_move = ~(pass_ | flag)
        m = m * active_move.float().unsqueeze(-1)   # [B, 64]

        # --- Junction ---
        x = torch.cat([h, o, m], dim=-1)             # [B, 256]

        # Hint-derived gate
        gate = torch.sigmoid(self.hint_gate(hint.unsqueeze(-1)))  # [B, 256]
        x = x * gate

        x = self.junction_norm(x)
        x = self.trunk(x)                             # [B, 256]

        # --- Output heads ---
        v_init = self.head_v_init(x)
        v_no_init = self.head_v_no_init(x)

        x_hint = torch.cat([x, hint.unsqueeze(-1)], dim=-1)  # [B, 257]
        p_trick_raw = self.head_p_trick(x_hint)

        # Post-process p_trick: flag → 1.0, pass → 0.0
        p_trick = torch.where(flag, torch.ones_like(p_trick_raw), p_trick_raw)
        p_trick = torch.where(pass_, torch.zeros_like(p_trick), p_trick)

        return v_init, v_no_init, p_trick

    def embedding_param_ids(self) -> set:
        """IDs of parameters in shared embedding layers (A, B, B').

        Used by train.py to exclude them from weight decay.
        """
        ids = set()
        for layer in (self.layer_a, self.layer_b_hands, self.layer_b_move):
            for p in layer.parameters():
                ids.add(id(p))
        return ids
