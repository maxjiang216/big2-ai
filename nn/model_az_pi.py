"""Perfect-information AlphaZero network (az_pi).

A single configurable net for the perfect-information game: both hands are
visible, the net is MEMORYLESS (no discard pile, no move history). It mirrors the
imperfect-information ``Big2NetAZ`` (nn/model_az.py) but feeds the opponent
branch the opponent's EXACT hand instead of a thermometer upper bound, and drops
the opponent/behavior/q_a/aux heads — only a scalar value and the factored
138-dim player policy head remain.

Inputs (all consumed verbatim from the Parquet ``enc`` column; the C++ self-play
writer fills them with exact encodings — see src/datagen/nn_encode.h):
  hand     [B,48] exact   : side-to-move's exact hand
  opp      [B,48] exact   : opponent's exact hand
  trick    [B,48] exact   : current trick (move to beat; all-zero == lead)
  my_pts   [B]   float/50 : mover's series points (series-score conditioning)
  opp_pts  [B]   float/50 : opponent's series points

Outputs:
  value         [B]                  sigmoid, P(side to move wins the SERIES)
  policy_logits [B, PLAYER_HEAD_DIM]  raw factored logits (compose via C-matrix)

The width/depth/embedding are constructor args so the same code drives model-size
sweeps; the shape is baked into the TorchScript export, so the C++ evaluator
needs no size flags.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.model import ENCODING_DIM, ResBlock, _init_selu, _init_swish
from nn.model_az import PLAYER_HEAD_DIM, PolicyHead, ValueHead


class Big2NetAZPI(nn.Module):
    def __init__(self, width: int = 256, blocks: int = 2, embed: int = 64) -> None:
        super().__init__()
        self.width = width
        self.blocks = blocks
        self.embed = embed

        e = embed
        c_player = 2 * e
        c_opp = (3 * e) // 2
        b_trick = e // 2

        # Tied card-literacy embedding shared across the three 48-dim inputs.
        self.layer_a = _init_swish(nn.Linear(ENCODING_DIM, e))
        self.layer_b_hands = _init_swish(nn.Linear(e, e))
        self.layer_b_trick = _init_swish(nn.Linear(e, b_trick))
        self.layer_c_player = _init_swish(nn.Linear(e, c_player))
        self.layer_c_opp = _init_swish(nn.Linear(e, c_opp))
        # Series-score conditioning: [my_pts/50, opp_pts/50]. Hand sizes are
        # derivable from the exact encodings and are no longer net inputs.
        self.pts_embed = _init_swish(nn.Linear(2, 16))

        junction_in = c_player + c_opp + b_trick + 16
        self.junction = _init_selu(nn.Linear(junction_in, width))
        self.junction_norm = nn.LayerNorm(width)
        self.trunk = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])

        self.value_head = ValueHead(width)
        self.policy_head = PolicyHead(width, PLAYER_HEAD_DIM)

    def forward(
        self,
        hand: torch.Tensor,  # [B, 48] exact
        opp: torch.Tensor,  # [B, 48] exact (opponent's true hand)
        trick: torch.Tensor,  # [B, 48] exact (all-zero == lead)
        my_pts: torch.Tensor,  # [B] float, mover series points (/50)
        opp_pts: torch.Tensor,  # [B] float, opponent series points (/50)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.mish(self.layer_a(hand))
        h = F.mish(self.layer_b_hands(h))
        h = F.mish(self.layer_c_player(h))

        o = F.mish(self.layer_a(opp))
        o = F.mish(self.layer_b_hands(o))
        o = F.mish(self.layer_c_opp(o))

        t = F.mish(self.layer_a(trick))
        t = F.mish(self.layer_b_trick(t))

        s = F.mish(self.pts_embed(torch.stack([my_pts, opp_pts], dim=-1)))

        x = torch.cat([h, o, t, s], dim=-1)
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
