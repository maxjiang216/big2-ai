"""History-transformer ("memory") network for the `az_search` player.

ONE unified net replacing the Big2NetAZ / Big2NetOpp pair: a causal
transformer trunk over the game's move history plus a per-position readout
with side inputs, emitting all four heads:

  * policy   [B, PLAYER_HEAD_DIM=138]  factored player-policy logits (raw)
  * value    [B]                       sigmoid, P(hand-owner wins)
  * behavior [B, OPP_HEAD_DIM=458]     opponent behavior logits (raw)
  * qa       [B, OPP_HEAD_DIM=458]     sigmoid, P(owner wins after opp move a)

All outputs are from the perspective of the HAND OWNER (the player whose
exact hand is fed in). `owner_to_move` says whether the owner is the side to
act at the readout position; the search uses (policy, value) when the owner
moves and (behavior, qa) when the opponent moves.

Tokens: each history move is its 48-dim exact-count card encoding (pass = all
zeros), looked up from a fixed [468, 48] buffer built from the C++ dump
nn/az_token_cards.json (research/az_tokens_gen, single source of truth). C++
only ever sends int move ids across the boundary.

Sequence/index convention (see plan): sequence = [BOS, move_0, move_1, ...];
move_k sits at sequence index k+1. A decision taken after `hist_idx` moves
reads the hidden state at sequence index `hist_idx` (BOS when 0).

TorchScript surface used by C++ (LibTorch, via get_method):
  config()        -> [n_layers, n_heads, head_dim, d_model, seq_cap]
  encode_prefix(tokens[B,P] i64, lens[B] i64)
                  -> kv[B,L,2,H,P+1,Dh], h_last[B,d]
  forward_leaf(kv, plen[B], h_last, path[B,T] i64, path_len[B],
               hand[B,48], oppmax[B,48], osz[B], usz[B], otm[B], mpts[B], opts[B])
                  -> value[B], policy[B,138], behavior[B,458], qa[B,458]
  readout(h, hand, oppmax, osz, usz, otm, mpts, opts) -> 4-tuple (debug/training)
  forward_full(tokens[B,T] i64) -> hidden[B,T+1,d] (training / --no-kv-cache)
"""

from __future__ import annotations

import json
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.model import ENCODING_DIM, ResBlock, _init_selu, _init_swish
from nn.model_az import (
    OPP_HEAD_DIM,
    PLAYER_HEAD_DIM,
    PerMoveValueHead,
    PolicyHead,
    ValueHead,
)

NUM_MOVES = 468
SEQ_CAP = 66  # BOS + 64-move history cap + 1 slack
SIZE_NORM = 16.0
PTS_NORM = 50.0  # series points normaliser (target = kSeriesTarget)
NEG_MASK = -1e9  # additive attention mask for invalid keys (finite: no NaN rows)


def load_token_feats(path: str = "nn/az_token_cards.json") -> torch.Tensor:
    """Fixed [NUM_MOVES, 48] token feature matrix from the C++ dump: per move
    id, the exact-count 48-dim encoding of the cards it plays (pass row 0 is
    all zeros)."""
    import numpy as np

    from nn.dataset import encode_exact_np

    with open(path) as f:
        d = json.load(f)
    assert d["num_moves"] == NUM_MOVES, "token dump move-count mismatch"
    cards = np.asarray(d["cards"], dtype=np.int64)
    assert cards.shape == (NUM_MOVES, 13)
    assert (cards[0] == 0).all(), "pass row must be all zeros"
    return torch.from_numpy(encode_exact_np(cards))


class _SeqBlock(nn.Module):
    """One pre-LN transformer layer, split so the KV-cache path can reuse the
    same projections: qkv_proj() -> (q, k, v) heads; post() applies the
    attention output projection + the feed-forward sublayer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        # GPT-style: shrink residual-path output projections with depth.
        nn.init.normal_(self.attn_out.weight, std=0.02 / math.sqrt(2.0 * n_layers))
        nn.init.normal_(self.ff2.weight, std=0.02 / math.sqrt(2.0 * n_layers))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def qkv_proj(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        return self._split(q), self._split(k), self._split(v)

    def post(self, x: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        B = attn.shape[0]
        T = attn.shape[2]
        a = attn.transpose(1, 2).reshape(B, T, self.d_model)
        x = x + self.attn_out(a)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x


class Big2NetSeqAZ(nn.Module):
    def __init__(
        self,
        token_feats: torch.Tensor,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        seq_cap: int = SEQ_CAP,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_cap = seq_cap
        self.neg_mask = NEG_MASK  # plain float attr: scriptable, unlike a global

        assert token_feats.shape == (NUM_MOVES, ENCODING_DIM)
        self.register_buffer("token_feats", token_feats.float())

        # Token pipeline: 48-dim card counts -> d_model, + learned positions.
        self.tok_proj = nn.Linear(ENCODING_DIM, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(seq_cap, d_model))
        self.bos = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.bos, std=0.02)

        # Pre-LN causal transformer trunk.
        self.blocks = nn.ModuleList(
            [_SeqBlock(d_model, n_heads, d_ff, n_layers) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Readout: card-literacy stack (Big2Net style) for the side inputs.
        self.layer_a = _init_swish(nn.Linear(ENCODING_DIM, 64))
        self.layer_b_hands = _init_swish(nn.Linear(64, 64))
        self.layer_c_player = _init_swish(nn.Linear(64, 128))
        self.layer_c_opp = _init_swish(nn.Linear(64, 96))
        # (opp_size/16, our_size/16, owner_to_move, my_pts/50, opp_pts/50)
        # -> small embedding. The two series-points inputs let the net modulate
        # aggression by match state; converting an old (3-input) checkpoint
        # zero-fills their weight columns (scripts/convert_az_seq_series.py).
        self.size_embed = _init_swish(nn.Linear(5, 16))

        # Junction: [hist d_model, hand128, opp96, size16] -> 256.
        self.junction = _init_selu(nn.Linear(d_model + 128 + 96 + 16, 256))
        self.junction_norm = nn.LayerNorm(256)
        self.trunk = nn.Sequential(ResBlock(256), ResBlock(256))

        self.value_head = ValueHead(256)
        self.policy_head = PolicyHead(256, PLAYER_HEAD_DIM)
        self.behavior_head = PolicyHead(256, OPP_HEAD_DIM)
        self.qa_head = PerMoveValueHead(256, OPP_HEAD_DIM)

    # -- trunk pieces -------------------------------------------------------

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, T] int64 move ids -> [B, T+1, d] with BOS at index 0 and
        positional embeddings added."""
        B, T = tokens.shape
        tok = self.tok_proj(F.embedding(tokens, self.token_feats))  # [B, T, d]
        bos = self.bos.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model)
        x = torch.cat([bos, tok], dim=1)  # [B, T+1, d]
        return x + self.pos_emb[: T + 1].unsqueeze(0)

    # -- exported surface ---------------------------------------------------

    @torch.jit.export
    def config(self) -> List[int]:
        return [self.n_layers, self.n_heads, self.head_dim, self.d_model, self.seq_cap]

    @torch.jit.export
    def forward_full(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full causal trunk: tokens [B, T] int64 -> hidden [B, T+1, d].
        Padded tail positions produce garbage hiddens but (causality) never
        contaminate positions <= the true length; callers only read those."""
        x = self._embed(tokens)
        S = x.shape[1]
        causal = torch.full((S, S), self.neg_mask, dtype=x.dtype, device=x.device)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            q, k, v = blk.qkv_proj(x)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            x = blk.post(x, attn)
        return self.ln_f(x)

    @torch.jit.export
    def encode_prefix(
        self, tokens: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode game-history prefixes (per real turn, once per slot).

        tokens [B, P] int64 (padded), lens [B] = true move counts.
        Returns kv [B, L, 2, H, P+1, Dh] (post-LN1 projections; padded key
        rows are garbage and must be masked by the consumer via plen=lens+1)
        and h_last [B, d] = final hidden at sequence index lens.
        """
        x = self._embed(tokens)
        B, S, _ = x.shape
        causal = torch.full((S, S), self.neg_mask, dtype=x.dtype, device=x.device)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
        kvs: List[torch.Tensor] = []
        for blk in self.blocks:
            q, k, v = blk.qkv_proj(x)
            kvs.append(torch.stack([k, v], dim=1))  # [B, 2, H, S, Dh]
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            x = blk.post(x, attn)
        kv = torch.stack(kvs, dim=1)  # [B, L, 2, H, S, Dh]
        h = self.ln_f(x)
        idx = lens.view(B, 1, 1).expand(B, 1, self.d_model)
        h_last = h.gather(1, idx).squeeze(1)
        return kv, h_last

    @torch.jit.export
    def forward_leaf(
        self,
        kv: torch.Tensor,  # [B, L, 2, H, P1, Dh] cached prefix (any float dtype)
        plen: torch.Tensor,  # [B] int64 valid prefix length INCL. BOS (= lens+1)
        h_last: torch.Tensor,  # [B, d] prefix readout hidden (used iff path empty)
        path: torch.Tensor,  # [B, T] int64 in-tree move ids (padded)
        path_len: torch.Tensor,  # [B] int64 true path lengths (0 allowed)
        hand: torch.Tensor,  # [B, 48] exact
        oppmax: torch.Tensor,  # [B, 48] thermo
        osz: torch.Tensor,  # [B] float (/16)
        usz: torch.Tensor,  # [B] float (/16)
        otm: torch.Tensor,  # [B] float owner_to_move
        mpts: torch.Tensor,  # [B] float owner series points (/50)
        opts: torch.Tensor,  # [B] float opponent series points (/50)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kv = kv.to(h_last.dtype)
        B, T = path.shape
        P1 = kv.shape[4]
        # Embed path tokens at their absolute sequence positions plen + i.
        tok = self.tok_proj(F.embedding(path, self.token_feats))  # [B, T, d]
        pos = plen.unsqueeze(1) + torch.arange(T, device=path.device).unsqueeze(0)
        pos = pos.clamp(max=self.seq_cap - 1)
        x = tok + F.embedding(pos, self.pos_emb)
        # Additive mask [B, 1, T, P1+T]: prefix key j valid iff j < plen_b;
        # path key i' valid iff i' <= i (causal). Padded queries (i >=
        # path_len) still see valid prefix keys, so no NaN softmax rows.
        ar_p = torch.arange(P1, device=path.device)
        mask_pre = torch.where(
            ar_p.unsqueeze(0) < plen.unsqueeze(1),
            torch.zeros(B, P1, dtype=x.dtype, device=x.device),
            torch.full((B, P1), self.neg_mask, dtype=x.dtype, device=x.device),
        )  # [B, P1]
        mask_pre = mask_pre.unsqueeze(1).unsqueeze(2).expand(B, 1, T, P1)
        ar_t = torch.arange(T, device=path.device)
        causal = torch.where(
            ar_t.unsqueeze(1) >= ar_t.unsqueeze(0),
            torch.zeros(T, T, dtype=x.dtype, device=x.device),
            torch.full((T, T), self.neg_mask, dtype=x.dtype, device=x.device),
        )  # [T, T] query i, key i'
        mask = torch.cat(
            [mask_pre, causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)], dim=3
        )
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv_proj(x)
            pk = kv[:, i, 0]  # [B, H, P1, Dh]
            pv = kv[:, i, 1]
            attn = F.scaled_dot_product_attention(
                q, torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2), attn_mask=mask
            )
            x = blk.post(x, attn)
        h = self.ln_f(x)  # [B, T, d]
        idx = (path_len - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, self.d_model)
        h_path = h.gather(1, idx).squeeze(1)
        h_out = torch.where((path_len == 0).unsqueeze(1), h_last, h_path)
        return self.readout(h_out, hand, oppmax, osz, usz, otm, mpts, opts)

    @torch.jit.export
    def readout(
        self,
        h: torch.Tensor,  # [B, d] history hidden at the decision position
        hand: torch.Tensor,  # [B, 48] exact (owner's hand)
        oppmax: torch.Tensor,  # [B, 48] thermo
        osz: torch.Tensor,  # [B] float (/16)
        usz: torch.Tensor,  # [B] float (/16)
        otm: torch.Tensor,  # [B] float owner_to_move
        mpts: torch.Tensor,  # [B] float owner series points (/50)
        opts: torch.Tensor,  # [B] float opponent series points (/50)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hh = F.mish(self.layer_a(hand))
        hh = F.mish(self.layer_b_hands(hh))
        hh = F.mish(self.layer_c_player(hh))  # [B, 128]
        oo = F.mish(self.layer_a(oppmax))
        oo = F.mish(self.layer_b_hands(oo))
        oo = F.mish(self.layer_c_opp(oo))  # [B, 96]
        ss = F.mish(
            self.size_embed(torch.stack([osz, usz, otm, mpts, opts], dim=-1))
        )  # [B,16]
        x = torch.cat([h, hh, oo, ss], dim=-1)
        x = self.junction_norm(F.selu(self.junction(x)))
        x = self.trunk(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        behavior = self.behavior_head(x)
        qa = self.qa_head(x)
        return value, policy, behavior, qa

    def forward(
        self,
        tokens: torch.Tensor,
        hist_idx: torch.Tensor,  # [B] int64 readout sequence index per row
        hand: torch.Tensor,
        oppmax: torch.Tensor,
        osz: torch.Tensor,
        usz: torch.Tensor,
        otm: torch.Tensor,
        mpts: torch.Tensor,
        opts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience single-pass path (one readout per sequence row)."""
        H = self.forward_full(tokens)
        B = tokens.shape[0]
        idx = hist_idx.view(B, 1, 1).expand(B, 1, self.d_model)
        h = H.gather(1, idx).squeeze(1)
        return self.readout(h, hand, oppmax, osz, usz, otm, mpts, opts)

    @torch.jit.ignore
    def embedding_param_ids(self) -> set:
        ids = set()
        for layer in (self.layer_a, self.layer_b_hands, self.tok_proj):
            for p in layer.parameters():
                ids.add(id(p))
        ids.add(id(self.pos_emb))
        ids.add(id(self.bos))
        return ids
