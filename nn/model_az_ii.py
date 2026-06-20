"""Imperfect-information net for the `az_ii` player — PI-distillation line.

Restructures the unified history-transformer (model_az_seq) into a clean
belief / play split (all in ONE nn.Module, joint-trained):

  1. BELIEF transformer — the causal trunk over the public move history
     (reused from model_az_seq). Its hidden at the decision position is the
     belief embedding `e`. Two belief outputs read off `e` (thermo-conditioned):
       * AR opp-hand head — autoregressive over rank counts, highest rank first
         ([2, A, K, ... , 3]); each step a masked softmax over counts 0..4,
         bounded by min(thermo upper bound, remaining budget). Teacher-forced in
         training; ancestral-sampled for determinization / belief stats.
       * P(bomb) — sigmoid, opponent holds any 4-of-a-kind.

  2. MAIN readout (non-recurrent) — consumes `e` as a SIDE input alongside our
     exact hand, the opponent thermo, sizes, owner-to-move, and series points.
     Heads:
       * value    [B]        sigmoid, P(hand-owner wins the series)
       * policy   [B, 138]   factored player-policy logits (NLL on chosen move)
       * behavior [B, 458]   opponent behavior logits (NLL on chosen opp move)
     Aux (training-only, shared-trunk regularisers):
       * outcome  [B, 32]    signed terminal-margin bucket (win1..16 / lose1..16)

`e` feeds BOTH the belief heads and the main junction, so the AR/bomb losses
pull `e` toward an accurate belief while the play losses pull it toward what is
useful for play — `e` ends up belief-rich AND directly consumable. There is no
qa head and no opponent hint: the az_ii search fully expands opponent moves and
backs up node value = sum_a behavior_prior(a) * V(child_a).

The TorchScript C++ surface (encode_prefix / forward_leaf for the KV-cache
search) is finalized in Phase 3; this module currently exposes the training
forward (forward_full + readout_train) and is eager-only.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.dataset import RANK_MAX_COUNTS
from nn.model import ENCODING_DIM, ResBlock, _init_selu, _init_swish
from nn.model_az import (
    OPP_HEAD_DIM,
    PLAYER_HEAD_DIM,
    PerMoveValueHead,
    PolicyHead,
    ValueHead,
)
from nn.model_az_seq import NUM_MOVES, SEQ_CAP, _SeqBlock, load_token_feats

# Autoregressive opp-hand decode order: highest impact rank first. Rank indices
# are [3,4,...,K,A,2] = 0..12, so 2=12, A=11, then K..3. The lowest ranks become
# forced once the remaining-card budget is spent.
AR_ORDER: List[int] = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
MAXC = 5  # per-rank count classes: 0,1,2,3,4
OUTCOME_DIM = 32  # win-by-1..16 -> 0..15 ; lose-by-1..16 -> 16..31
_RANK_MAX = torch.tensor(RANK_MAX_COUNTS, dtype=torch.float32)  # [13]


def margin_to_bucket(margin: torch.Tensor) -> torch.Tensor:
    """Signed loser_cards (won:+1..16, lost:-1..16) -> outcome bucket 0..31."""
    m = margin.long()
    win = m > 0
    return torch.where(win, m - 1, 16 - m - 1).clamp(0, OUTCOME_DIM - 1)


class ARHand(nn.Module):
    """Autoregressive opponent-hand decoder. A weight-shared GRU steps over the
    13 ranks in AR_ORDER, seeded from the belief vector; each step emits a masked
    softmax over counts 0..4. Eager-only (training NLL + sampling)."""

    def __init__(self, belief_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden
        self.seed = nn.Linear(belief_dim, hidden)
        # step input: prev-count one-hot (5) + remaining/16 + thermo_bound/4 + rank one-hot (13)
        self.cell = nn.GRUCell(MAXC + 1 + 1 + 13, hidden)
        self.out = nn.Linear(hidden, MAXC)

    def _step_inputs(self, prev_oh, remaining, thermo_r, rank_oh):
        return torch.cat(
            [prev_oh, (remaining / 16.0).unsqueeze(1),
             (thermo_r / 4.0).unsqueeze(1), rank_oh], dim=-1
        )

    def _suffix_cap(self, thermo: torch.Tensor) -> torch.Tensor:
        """[N, len(AR_ORDER)] capacity of ranks AFTER each decode position — the
        budget the still-to-come ranks can still absorb. Used for the lower-bound
        mask that forces the remainder onto the tail ranks."""
        order = torch.tensor([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                             dtype=torch.long, device=thermo.device)
        thermo_ord = thermo.index_select(1, order).float()  # [N, R] in decode order
        incl = torch.flip(torch.cumsum(torch.flip(thermo_ord, [1]), 1), [1])
        return incl - thermo_ord  # exclusive suffix (ranks strictly after pos)

    def _masked_logp(self, h, karange, remaining, bound, cap_after):
        """Log-softmax over counts 0..4 with both bounds: lo = max(0, remaining -
        cap_after) (implied minimum), hi = bound. Never masks a feasible count."""
        logits = self.out(h)  # [N, 5]
        lo = (remaining - cap_after).clamp(min=0.0)
        valid = (karange.unsqueeze(0) >= lo.unsqueeze(1)) & (
            karange.unsqueeze(0) <= bound.unsqueeze(1)
        )
        return F.log_softmax(logits.masked_fill(~valid, -1e9), dim=-1)

    @torch.jit.ignore
    def nll(self, belief: torch.Tensor, thermo: torch.Tensor,
            true_counts: torch.Tensor) -> torch.Tensor:
        """Per-row summed NLL of the true opponent hand. belief [N, bd];
        thermo [N, 13] integer upper bounds; true_counts [N, 13] integer."""
        N = belief.shape[0]
        dev = belief.device
        h = torch.tanh(self.seed(belief))
        opp_size = true_counts.sum(1).float()
        cap = self._suffix_cap(thermo)
        decided = torch.zeros(N, device=dev)
        prev = torch.zeros(N, MAXC, device=dev)
        eye13 = torch.eye(13, device=dev)
        karange = torch.arange(MAXC, device=dev).float()
        total = torch.zeros(N, device=dev)
        for i, r in enumerate(AR_ORDER):
            remaining = opp_size - decided
            bound = torch.minimum(thermo[:, r].float(), remaining)
            rank_oh = eye13[r].expand(N, 13)
            h = self.cell(self._step_inputs(prev, remaining, thermo[:, r].float(), rank_oh), h)
            logp = self._masked_logp(h, karange, remaining, bound, cap[:, i])
            tc = true_counts[:, r].long()
            total = total - logp.gather(1, tc.unsqueeze(1)).squeeze(1)
            decided = decided + tc.float()
            prev = F.one_hot(tc, MAXC).float()
        return total  # [N]

    @torch.jit.ignore
    @torch.no_grad()
    def sample(self, belief: torch.Tensor, thermo: torch.Tensor,
               opp_size: torch.Tensor, generator=None) -> torch.Tensor:
        """Ancestral-sample opponent hands. Returns [N, 13] integer counts that
        sum to opp_size and respect the thermo bounds."""
        N = belief.shape[0]
        dev = belief.device
        h = torch.tanh(self.seed(belief))
        cap = self._suffix_cap(thermo)
        decided = torch.zeros(N, device=dev)
        prev = torch.zeros(N, MAXC, device=dev)
        eye13 = torch.eye(13, device=dev)
        karange = torch.arange(MAXC, device=dev).float()
        out = torch.zeros(N, 13, dtype=torch.long, device=dev)
        for i, r in enumerate(AR_ORDER):
            remaining = opp_size.float() - decided
            bound = torch.minimum(thermo[:, r].float(), remaining)
            rank_oh = eye13[r].expand(N, 13)
            h = self.cell(self._step_inputs(prev, remaining, thermo[:, r].float(), rank_oh), h)
            logp = self._masked_logp(h, karange, remaining, bound, cap[:, i])
            c = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
            out[:, r] = c
            decided = decided + c.float()
            prev = F.one_hot(c, MAXC).float()
        return out

    def sample_n(self, belief: torch.Tensor, thermo: torch.Tensor,
                 opp_size: torch.Tensor) -> torch.Tensor:
        """Scriptable ancestral sampler (no generator arg). belief [N, bd],
        thermo [N, 13] float upper bounds, opp_size [N] float -> [N, 13] long
        counts that sum to opp_size and respect the thermo bounds."""
        maxc = 5
        N = belief.shape[0]
        dev = belief.device
        h = torch.tanh(self.seed(belief))
        cap = self._suffix_cap(thermo)
        decided = torch.zeros(N, device=dev)
        prev = torch.zeros(N, maxc, device=dev)
        eye13 = torch.eye(13, device=dev)
        karange = torch.arange(maxc, device=dev).float()
        out = torch.zeros(N, 13, dtype=torch.long, device=dev)
        order = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        for i, r in enumerate(order):
            remaining = opp_size - decided
            bound = torch.minimum(thermo[:, r], remaining)
            rank_oh = eye13[r].expand(N, 13)
            h = self.cell(self._step_inputs(prev, remaining, thermo[:, r], rank_oh), h)
            logp = self._masked_logp(h, karange, remaining, bound, cap[:, i])
            c = torch.multinomial(logp.exp(), 1).squeeze(1)
            out[:, r] = c
            decided = decided + c.float()
            prev = F.one_hot(c, maxc).float()
        return out


class Big2NetII(nn.Module):
    def __init__(
        self,
        token_feats: torch.Tensor,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 256,
        seq_cap: int = SEQ_CAP,
        ar_hidden: int = 128,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_cap = seq_cap
        self.neg_mask = -1e9

        assert token_feats.shape == (NUM_MOVES, ENCODING_DIM)
        self.register_buffer("token_feats", token_feats.float())

        # Belief transformer trunk (same as Big2NetSeqAZ).
        self.tok_proj = nn.Linear(ENCODING_DIM, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(seq_cap, d_model))
        self.bos = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        self.blocks = nn.ModuleList(
            [_SeqBlock(d_model, n_heads, d_ff, n_layers) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Card-literacy stacks for the side inputs.
        self.layer_a = _init_swish(nn.Linear(ENCODING_DIM, 64))
        self.layer_b_hands = _init_swish(nn.Linear(64, 64))
        self.layer_c_player = _init_swish(nn.Linear(64, 128))
        self.layer_c_opp = _init_swish(nn.Linear(64, 96))
        # (opp_size/16, our_size/16, owner_to_move, my_pts/50, opp_pts/50) -> 16.
        self.size_embed = _init_swish(nn.Linear(5, 16))

        # Junction: [belief d_model, hand128, opp96, size16] -> 256.
        self.junction = _init_selu(nn.Linear(d_model + 128 + 96 + 16, 256))
        self.junction_norm = nn.LayerNorm(256)
        self.trunk = nn.Sequential(ResBlock(256), ResBlock(256))

        self.value_head = ValueHead(256)
        self.policy_head = PolicyHead(256, PLAYER_HEAD_DIM)
        self.behavior_head = PolicyHead(256, OPP_HEAD_DIM)
        # qa: per-opp-move value (P(searcher wins after opp move a)). Kept so the
        # scripted surface matches model_az_seq and az_ii drops into the existing
        # az_search MCTS / NNEvaluator unchanged (opp-to-move nodes have no
        # trained scalar value otherwise). Full-expansion-without-qa is a later
        # search optimisation, not needed to play az_ii as an AZ model.
        self.qa_head = PerMoveValueHead(256, OPP_HEAD_DIM)
        self.outcome_head = nn.Linear(256, OUTCOME_DIM)  # aux: signed margin bucket

        # Belief heads off e (+ opp-thermo embedding). The AR loss + P(bomb) shape
        # the transformer hidden into a usable belief; the main junction reads the
        # same e, so play losses co-shape it.
        belief_dim = d_model + 96
        self.ar = ARHand(belief_dim, ar_hidden)
        self.bomb_head = nn.Linear(belief_dim, 1)

    # -- trunk --------------------------------------------------------------

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        tok = self.tok_proj(F.embedding(tokens, self.token_feats))
        bos = self.bos.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d_model)
        x = torch.cat([bos, tok], dim=1)
        return x + self.pos_emb[: T + 1].unsqueeze(0)

    @torch.jit.export
    def config(self) -> List[int]:
        return [self.n_layers, self.n_heads, self.head_dim, self.d_model, self.seq_cap]

    @torch.jit.export
    def forward_full(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, T] int64 -> hidden [B, T+1, d] (causal)."""
        x = self._embed(tokens)
        S = x.shape[1]
        causal = torch.full((S, S), self.neg_mask, dtype=x.dtype, device=x.device)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            q, k, v = blk.qkv_proj(x)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            x = blk.post(x, attn)
        return self.ln_f(x)

    # -- KV-cache search surface (matches model_az_seq; used by az_search's
    #    NNEvaluator so az_ii drops into the existing MCTS unchanged) ----------

    @torch.jit.export
    def encode_prefix(
        self, tokens: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._embed(tokens)
        B, S, _ = x.shape
        causal = torch.full((S, S), self.neg_mask, dtype=x.dtype, device=x.device)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
        kvs: List[torch.Tensor] = []
        for blk in self.blocks:
            q, k, v = blk.qkv_proj(x)
            kvs.append(torch.stack([k, v], dim=1))
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            x = blk.post(x, attn)
        kv = torch.stack(kvs, dim=1)
        h = self.ln_f(x)
        idx = lens.view(B, 1, 1).expand(B, 1, self.d_model)
        h_last = h.gather(1, idx).squeeze(1)
        return kv, h_last

    @torch.jit.export
    def forward_leaf(
        self,
        kv: torch.Tensor,
        plen: torch.Tensor,
        h_last: torch.Tensor,
        path: torch.Tensor,
        path_len: torch.Tensor,
        hand: torch.Tensor,
        oppmax: torch.Tensor,
        osz: torch.Tensor,
        usz: torch.Tensor,
        otm: torch.Tensor,
        mpts: torch.Tensor,
        opts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kv = kv.to(h_last.dtype)
        B, T = path.shape
        P1 = kv.shape[4]
        tok = self.tok_proj(F.embedding(path, self.token_feats))
        pos = plen.unsqueeze(1) + torch.arange(T, device=path.device).unsqueeze(0)
        pos = pos.clamp(max=self.seq_cap - 1)
        x = tok + F.embedding(pos, self.pos_emb)
        ar_p = torch.arange(P1, device=path.device)
        mask_pre = torch.where(
            ar_p.unsqueeze(0) < plen.unsqueeze(1),
            torch.zeros(B, P1, dtype=x.dtype, device=x.device),
            torch.full((B, P1), self.neg_mask, dtype=x.dtype, device=x.device),
        )
        mask_pre = mask_pre.unsqueeze(1).unsqueeze(2).expand(B, 1, T, P1)
        ar_t = torch.arange(T, device=path.device)
        causal = torch.where(
            ar_t.unsqueeze(1) >= ar_t.unsqueeze(0),
            torch.zeros(T, T, dtype=x.dtype, device=x.device),
            torch.full((T, T), self.neg_mask, dtype=x.dtype, device=x.device),
        )
        mask = torch.cat(
            [mask_pre, causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)], dim=3
        )
        for i, blk in enumerate(self.blocks):
            q, k, v = blk.qkv_proj(x)
            pk = kv[:, i, 0]
            pv = kv[:, i, 1]
            attn = F.scaled_dot_product_attention(
                q, torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2), attn_mask=mask
            )
            x = blk.post(x, attn)
        h = self.ln_f(x)
        idx = (path_len - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, self.d_model)
        h_path = h.gather(1, idx).squeeze(1)
        h_out = torch.where((path_len == 0).unsqueeze(1), h_last, h_path)
        return self.readout(h_out, hand, oppmax, osz, usz, otm, mpts, opts)

    @torch.jit.export
    def readout(
        self, h, hand, oppmax, osz, usz, otm, mpts, opts
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """4-head scripted readout for the search: value, policy, behavior, qa."""
        x, _ = self._features(h, hand, oppmax, osz, usz, otm, mpts, opts)
        return (
            self.value_head(x),
            self.policy_head(x),
            self.behavior_head(x),
            self.qa_head(x),
        )

    # -- readout ------------------------------------------------------------

    def _features(self, h, hand, oppmax, osz, usz, otm, mpts, opts):
        """Returns (trunk_x[256], belief[belief_dim]). h = belief embedding e."""
        hh = F.mish(self.layer_a(hand))
        hh = F.mish(self.layer_b_hands(hh))
        hh = F.mish(self.layer_c_player(hh))  # [N, 128]
        oo = F.mish(self.layer_a(oppmax))
        oo = F.mish(self.layer_b_hands(oo))
        oo = F.mish(self.layer_c_opp(oo))  # [N, 96]
        ss = F.mish(self.size_embed(torch.stack([osz, usz, otm, mpts, opts], dim=-1)))
        x = torch.cat([h, hh, oo, ss], dim=-1)
        x = self.junction_norm(F.selu(self.junction(x)))
        x = self.trunk(x)
        belief = torch.cat([h, oo], dim=-1)
        return x, belief

    @torch.jit.export
    def forward(
        self,
        tokens: torch.Tensor,    # [B, T] int64 move-id history
        hist_idx: torch.Tensor,  # [B] int64 readout position (moves before decision)
        hand: torch.Tensor,      # [B, 48] exact
        oppmax: torch.Tensor,    # [B, 48] thermo upper bound
        osz: torch.Tensor,       # [B] float (/16)
        usz: torch.Tensor,       # [B] float (/16)
        otm: torch.Tensor,       # [B] float owner-to-move
        mpts: torch.Tensor,      # [B] float owner series pts (/50)
        opts: torch.Tensor,      # [B] float opp series pts (/50)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Play/eval surface for C++: belief transformer over `tokens`, read off
        at `hist_idx`, then the main heads. (value, policy, behavior); the AR /
        bomb / outcome heads are training-only and not on this path."""
        H = self.forward_full(tokens)
        B = tokens.shape[0]
        idx = hist_idx.view(B, 1, 1).expand(B, 1, self.d_model)
        h = H.gather(1, idx).squeeze(1)
        x, _ = self._features(h, hand, oppmax, osz, usz, otm, mpts, opts)
        return self.value_head(x), self.policy_head(x), self.behavior_head(x)

    @torch.jit.export
    def sample_opp(
        self,
        tokens: torch.Tensor,    # [B, T] int64 move-id history
        hist_idx: torch.Tensor,  # [B] int64 belief readout position
        oppmax: torch.Tensor,    # [B, 48] thermo encoding (for the belief vector)
        thermo13: torch.Tensor,  # [B, 13] float per-rank upper bounds (AR mask)
        opp_size: torch.Tensor,  # [B] float opponent hand size
        n: int,                  # samples per batch row
    ) -> torch.Tensor:
        """Determinization sampler for the C++ PIMC player: belief transformer
        over `tokens`, read off at `hist_idx`, then AR-sample `n` opponent hands
        per row. Returns [B, n, 13] long counts summing to opp_size."""
        H = self.forward_full(tokens)
        B = tokens.shape[0]
        idx = hist_idx.view(B, 1, 1).expand(B, 1, self.d_model)
        e = H.gather(1, idx).squeeze(1)  # [B, d_model]
        oo = F.mish(self.layer_a(oppmax))
        oo = F.mish(self.layer_b_hands(oo))
        oo = F.mish(self.layer_c_opp(oo))  # [B, 96]
        belief = torch.cat([e, oo], dim=-1)  # [B, belief_dim]
        belief = belief.repeat_interleave(n, dim=0)
        th = thermo13.repeat_interleave(n, dim=0)
        osz = opp_size.repeat_interleave(n, dim=0)
        hands = self.ar.sample_n(belief, th, osz)  # [B*n, 13] long
        return hands.view(B, n, 13)

    @torch.jit.ignore
    def readout_train(self, h, hand, oppmax, osz, usz, otm, mpts, opts,
                      thermo_counts, opp_hand_counts):
        """Eager training readout. Returns
        (value, policy, behavior, qa, outcome_logits, bomb_logit, ar_nll).
        thermo_counts / opp_hand_counts are integer [N, 13] (AR bounds + target)."""
        x, belief = self._features(h, hand, oppmax, osz, usz, otm, mpts, opts)
        value = self.value_head(x)
        policy = self.policy_head(x)
        behavior = self.behavior_head(x)
        qa = self.qa_head(x)
        outcome = self.outcome_head(x)
        bomb = self.bomb_head(belief).squeeze(-1)
        ar_nll = self.ar.nll(belief, thermo_counts, opp_hand_counts)
        return value, policy, behavior, qa, outcome, bomb, ar_nll

    def embedding_param_ids(self) -> set:
        ids = set()
        for layer in (self.layer_a, self.layer_b_hands, self.tok_proj):
            for p in layer.parameters():
                ids.add(id(p))
        ids.add(id(self.pos_emb))
        ids.add(id(self.bos))
        return ids
