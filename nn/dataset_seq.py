"""Per-game sequence datasets for the history-transformer net (model_az_seq).

Loads the parquet TRIPLE emitted per generation by src/datagen/az_selfplay.cpp:

  games  (az_games_genN.parquet):  game_id, first_player, winner,
                                   moves list<int32>   (the token sequence)
  player (az_player_genN.parquet): game_id, turn_idx, hist_idx, owner_seat,
                                   opp_size, our_size, value, hand_0..12,
                                   legal list<int32> (concrete ids),
                                   visit_moves/visit_counts list<int32>
  opp    (az_opp_genN.parquet):    same base + legal (head slots), target_idx

Training is per-GAME: a batch is a set of games; the transformer trunk runs
once per game over its full (padded) token sequence, and every sample of those
games reads out at its hist_idx. `SeqLoader` yields everything needed for one
such step, resident on a single device (cuda == the old --gpu-resident path;
cpu == the debug/fallback path — same code either way).

opp_max is NOT stored: it is recomputed here from the move list + hand
(max_in_deck - hand - cum_discard, clipped at 0 — the exact twin of
az_search/features.h opp_max_counts). Stored hand sizes are cross-checked
against a replay of the move list at load time (catches hist_idx/seat drift).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from nn.dataset import RANK_MAX_COUNTS
from nn.model_az import NUM_MOVES, OPP_HEAD_DIM

MAX_HIST = 64  # > 61, the true maximal game length
_RANK_MAX = np.asarray(RANK_MAX_COUNTS, dtype=np.int64)  # [13]
# 48-dim encoding bit offsets per rank (exact/thermo share the layout).
_RANK_OFF = np.concatenate([[0], np.cumsum(_RANK_MAX)[:-1]])


def _move_cards(path: str = "nn/az_token_cards.json") -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    cards = np.asarray(d["cards"], dtype=np.int64)
    assert cards.shape == (NUM_MOVES, 13)
    return cards


def encode_exact_t(counts: torch.Tensor) -> torch.Tensor:
    """[N, 13] int counts -> [N, 48] exact one-hot, on counts.device."""
    N = counts.shape[0]
    dev = counts.device
    off = torch.as_tensor(_RANK_OFF, device=dev)
    out = torch.zeros(N, 48, dtype=torch.float32, device=dev)
    c = counts.long()
    pos = off.unsqueeze(0) + c - 1  # bit index per rank (invalid where c == 0)
    nz = c > 0
    rows = nz.nonzero(as_tuple=True)[0]
    out[rows, pos[nz]] = 1.0
    return out


def encode_thermo_t(counts: torch.Tensor) -> torch.Tensor:
    """[N, 13] int counts -> [N, 48] thermometer (>=k), on counts.device."""
    N = counts.shape[0]
    dev = counts.device
    rmax = torch.as_tensor(_RANK_MAX, device=dev)
    c = torch.minimum(counts.long(), rmax.unsqueeze(0))  # [N, 13]
    # bit (r, k) set iff c[r] >= k+1; build per-rank ramps once.
    ramp = torch.cat([torch.arange(int(m), device=dev) for m in _RANK_MAX])  # [48]
    rank_of_bit = torch.repeat_interleave(torch.arange(13, device=dev), rmax)  # [48]
    return (c[:, rank_of_bit] > ramp.unsqueeze(0)).float()


def _ragged_np(col) -> tuple[np.ndarray, np.ndarray]:
    off = col.offsets.to_numpy().astype(np.int64)
    flat = col.values.to_numpy().astype(np.int64)
    return flat, off


class SeqData:
    """All generations' (games, player, opp) parquets merged into flat arrays.

    Game rows from later files are re-indexed (game_id collides across files);
    sample `grow` columns point at the merged game-row index. mix_decay < 1
    drops whole GAMES of older files (file 0 = newest = full strength).
    """

    def __init__(
        self,
        triples: list[tuple[str, str, str]],  # [(games, player, opp), ...]
        mix_decay: float = 1.0,
        seed: int = 0,
        validate: bool = True,
    ) -> None:
        mc = _move_cards()
        g_tok, g_len, g_first, g_win, g_pts = [], [], [], [], []
        p_parts, o_parts = [], []
        base = 0
        for i, (gp, pp, op) in enumerate(triples):
            gt = pq.read_table(gp).combine_chunks()
            gid = gt.column("game_id").chunk(0).to_numpy()
            mflat, moff = _ragged_np(gt.column("moves").chunk(0))
            G = len(gid)
            keep = np.ones(G, dtype=bool)
            frac = mix_decay**i
            if frac < 1.0:
                keep &= np.random.default_rng(seed + i).random(G) < frac
            lens = (moff[1:] - moff[:-1])[keep]
            assert lens.max(initial=0) <= MAX_HIST, "game longer than MAX_HIST"
            kept_rows = np.flatnonzero(keep)
            toks = np.zeros((len(kept_rows), MAX_HIST), dtype=np.int16)
            for j, r in enumerate(kept_rows):  # ragged copy; G is small vs samples
                toks[j, : moff[r + 1] - moff[r]] = mflat[moff[r] : moff[r + 1]]
            g_tok.append(toks)
            g_len.append(lens)
            g_first.append(gt.column("first_player").chunk(0).to_numpy()[keep])
            g_win.append(gt.column("winner").chunk(0).to_numpy()[keep])
            cols = set(gt.schema.names)
            if "pts0" in cols and "pts1" in cols:
                pts = np.stack(
                    [
                        gt.column("pts0").chunk(0).to_numpy()[keep],
                        gt.column("pts1").chunk(0).to_numpy()[keep],
                    ],
                    axis=1,
                )  # [G, 2] seat-indexed series points at game start
            else:
                pts = np.zeros((int(keep.sum()), 2), dtype=np.int64)  # legacy data
            g_pts.append(pts)
            # game_id -> merged game row for THIS file's samples
            row_of_gid = {int(g): base + j for j, g in enumerate(gid[kept_rows])}
            p_parts.append(self._load_samples(pp, row_of_gid, player=True))
            o_parts.append(self._load_samples(op, row_of_gid, player=False))
            base += len(kept_rows)

        self.tokens = torch.from_numpy(np.concatenate(g_tok))  # [G, 64] int16
        self.glen = torch.from_numpy(np.concatenate(g_len).astype(np.int64))
        self.first_player = torch.from_numpy(np.concatenate(g_first).astype(np.int64))
        self.winner = torch.from_numpy(np.concatenate(g_win).astype(np.int64))
        self.pts = torch.from_numpy(np.concatenate(g_pts).astype(np.int64))  # [G, 2]
        self.game_w = torch.ones(self.tokens.shape[0], dtype=torch.float32)  # [G]
        self.G = self.tokens.shape[0]
        self.p = self._merge(p_parts, player=True)
        self.o = self._merge(o_parts, player=False)
        self._derive_oppmax(mc)
        if validate:
            self._validate(mc)
        self._sort_and_index()

    # -- loading ------------------------------------------------------------

    @staticmethod
    def _load_samples(path: str, row_of_gid: dict, player: bool) -> dict:
        t = pq.read_table(path).combine_chunks()
        df = t.to_pandas()
        n = len(df)
        grow = df["game_id"].map(lambda g: row_of_gid.get(int(g), -1)).to_numpy()
        keep = grow >= 0  # samples of dropped (mix-decay) games go too
        d = {
            "grow": grow[keep].astype(np.int64),
            "hist": df["hist_idx"].to_numpy()[keep].astype(np.int64),
            "owner": df["owner_seat"].to_numpy()[keep].astype(np.int64),
            "osz": df["opp_size"].to_numpy()[keep].astype(np.float32),
            "usz": df["our_size"].to_numpy()[keep].astype(np.float32),
            "value": df["value"].to_numpy()[keep].astype(np.float32),
            "hand": df[[f"hand_{r}" for r in range(13)]]
            .to_numpy()[keep]
            .astype(np.int8),
        }
        # Auxiliary targets (margin + opponent's exact hand). Absent in legacy
        # parquets; sentinel-filled so the loader can mask them out.
        if "margin" in df.columns:
            d["margin"] = df["margin"].to_numpy()[keep].astype(np.float32)
            d["opp_hand"] = (
                df[[f"opp_hand_{r}" for r in range(13)]].to_numpy()[keep].astype(np.int8)
            )
            d["has_aux"] = np.ones(int(keep.sum()), dtype=np.float32)
        else:
            d["margin"] = np.zeros(int(keep.sum()), dtype=np.float32)
            d["opp_hand"] = np.zeros((int(keep.sum()), 13), dtype=np.int8)
            d["has_aux"] = np.zeros(int(keep.sum()), dtype=np.float32)
        lflat, loff = _ragged_np(t.column("legal").chunk(0))
        d["legal_flat"], d["legal_off"] = _sub_ragged(lflat, loff, keep)
        if player:
            vflat, voff = _ragged_np(t.column("visit_moves").chunk(0))
            cflat, _ = _ragged_np(t.column("visit_counts").chunk(0))
            d["vm_flat"], d["vm_off"] = _sub_ragged(vflat, voff, keep)
            d["vc_flat"], _ = _sub_ragged(cflat, voff, keep)
        else:
            d["target"] = df["target_idx"].to_numpy()[keep].astype(np.int64)
        return d

    @staticmethod
    def _merge(parts: list[dict], player: bool) -> dict:
        out = {}
        for k in parts[0]:
            if k.endswith("_off"):
                continue
            out[k] = np.concatenate([p[k] for p in parts])
        for fk, ok in (("legal_flat", "legal_off"),) + (
            (("vm_flat", "vm_off"),) if player else ()
        ):
            lens = np.concatenate([p[ok][1:] - p[ok][:-1] for p in parts])
            out[ok] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        return out

    # -- derived features & validation ---------------------------------------

    def _cumdisc(self, mc: np.ndarray) -> np.ndarray:
        """[G, MAX_HIST+1, 13] cumulative discard counts before each position."""
        toks = self.tokens.numpy().astype(np.int64)
        seq_cards = mc[toks]  # [G, T, 13]
        cum = np.zeros((self.G, MAX_HIST + 1, 13), dtype=np.int16)
        np.cumsum(seq_cards, axis=1, out=cum[:, 1:])
        return cum

    def _derive_oppmax(self, mc: np.ndarray) -> None:
        cum = self._cumdisc(mc)
        for d in (self.p, self.o):
            cd = cum[d["grow"], d["hist"]]  # [N, 13]
            om = _RANK_MAX[None, :] - d["hand"].astype(np.int64) - cd
            d["oppmax"] = np.clip(om, 0, None).astype(np.int8)

    def _validate(self, mc: np.ndarray) -> None:
        toks = self.tokens.numpy().astype(np.int64)
        totals = mc.sum(-1)[toks]  # [G, T] cards played per move
        ar = np.arange(MAX_HIST)[None, :]
        valid = ar < self.glen.numpy()[:, None]
        totals = totals * valid
        # seat of move k = (first_player + k) % 2  (strict alternation)
        seat0 = (ar % 2) == 0
        fp = self.first_player.numpy()[:, None]
        played_by = np.zeros((self.G, 2, MAX_HIST + 1), dtype=np.int64)
        for s in (0, 1):
            mine = totals * ((seat0 == (fp == s)))
            played_by[:, s, 1:] = np.cumsum(mine, axis=1)
        for name, d in (("player", self.p), ("opp", self.o)):
            assert (d["hand"].sum(1) == d["usz"]).all(), f"{name}: hand!=our_size"
            own_played = played_by[d["grow"], d["owner"], d["hist"]]
            opp_played = played_by[d["grow"], 1 - d["owner"], d["hist"]]
            assert (16 - own_played == d["usz"]).all(), f"{name}: our_size replay"
            assert (16 - opp_played == d["osz"]).all(), f"{name}: opp_size replay"
            assert (d["hist"] <= self.glen.numpy()[d["grow"]]).all()

    def _sort_and_index(self) -> None:
        """Sort samples by game row and build per-game CSR offsets, so a batch
        of games maps to contiguous sample ranges."""
        for d in (self.p, self.o):
            order = np.argsort(d["grow"], kind="stable")
            for k in list(d):
                if k.endswith("_off") or k.endswith("_flat"):
                    continue
                d[k] = d[k][order]
            # reorder ragged fields: one permutation per offsets array, applied
            # to every flat sharing it, THEN the offsets are replaced.
            groups = [("legal_off", ["legal_flat"])]
            if "vm_flat" in d:
                groups.append(("vm_off", ["vm_flat", "vc_flat"]))
            for ok, fks in groups:
                off = d[ok]
                lens = off[1:] - off[:-1]
                idx = np.repeat(off[order], lens[order]) + _within(lens[order])
                for fk in fks:
                    d[fk] = d[fk][idx]
                d[ok] = np.concatenate([[0], np.cumsum(lens[order])]).astype(np.int64)
            d["csr"] = np.searchsorted(d["grow"], np.arange(self.G + 1)).astype(
                np.int64
            )

    def compute_game_weights(self, v_csv_path: str) -> None:
        """Per-game importance weights so the (state-diversified) selfplay sample
        distribution is reweighted toward the chain's NATURAL start-state
        frequencies. Keyed by leader-perspective (a, b) = (pts[first_player],
        pts[other]); w_g = natural_freq(a,b) / empirical_freq(a,b in this
        dataset). Using the dataset's own empirical frequencies makes the weights
        correct for any sampling mixture and recomputable without regenerating
        data. Clipped to [0.1, 10] and normalised to mean 1."""
        natural = np.zeros((50, 50), dtype=np.float64)
        with open(v_csv_path) as f:
            for line in f:
                if not line or line[0] == "#" or line[0] < "0" or line[0] > "9":
                    continue
                a, b, _v, nat = line.strip().split(",")
                natural[int(a), int(b)] = float(nat)
        pts = self.pts.numpy()
        fp = self.first_player.numpy()
        a = pts[np.arange(self.G), fp]
        b = pts[np.arange(self.G), 1 - fp]
        a = np.clip(a, 0, 49)
        b = np.clip(b, 0, 49)
        # Empirical count of games per (a, b) in this dataset.
        counts = np.zeros((50, 50), dtype=np.float64)
        np.add.at(counts, (a, b), 1.0)
        emp = counts[a, b]  # >= 1 (each game contributes its own state)
        w = natural[a, b] / np.maximum(emp, 1.0)
        w = np.clip(w, 0.1, 10.0)
        w = w / max(w.mean(), 1e-8)
        self.game_w = torch.from_numpy(w.astype(np.float32))


def _within(lens: np.ndarray) -> np.ndarray:
    """[sum(lens)] 0..len_i-1 ramps, vectorised."""
    total = int(lens.sum())
    out = np.arange(total)
    out -= np.repeat(np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return out


def _sub_ragged(flat: np.ndarray, off: np.ndarray, keep: np.ndarray):
    lens = off[1:] - off[:-1]
    row_of_pos = np.repeat(np.arange(len(lens)), lens)
    return flat[keep[row_of_pos]], np.concatenate([[0], np.cumsum(lens[keep])]).astype(
        np.int64
    )


# ---------------------------------------------------------------------------
# Device-resident per-game loader.
# ---------------------------------------------------------------------------


class SeqLoader:
    """Iterates over batches of GAMES; per batch yields the padded token block
    plus the player/opp sample rows of those games, densified on device.

    Yields dict with:
      tokens   [b, Tb] int64        glen [b]
      p_*: game (local) index, hist, hand48, oppmax48, osz, usz, otm(=1),
           value, mask [np, NUM_MOVES], policy [np, NUM_MOVES]
      o_*: same sides (otm=0), mask [no, OPP_HEAD_DIM], target [no]
    """

    def __init__(
        self,
        data: SeqData,
        game_rows: np.ndarray,
        batch_games: int,
        device: torch.device,
        shuffle: bool,
    ) -> None:
        self.dev = device
        self.rows = torch.from_numpy(np.asarray(game_rows, dtype=np.int64)).to(device)
        self.B = batch_games
        self.shuffle = shuffle
        self.tokens = data.tokens.to(device)
        self.glen = data.glen.to(device)
        self.pts = data.pts.to(device)          # [G, 2] seat-indexed series points
        self.game_w = data.game_w.to(device)    # [G] importance weight per game
        self.p = {k: _to_dev(v, device) for k, v in data.p.items()}
        self.o = {k: _to_dev(v, device) for k, v in data.o.items()}
        self.n_p = self._count(self.p)  # sample rows in THIS split's games
        self.n_o = self._count(self.o)

    def _count(self, d) -> int:
        csr = d["csr"]
        return int((csr[self.rows + 1] - csr[self.rows]).sum())

    def __len__(self) -> int:
        return (self.rows.numel() + self.B - 1) // self.B

    def _gather_ragged(self, flat, off, idx):
        lens = off[idx + 1] - off[idx]
        rows = torch.repeat_interleave(torch.arange(idx.numel(), device=self.dev), lens)
        out_off = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev), torch.cumsum(lens, 0)]
        )
        within = torch.arange(rows.numel(), device=self.dev) - out_off[rows]
        return rows, flat[off[idx][rows] + within]

    def _sample_rows(self, d, grows):
        """Sample indices (concatenated ranges) + local game index per sample."""
        csr = d["csr"]
        starts, ends = csr[grows], csr[grows + 1]
        lens = ends - starts
        local = torch.repeat_interleave(
            torch.arange(grows.numel(), device=self.dev), lens
        )
        out_off = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.dev), torch.cumsum(lens, 0)]
        )
        within = torch.arange(local.numel(), device=self.dev) - out_off[local]
        sidx = starts[local] + within
        return sidx, local

    def _side_inputs(self, d, sidx):
        hand48 = encode_exact_t(d["hand"][sidx])
        opp48 = encode_thermo_t(d["oppmax"][sidx])
        grow = d["grow"][sidx]
        owner = d["owner"][sidx]
        # Owner-relative series points (the NN input is from the hand owner's POV).
        mpts = self.pts[grow, owner].float() / 50.0
        opts = self.pts[grow, 1 - owner].float() / 50.0
        w = self.game_w[grow]
        return (
            hand48,
            opp48,
            d["osz"][sidx] / 16.0,
            d["usz"][sidx] / 16.0,
            mpts,
            opts,
            w,
        )

    def __iter__(self):
        order = (
            self.rows[torch.randperm(self.rows.numel(), device=self.dev)]
            if self.shuffle
            else self.rows
        )
        for s in range(0, order.numel(), self.B):
            grows = order[s : s + self.B]
            tb = int(self.glen[grows].max().clamp(min=1))
            batch = {
                "tokens": self.tokens[grows, :tb].long(),
                "glen": self.glen[grows],
            }
            # player rows
            sidx, local = self._sample_rows(self.p, grows)
            np_ = sidx.numel()
            mask = torch.zeros(np_, NUM_MOVES, dtype=torch.bool, device=self.dev)
            r, v = self._gather_ragged(self.p["legal_flat"], self.p["legal_off"], sidx)
            mask[r, v] = True
            policy = torch.zeros(np_, NUM_MOVES, dtype=torch.float32, device=self.dev)
            pr, pm = self._gather_ragged(self.p["vm_flat"], self.p["vm_off"], sidx)
            _, pc = self._gather_ragged(self.p["vc_flat"], self.p["vm_off"], sidx)
            policy.index_put_((pr, pm), pc.float(), accumulate=True)
            policy /= policy.sum(1, keepdim=True).clamp_min(1.0)  # value-only -> 0
            h48, o48, osz, usz, mpts, opts, w = self._side_inputs(self.p, sidx)
            batch.update(
                p_local=local,
                p_hist=self.p["hist"][sidx],
                p_hand=h48,
                p_opp=o48,
                p_osz=osz,
                p_usz=usz,
                p_mpts=mpts,
                p_opts=opts,
                p_w=w,
                p_value=self.p["value"][sidx],
                p_mask=mask,
                p_policy=policy,
                p_margin=self.p["margin"][sidx] / 16.0,
                p_opp_hand=encode_thermo_t(self.p["opp_hand"][sidx]),
                p_haux=self.p["has_aux"][sidx],
                # az_ii raw targets: AR opp-hand decode + bomb + outcome bucket.
                p_opp_hand_cnt=self.p["opp_hand"][sidx],
                p_oppmax_cnt=self.p["oppmax"][sidx],
                p_margin_raw=self.p["margin"][sidx],
            )
            # opp rows
            sidx, local = self._sample_rows(self.o, grows)
            no = sidx.numel()
            mask = torch.zeros(no, OPP_HEAD_DIM, dtype=torch.bool, device=self.dev)
            r, v = self._gather_ragged(self.o["legal_flat"], self.o["legal_off"], sidx)
            mask[r, v] = True
            target = self.o["target"][sidx]
            mask[torch.arange(no, device=self.dev), target] = True
            h48, o48, osz, usz, mpts, opts, w = self._side_inputs(self.o, sidx)
            batch.update(
                o_local=local,
                o_hist=self.o["hist"][sidx],
                o_hand=h48,
                o_opp=o48,
                o_osz=osz,
                o_usz=usz,
                o_mpts=mpts,
                o_opts=opts,
                o_w=w,
                o_value=self.o["value"][sidx],
                o_mask=mask,
                o_target=target,
                o_margin=self.o["margin"][sidx] / 16.0,
                o_opp_hand=encode_thermo_t(self.o["opp_hand"][sidx]),
                o_haux=self.o["has_aux"][sidx],
                o_opp_hand_cnt=self.o["opp_hand"][sidx],
                o_oppmax_cnt=self.o["oppmax"][sidx],
                o_margin_raw=self.o["margin"][sidx],
            )
            yield batch


def _to_dev(v, device):
    if isinstance(v, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(v)).to(device)
    return v.to(device)


def make_seq_split(
    triples: list[tuple[str, str, str]],
    batch_games: int,
    device: torch.device,
    val_frac: float = 0.1,
    seed: int = 0,
    mix_decay: float = 1.0,
    validate: bool = True,
    series_v: str | None = None,
) -> tuple[SeqLoader, SeqLoader]:
    """(train_loader, val_loader) split by GAME over the merged dataset. When
    series_v is given, per-game natural-frequency importance weights are attached
    (else all weights are 1)."""
    data = SeqData(triples, mix_decay=mix_decay, seed=seed, validate=validate)
    if series_v:
        data.compute_game_weights(series_v)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.G)
    n_val = max(1, int(data.G * val_frac))
    val_rows, train_rows = perm[:n_val], perm[n_val:]
    return (
        SeqLoader(data, train_rows, batch_games, device, shuffle=True),
        SeqLoader(data, val_rows, batch_games, device, shuffle=False),
    )
