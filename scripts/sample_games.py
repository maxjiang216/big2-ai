"""Pretty-print sample games from a self-play parquet file with per-turn decision blocks.

Usage:
    uv run python scripts/sample_games.py data/gen9.parquet --games 3 --out samples/gen9.txt
    uv run python scripts/sample_games.py data/gen9.parquet --model models/gen9.pt --games 3 --out samples/gen9.txt
"""

from __future__ import annotations
import argparse, re, numpy as np, pandas as pd
from pathlib import Path

# ── card / rank constants ──────────────────────────────────────────────────
R = ["3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K", "A", "2"]
RM = [
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    3,
    1,
]  # max copies per rank (deck per-player max)

# ── move-ID constants (mirror src/core/util.h) ─────────────────────────────
kPASS = 0
kSINGLE_START = 1
kDOUBLE_START = 14  # +13
kTRIPLE_START = 26  # +12
kFULL_HOUSE_START = 37  # +11
kBOMB_START = 169  # +132
kSTRAIGHT5_START = 325  # +156
LEGAL_MOVES_SIZE = 468

_SRC = Path(__file__).parent.parent / "src" / "core"

# ── lazy-loaded tables ─────────────────────────────────────────────────────
_MTCcache: list[list[int]] | None = None
_KHTcache: list[float] | None = None


def _mtc() -> list[list[int]]:
    global _MTCcache
    if _MTCcache is None:
        text = (_SRC / "move_to_cards.inc").read_text()
        result = []
        for m in re.finditer(r"\{([^{}]+)\}", text):
            nums = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
            if len(nums) == 14:
                result.append(nums)
        assert len(result) == LEGAL_MOVES_SIZE
        _MTCcache = result
    return _MTCcache


def _kht() -> list[float]:
    global _KHTcache
    if _KHTcache is None:
        text = (_SRC / "hint_table.h").read_text()
        nums = [float(x) for x in re.findall(r"([\d]+\.[\d]+(?:e[+-]?\d+)?)f", text)]
        assert len(nums) == LEGAL_MOVES_SIZE
        _KHTcache = nums
    return _KHTcache


# ── encoding helpers ───────────────────────────────────────────────────────
def hbits(at1, at2, at3, at4):
    return [
        ((at1 >> r & 1) + ((at2 >> r & 1)) + ((at3 >> r & 1)) + ((at4 >> r & 1)))
        for r in range(13)
    ]


def encode_exact(counts):
    out = []
    for r, mx in enumerate(RM):
        c = counts[r]
        out.extend([(1.0 if c == k else 0.0) for k in range(1, mx + 1)])
    return out


def encode_thermo(counts):
    out = []
    for r, mx in enumerate(RM):
        c = min(counts[r], mx)
        out.extend([(1.0 if c >= k else 0.0) for k in range(1, mx + 1)])
    return out


# ── move formatting ────────────────────────────────────────────────────────
def fmt_hand(c: list[int]) -> str:
    return "".join(R[r] * c[r] for r in range(13))


def fmt_move(mc: list[int] | None, is_pass: bool) -> str:
    if is_pass or mc is None:
        return "pass"
    cards = "".join(R[r] * mc[r] for r in range(13) if mc[r])
    total = sum(mc)
    nz = [(r, c) for r, c in enumerate(mc) if c]
    if total == 1:
        t = "single"
    elif total == 2 and len(nz) == 1:
        t = "pair"
    elif total == 3 and len(nz) == 1:
        t = "trips"
    elif total == 4 and len(nz) == 1 and nz[0][1] == 4:
        t = "quad"
    elif total == 5 and len(nz) == 2 and any(c == 3 for _, c in nz):
        t = "FH"
    elif total >= 5 and len(nz) == total and all(c == 1 for _, c in nz):
        t = "str"
    elif total >= 4 and any(c == 4 for _, c in nz):
        extra = total - 4
        t = f"bomb+{extra}" if extra else "bomb"
    elif total == 3 and len(nz) == 1:
        t = "3A"  # 3 aces = bare bomb
    else:
        t = f"{total}c"
    return f"{t}({cards})"


# ── move classification ────────────────────────────────────────────────────
def move_combo_rank(mid: int) -> tuple[str, int]:
    """(combo_type, rank) – rank is the dominance rank for comparison."""
    mtc = _mtc()
    if mid == kPASS:
        return ("pass", -1)
    if mid < kDOUBLE_START:
        return ("single", mid - kSINGLE_START)
    if mid < kTRIPLE_START:
        return ("pair", mid - kDOUBLE_START)
    if mid < kFULL_HOUSE_START:
        return ("triple", mid - kTRIPLE_START)
    cards = mtc[mid][:13]
    if mid < kBOMB_START:
        triple_r = next(r for r, c in enumerate(cards) if c == 3)
        return ("fh", triple_r)
    if mid < kSTRAIGHT5_START:
        max_c = max(cards)
        base_r = next(r for r, c in enumerate(cards) if c == max_c)
        return ("bomb", base_r)
    # Straight: level = cards per rank slot (1/2/3), rank = highest non-2-low rank
    max_c = max(cards)
    nz = [r for r, c in enumerate(cards) if c > 0]
    # 2-as-low: straight contains rank 12 ("2") AND rank 0 ("3")
    if 12 in nz and 0 in nz:
        rank = max(r for r in nz if r != 12)
    else:
        rank = max(nz)
    return (f"str{max_c}", rank)


def beats_move(mid: int, last_mid: int) -> bool:
    if mid == kPASS:
        return False
    tc_n, rc_n = move_combo_rank(mid)
    tc_l, rc_l = move_combo_rank(last_mid)
    if tc_n == "bomb" and tc_l != "bomb":
        return True
    if tc_n == "bomb" and tc_l == "bomb":
        return rc_n > rc_l
    if tc_n != tc_l:
        return False
    return rc_n > rc_l


def can_play_move(hand: list[int], mid: int) -> bool:
    mtc = _mtc()
    return all(hand[r] >= mtc[mid][r] for r in range(13))


def compute_legal_moves_py(hand: list[int], last_mid: int) -> list[int]:
    if last_mid == kPASS:
        return [m for m in range(1, LEGAL_MOVES_SIZE) if can_play_move(hand, m)]
    return [kPASS] + [
        m
        for m in range(1, LEGAL_MOVES_SIZE)
        if can_play_move(hand, m) and beats_move(m, last_mid)
    ]


def find_move_id(mc: list[int]) -> int:
    """Find move_id for given card-count array; returns kPASS if not found."""
    if sum(mc) == 0:
        return kPASS
    mtc = _mtc()
    mc_l = list(mc)
    for mid in range(1, LEGAL_MOVES_SIZE):
        if mtc[mid][:13] == mc_l:
            return mid
    return kPASS


# ── model scoring ──────────────────────────────────────────────────────────
def score_candidates(model, device, hand_before, opp_counts, legal_mids, opp_hand):
    """Batch-score all legal_mids. Returns [(mid, vi, vn, pt, w)] sorted by w↓."""
    import torch

    mtc = _mtc()
    kht = _kht()
    n = len(legal_mids)
    if n == 0:
        return []

    # Pre-compute flag (opp definitely can't respond)
    def opp_cant(move_id):
        if move_id == kPASS or opp_hand is None:
            return False
        return len(compute_legal_moves_py(opp_hand, move_id)) == 1

    hand_arr = np.zeros((n, 48), dtype=np.float32)
    opp_arr = np.zeros((n, 48), dtype=np.float32)
    move_arr = np.zeros((n, 48), dtype=np.float32)
    hint_arr = np.zeros(n, dtype=np.float32)
    flag_arr = np.zeros(n, dtype=bool)
    pass_arr = np.zeros(n, dtype=bool)

    for i, mid in enumerate(legal_mids):
        mc = mtc[mid][:13] if mid != kPASS else [0] * 13
        hand_after = [hand_before[r] - mc[r] for r in range(13)]
        is_flag = opp_cant(mid)
        hint_val = 0.0 if mid == kPASS else (1.0 if is_flag else 1.0 - kht[mid])
        hand_arr[i] = encode_exact(hand_after)
        opp_arr[i] = encode_thermo(opp_counts)
        move_arr[i] = encode_exact(mc)
        hint_arr[i] = hint_val
        flag_arr[i] = is_flag
        pass_arr[i] = mid == kPASS

    with torch.no_grad():
        vi_t, vn_t, pt_t = model.forward(
            torch.tensor(hand_arr).to(device),
            torch.tensor(opp_arr).to(device),
            torch.tensor(move_arr).to(device),
            torch.tensor(hint_arr).to(device),
            torch.tensor(flag_arr).to(device),
            torch.tensor(pass_arr).to(device),
        )

    results = []
    for i, mid in enumerate(legal_mids):
        vi = vi_t[i].item()
        vn = vn_t[i].item()
        pt = pt_t[i].item()
        w = pt * vi + (1 - pt) * vn
        results.append((mid, vi, vn, pt, w))
    results.sort(key=lambda x: -x[4])
    return results


# ── turn-block printer ─────────────────────────────────────────────────────
_COL = 17  # move string column width


def _score_str(vi, vn, pt, w):
    return f"pt={pt:.2f} vi={vi:.2f} vn={vn:.2f} w={w:.2f}"


def print_turn(
    f,
    trick,
    p,
    hand_before,
    opp_hand,
    opp_counts,
    last_trick_mid,
    last_trick_cards,
    chosen_mid,
    chosen_cards,
    model,
    device,
    forced: bool = False,
    tablebase: bool = False,
    implicit: bool = False,
):
    n_before = sum(hand_before) if hand_before is not None else None
    n_str = f"{n_before:2d}" if n_before is not None else " ?"
    ctx = (
        "Initiative"
        if last_trick_mid == kPASS
        else f"vs {fmt_move(last_trick_cards, False)}"
    )
    tb_tag = "  [tb:opp=1]" if tablebase else ""

    f.write(f"\n  ── T{trick}  P{p}  {n_str} cards  {ctx}{tb_tag} ─\n")
    hand_str = fmt_hand(hand_before) if hand_before is not None else "?"
    f.write(f"  Hand [P{p}]: {hand_str}\n")
    opp_str = fmt_hand(opp_hand) if opp_hand is not None else "?"
    f.write(f"  Opp  [P{1-p}]: {opp_str}\n")

    is_pass_chosen = chosen_mid == kPASS

    if implicit:
        f.write(f"  ** pass  [implicit]\n")
        return

    if forced:
        f.write(f"  ** pass  [forced]\n")
        return

    if model is None:
        f.write(f"  ** {fmt_move(chosen_cards, is_pass_chosen)}\n")
        return

    if hand_before is None:
        f.write(f"  ** {fmt_move(chosen_cards, is_pass_chosen)}\n")
        return

    legal = compute_legal_moves_py(hand_before, last_trick_mid)

    # Single-move: forced (should have been caught above, but guard)
    if len(legal) == 1:
        f.write(f"  ** {fmt_move(chosen_cards, is_pass_chosen)}  [forced]\n")
        return

    scored = score_candidates(model, device, hand_before, opp_counts, legal, opp_hand)

    top2 = scored[:2]
    chosen_in_top2 = any(mid == chosen_mid for mid, *_ in top2)

    for rank_idx, (mid, vi, vn, pt, w) in enumerate(top2):
        marker = "→" if mid == chosen_mid else " "
        label = "#1" if rank_idx == 0 else "#2"
        mc_i = _mtc()[mid][:13] if mid != kPASS else [0] * 13
        f.write(
            f"  {label}{marker} {fmt_move(mc_i, mid==kPASS):<{_COL}}  {_score_str(vi,vn,pt,w)}\n"
        )

    if not chosen_in_top2:
        for mid, vi, vn, pt, w in scored:
            if mid == chosen_mid:
                mc_i = _mtc()[mid][:13] if mid != kPASS else [0] * 13
                f.write(
                    f"  ** {fmt_move(mc_i, mid==kPASS):<{_COL}}  {_score_str(vi,vn,pt,w)}  [played]\n"
                )
                break


# ── game printer ───────────────────────────────────────────────────────────
def print_game(gdf, f, model=None, device=None):
    gdf = gdf.sort_values("turn_idx").reset_index(drop=True)
    winner = gdf["winner"].iloc[0]
    f.write(f"  {'─'*72}\n")

    exact_hand = [None, None]  # last known hand per player (after their play)
    last_mid = kPASS  # last non-pass move_id (0 = initiative)
    last_cards = None  # card counts of last_mid
    trick = 1
    prev_player = None
    prev_pass = True  # treat first turn as "just after pass" (initiative)

    for _, row in gdf.iterrows():
        p = row["current_player"]
        opp = 1 - p
        is_pass = bool(row["pass_"])
        mc = [row[f"move_at{r}"] for r in range(13)]
        ha = hbits(row["hand_at1"], row["hand_at2"], row["hand_at3"], row["hand_at4"])
        hb = [ha[r] + mc[r] for r in range(13)]  # hand before this move

        # Opp max-possible counts from parquet (consistent with training)
        opp_counts = hbits(
            row["opp_at1"], row["opp_at2"], row["opp_at3"], row["opp_at4"]
        )

        # Detect implicit pass (same player appears twice with no pass row)
        if prev_player == p and not prev_pass:
            opp_hb = exact_hand[opp]  # None if opp hasn't played yet
            if opp_hb is not None:
                p_counts_as_opp = [RM[r] - opp_hb[r] for r in range(13)]
                forced_pass = len(compute_legal_moves_py(opp_hb, last_mid)) == 1
            else:
                # Opponent hasn't played yet; use complement of active player's hand
                p_counts_as_opp = opp_counts  # from current row's opp_at
                forced_pass = False  # unknown; don't claim forced
            print_turn(
                f,
                trick,
                opp,
                opp_hb,
                exact_hand[p],
                p_counts_as_opp,
                last_mid,
                last_cards,
                kPASS,
                None,
                model,
                device,
                forced=forced_pass,
                implicit=(opp_hb is None),
            )
            # Implicit pass → initiative resets
            last_mid = kPASS
            last_cards = None
            trick += 1

        # Detect forced pass for explicit pass rows
        forced = False
        if is_pass and last_mid != kPASS:
            if exact_hand[p] is not None:
                forced = len(compute_legal_moves_py(hb, last_mid)) == 1
            else:
                forced = True  # can't verify; assume forced

        chosen_mid = kPASS if is_pass else find_move_id(mc)

        print_turn(
            f,
            trick,
            p,
            hb,
            exact_hand[opp],
            opp_counts,
            last_mid,
            last_cards,
            chosen_mid,
            mc,
            model,
            device,
            forced=forced,
            tablebase=(exact_hand[opp] is not None and sum(exact_hand[opp]) == 1),
        )

        # Update state
        exact_hand[p] = ha
        if is_pass:
            last_mid = kPASS
            last_cards = None
            trick += 1
        else:
            last_mid = chosen_mid
            last_cards = mc[:]

        prev_player = p
        prev_pass = is_pass

    f.write(f"\n  {'─'*72}\n")
    f.write(f"  winner: P{winner}\n\n")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--model", default=None, help="TorchScript .pt model for scoring")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = device = None
    if args.model:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(args.model, map_location=device)
        model.eval()
        print(f"Loaded model: {args.model} on {device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.out, "w") as f:
        for path in args.parquets:
            label = Path(path).stem
            score_label = (
                f"  scored by: {Path(args.model).stem}"
                if args.model
                else "  (no scoring model)"
            )
            f.write(f"\n{'═'*72}\n  {label.upper()}{score_label}\n{'═'*72}\n")
            df = pd.read_parquet(path)
            ids = rng.choice(
                df["game_id"].unique(),
                size=min(args.games, df["game_id"].nunique()),
                replace=False,
            )
            for i, gid in enumerate(ids, 1):
                f.write(f"\n  game {i}\n")
                print_game(df[df["game_id"] == gid], f, model=model, device=device)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
