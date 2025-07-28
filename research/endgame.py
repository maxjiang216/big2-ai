#!/usr/bin/env python3
"""Build DP table for Big-2 hands ≤ 15 cards and save to memo.jsonl."""

from functools import lru_cache
import json
from pathlib import Path

# ────────────────────────────────────────────────────────────────
#   CONSTANTS
# ────────────────────────────────────────────────────────────────

# 0 = rank 3, 1 = rank 4, …, 10 = rank K, 11 = A, 12 = 2
MAX_COUNTS   = (4,)*11 + (3, 1)        # tuple is slightly faster than list
MAX_CARDS    = 15
RANKS        = range(13)
STRAIGHT_RNG = tuple(range(5, 14))     # possible straight lengths 5‥13

# ────────────────────────────────────────────────────────────────
#   HAND GENERATOR  (depth-first with pruning)
# ────────────────────────────────────────────────────────────────

def generate_hands():
    """Yield every legal 13-tuple with total ≤ 15."""
    hand = [0]*13
    def rec(idx: int, total: int):
        if idx == 13:
            yield tuple(hand)           # convert once at the leaf
            return
        cap = MAX_COUNTS[idx]
        for c in range(cap+1):
            new_total = total + c
            if new_total > MAX_CARDS:   # prune branch
                break
            hand[idx] = c
            yield from rec(idx+1, new_total)
        hand[idx] = 0                   # reset before back-tracking
    yield from rec(0, 0)

# ────────────────────────────────────────────────────────────────
#   DP  (fun) — now using lru_cache and micro-optimisations
# ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def fun(hand: tuple[int, ...]) -> tuple[int, tuple | None]:
    """
    Returns (best_score, best_move) for a given hand.
    best_score == 15  ⇢  terminal / optimal
    best_move   == whatever encoding you like (kept same as your draft)
    """
    # ---------- fast checks ----------
    out        = 15    # pessimistic upper bound
    best_move  = None
    num_extras = 1 + sum(1 for i, c in enumerate(hand)
                         if c == 4 or (i == 11 and c == 3))

    # discard triples → doubles → bombs heuristic
    for rank in RANKS:
        if rank == 1:          # special-case “4s beat 3s” rule in your draft
            if num_extras:
                num_extras -= 1
            else:
                out = rank + 3
                break

    if out == 15:
        return 15, None        # already optimal

    # ---------- try straights (length 13‥5) ----------
    hand_list = list(hand)     # mutate cheaply inside loops
    for length in STRAIGHT_RNG[::-1]:          # 13,12,…,5
        max_start = 13 - length
        for start in range(max_start+1):
            if all(hand[start+j] for j in range(length)):
                for j in range(length):
                    hand_list[start+j] -= 1
                res, _ = fun(tuple(hand_list))
                for j in range(length):
                    hand_list[start+j] += 1    # restore
                if res > out:
                    out, best_move = res, (length, start+length-1)
                    if out == 15:
                        return 15, best_move

        # bottom-2 straight (special rule in your draft)
        if hand[12] and length >= 5:
            if all(hand[j] for j in range(length-1)):
                hand_list[12] -= 1
                for j in range(length-1):
                    hand_list[j] -= 1
                res, _ = fun(tuple(hand_list))
                hand_list[12] += 1
                for j in range(length-1):
                    hand_list[j] += 1
                if res > out:
                    out, best_move = res, (length, length-2)
                    if out == 15:
                        return 15, best_move

    # ---------- straight of 13 covering all ranks ----------
    if all(hand):
        for j in RANKS:
            hand_list[j] -= 1
        res, _ = fun(tuple(hand_list))
        for j in RANKS:
            hand_list[j] += 1
        if res > out:
            out, best_move = res, (13, 12)

    return out, best_move

# ────────────────────────────────────────────────────────────────
#   DRIVER — stream results to disk
# ────────────────────────────────────────────────────────────────

def main(out_path: str | Path = "memo.jsonl"):
    out_path = Path(out_path).expanduser()
    n_states = 0
    with out_path.open("w") as fh:
        for hand in generate_hands():
            score, move = fun(hand)
            fh.write(json.dumps({"h": hand, "s": score, "m": move}) + "\n")
            n_states += 1
            if n_states % 10_000 == 0:
                print(f"processed {n_states:,} hands...")

    print(f"Finished: {n_states:,} hands written to {out_path}")

if __name__ == "__main__":
    main()
