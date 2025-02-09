import math

# --- Global definitions ---
# There are 13 ranks: indices 0..12 correspond to card ranks as follows:
# 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2.
# We use numeric values: 3..13 for 3..K, 14 for Ace, and 15 for 2.
# Limits: 3–K: 4 copies each; Ace: 3 copies; 2: 1 copy.
limits = [4] * 11 + [3, 1]

def r2i(r):
    """Convert a card’s rank (3..15, where 15 means 2) to index 0..12."""
    return 12 if r == 15 else r - 3

# --- Enumerate all starting hands (by rank composition) ---
def gen_compositions(i, remaining, current, weight):
    """Recursively yield all (composition, weight) tuples.
       current: list of length 13 of chosen counts so far.
       weight: product_{i} math.comb(limits[i], count) for ranks processed.
    """
    if i == len(limits):
        if remaining == 0:
            yield (tuple(current), weight)
        return
    for x in range(0, min(limits[i], remaining) + 1):
        current[i] = x
        new_weight = weight * math.comb(limits[i], x)
        yield from gen_compositions(i + 1, remaining - x, current, new_weight)

# Build a list of (composition, weight) for all valid 16–card hands.
all_hands = list(gen_compositions(0, 16, [0] * len(limits), 1))
print("Total number of compositions:", len(all_hands))

# --- Helper functions to test if a hand (composition) can play a move ---

def can_play_single(comp, r):
    return comp[r2i(r)] >= 1

def can_play_double(comp, r):
    return comp[r2i(r)] >= 2

def can_play_triple(comp, r):
    return comp[r2i(r)] >= 3

def can_play_full_house(comp, triple, pair):
    return comp[r2i(triple)] >= 3 and comp[r2i(pair)] >= 2

def can_play_bomb(comp, r, aux):
    # For non-Aces, need 4 copies; for Ace (14), need 3 copies.
    if r == 14:
        ok = comp[r2i(r)] >= 3
    else:
        ok = comp[r2i(r)] >= 4
    if aux != 0:
        ok = ok and (comp[r2i(aux)] >= 1)
    return ok

def can_play_straight(comp, L, H):
    """
    For a straight of length L (5 <= L <= 13) with highest card H.
    If H == L+1, interpret that as the special case using 2 as the low card,
    so required cards are {2} plus {3,...,H}. Otherwise, required cards
    are {H-L+1, ..., H}.
    """
    if H == L + 1:
        req = [15] + list(range(3, H + 1))
    else:
        req = list(range(H - L + 1, H + 1))
    return all(comp[r2i(r)] >= 1 for r in req)

def can_play_sister(comp, L, H):
    """
    For a double straight ("sister") of length L (2 <= L <= 8) with highest pair H.
    Requires at least 2 copies in each rank of the block.
    """
    req = list(range(H - L + 1, H + 1))
    return all(comp[r2i(r)] >= 2 for r in req)

def can_play_triple_straight(comp, L, H):
    """
    For a triple straight of length L (2 <= L <= 5) with highest triple H.
    Requires at least 3 copies in each rank of the block.
    """
    req = list(range(H - L + 1, H + 1))
    return all(comp[r2i(r)] >= 3 for r in req)

# --- Count the number of legal moves available from a given hand composition ---
def legal_moves_count(comp):
    count = 0
    # 1. Singles: for each rank r in 3..15.
    for r in range(3, 16):
        if can_play_single(comp, r):
            count += 1
    # 2. Doubles: for each rank r in 3..14.
    for r in range(3, 15):
        if can_play_double(comp, r):
            count += 1
    # 3. Triples: for each rank r in 3..13.
    for r in range(3, 14):
        if can_play_triple(comp, r):
            count += 1
    # 4. Full houses: for triple in 3..15 and pair in 3..15 (pair != triple).
    for triple in range(3, 16):
        for pair in range(3, 16):
            if pair == triple:
                continue
            if can_play_full_house(comp, triple, pair):
                count += 1
    # 5. Bombs: for bomb rank r in 3..14.
    for r in range(3, 15):
        # Bomb without extra.
        if can_play_bomb(comp, r, 0):
            count += 1
        # Bomb with extra card.
        for a in list(range(3, r)) + list(range(r + 1, 16)):
            if can_play_bomb(comp, r, a):
                count += 1
    # 6. Straights: for each length L from 5 to 12, for H in [L+1, 15];
    # for L == 13, only H == 15 is allowed.
    for L in range(5, 13):
        for H in range(L + 1, 16):
            if can_play_straight(comp, L, H):
                count += 1
    if can_play_straight(comp, 13, 15):
        count += 1
    # 7. Sisters (Double straights): for L from 2 to 8, for H in range(L+2, 15).
    for L in range(2, 9):
        for H in range(L + 2, 15):
            if can_play_sister(comp, L, H):
                count += 1
    # 8. Triple straights: for L from 2 to 5, for H in range(L+2, 15).
    for L in range(2, 6):
        for H in range(L + 2, 15):
            if can_play_triple_straight(comp, L, H):
                count += 1
    return count

# --- Find the starting hand with the most legal moves ---
max_moves = -1
max_comp = None
for comp, weight in all_hands:
    moves = legal_moves_count(comp)
    if moves > max_moves:
        max_moves = moves
        max_comp = comp

print("Maximum legal moves count:", max_moves)
print("Starting hand composition with most legal moves:")

# --- Print the composition in a readable format ---
# Map indices to ranks: index 0->'3', 1->'4', ..., 8->'11' (which we show as 'J'), 9->'Q', 10->'K', 11->'A', 12->'2'
rank_labels = [ "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2" ]
for i, cnt in enumerate(max_comp):
    print(f"{rank_labels[i]}: {cnt}", end="  ")
print()
