// Exact P(opponent can beat move M) by exhaustive enumeration over all valid
// 16-card opponent starting hands, weighted by multinomial hypergeometric:
//
//   P(beat M) = sum_{H: can_beat(H,M)} prod_r C(u_r, H_r)  /  C(sum_u, 16)
//
// where u_r = max_deck[r] - MOVE_TO_CARDS[M][r].
//
// Acceleration: Vandermonde identity short-circuits entire subtrees when a
// partial hand already guarantees a beat:
//   sum_{H: sum=rem, 0<=H_r<=u_r} prod C(u_r, H_r)  =  C(suffix_cap, rem)
//
// Beat categories (handled during recursion via Vandermonde):
//   - Non-bomb M: any bomb in hand (4-of-a-kind or triple-A)
//   - Bomb M: higher-ranked bomb in hand
//   - Singles/doubles/triples: also rank-local same-type beat detected per rank
//
// Beat categories (handled at leaf node for remaining move types):
//   - Full houses: higher triple rank, with a valid pair present
//   - Straights: same combination, higher top rank
//
// Usage: make exact_hint && ./bin/exact_hint

#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "util.h"  // MOVE_TO_CARDS, kSINGLE_START etc., max_cards_in_deck_for_rank

// ---------------------------------------------------------------------------
// Binomial table  (n, k up to 51)
// ---------------------------------------------------------------------------

static long long binom_tbl[52][52];

static void init_binom() {
    std::memset(binom_tbl, 0, sizeof(binom_tbl));
    for (int n = 0; n < 52; ++n) {
        binom_tbl[n][0] = 1;
        for (int k = 1; k <= n; ++k)
            binom_tbl[n][k] = binom_tbl[n - 1][k - 1] + binom_tbl[n - 1][k];
    }
}

// ---------------------------------------------------------------------------
// Move name helper
// ---------------------------------------------------------------------------

static std::string move_name(int move_id) {
    std::ostringstream oss;
    oss << Move(move_id);
    return oss.str();
}

// ---------------------------------------------------------------------------
// Rank value → rank index conversion
// Rank values: 3-14 map to indices 0-11; 2 and 15 both map to index 12 (the 2 card).
// ---------------------------------------------------------------------------

static inline int rv_to_idx(int rv) {
    return (rv == 2) ? 12 : (rv - 3);
}

// ---------------------------------------------------------------------------
// Leaf same-type beat checks  (for move types not covered by Vandermonde)
// ---------------------------------------------------------------------------

// Consecutive-rank straight: same length, higher top rank, each position >= min_cnt cards.
// Used for single/double/triple straights.
// max_rank_val: maximum possible top-rank value for this combination type.
static bool can_beat_consec(
    const std::array<int, 13> &h,
    int len, int min_cnt, int cur_rank_val, int max_rank_val)
{
    for (int r = cur_rank_val + 1; r <= max_rank_val; ++r) {
        bool ok = true;
        for (int i = 0; i < len && ok; ++i)
            if (h[rv_to_idx(r - i)] < min_cnt) ok = false;
        if (ok) return true;
    }
    return false;
}

// Full house: higher triple rank (values 3..13), any pair of a different rank.
static bool can_beat_full_house(const std::array<int, 13> &h, int triple_rank_val) {
    for (int j = triple_rank_val - 3 + 1; j <= 10; ++j) {
        if (h[j] < 3) continue;
        for (int k = 0; k < 13; ++k)
            if (k != j && h[k] >= 2) return true;
    }
    return false;
}

// Dispatches to the right leaf check for each combination type.
// Returns false for move types that are fully covered by Vandermonde.
static bool can_beat_at_leaf(const std::array<int, 13> &h, const Move &m) {
    using C = Move::Combination;
    switch (m.combination) {
    // Fully covered by Vandermonde during enumeration — never need leaf check.
    case C::kSingle: case C::kDouble: case C::kTriple: case C::kBomb:
        return false;

    case C::kFullHouse:
        return can_beat_full_house(h, m.rank);

    // Single straights: max top rank = 15 (2-card wrap in lowest straights).
    case C::kStraight5:  return can_beat_consec(h, 5,  1, m.rank, 15);
    case C::kStraight6:  return can_beat_consec(h, 6,  1, m.rank, 15);
    case C::kStraight7:  return can_beat_consec(h, 7,  1, m.rank, 15);
    case C::kStraight8:  return can_beat_consec(h, 8,  1, m.rank, 15);
    case C::kStraight9:  return can_beat_consec(h, 9,  1, m.rank, 15);
    case C::kStraight10: return can_beat_consec(h, 10, 1, m.rank, 15);
    case C::kStraight11: return can_beat_consec(h, 11, 1, m.rank, 15);
    case C::kStraight12: return can_beat_consec(h, 12, 1, m.rank, 15);
    case C::kStraight13: return false;  // rank=15 is max; nothing higher

    // Double straights: max top rank = 14 (A); 2 has count 1 so no pairs.
    case C::kDoubleStraight2: return can_beat_consec(h, 2, 2, m.rank, 14);
    case C::kDoubleStraight3: return can_beat_consec(h, 3, 2, m.rank, 14);
    case C::kDoubleStraight4: return can_beat_consec(h, 4, 2, m.rank, 14);
    case C::kDoubleStraight5: return can_beat_consec(h, 5, 2, m.rank, 14);
    case C::kDoubleStraight6: return can_beat_consec(h, 6, 2, m.rank, 14);
    case C::kDoubleStraight7: return can_beat_consec(h, 7, 2, m.rank, 14);
    case C::kDoubleStraight8: return can_beat_consec(h, 8, 2, m.rank, 14);

    // Triple straights: max top rank = 13 (K); triple-A is a bomb, not a triple.
    case C::kTripleStraight2: return can_beat_consec(h, 2, 3, m.rank, 13);
    case C::kTripleStraight3: return can_beat_consec(h, 3, 3, m.rank, 13);
    case C::kTripleStraight4: return can_beat_consec(h, 4, 3, m.rank, 13);
    case C::kTripleStraight5: return can_beat_consec(h, 5, 3, m.rank, 13);

    default: return false;
    }
}

// ---------------------------------------------------------------------------
// Recursive enumeration
// ---------------------------------------------------------------------------

// rank         : current rank index being assigned (0..12)
// remaining    : cards still to assign to ranks rank..12
// upper        : upper bound for each rank (max_deck[r] - move_cards[r])
// suffix_cap   : suffix_cap[r] = sum(upper[r..12]); suffix_cap[13] = 0
// pw           : product of binom(upper[r'], h[r']) for r' < rank
// h            : current partial hand
// is_bomb      : M is a bomb
// bomb_lo_idx  : lower bound for bomb rank threshold:
//                  is_bomb  → only ranks > bomb_lo_idx can form a beating bomb
//                  !is_bomb → -1 (any bomb rank beats)
// same_type_lo : for singles/doubles/triples: rank-local threshold index;
//                rank > same_type_lo AND k >= same_type_min triggers a beat
//                (-1 if not applicable)
// same_type_min: min count for rank-local same-type beat
// m            : Move object for leaf can_beat_at_leaf() dispatch
// already_beats: partial hand already guarantees a beat
// total_w / beat_w: weighted-count accumulators

static void enumerate(
    int rank, int remaining,
    const std::array<int, 13> &upper,
    const std::array<int, 14> &suffix_cap,
    long long pw,
    std::array<int, 13> &h,
    bool is_bomb, int bomb_lo_idx,
    int same_type_lo, int same_type_min,
    const Move &m,
    bool already_beats,
    long double &total_w, long double &beat_w)
{
    // Vandermonde short-circuit: all completions of this partial hand beat M.
    if (already_beats) {
        long double delta = pw * (long double)binom_tbl[suffix_cap[rank]][remaining];
        total_w += delta;
        beat_w  += delta;
        return;
    }

    if (rank == 13) {
        if (remaining == 0) {
            total_w += pw;
            if (can_beat_at_leaf(h, m)) beat_w += pw;
        }
        return;
    }

    int lo = std::max(0, remaining - suffix_cap[rank + 1]);
    int hi = std::min(upper[rank], remaining);
    if (lo > hi) return;

    for (int k = lo; k <= hi; ++k) {
        h[rank] = k;
        long long new_pw = pw * binom_tbl[upper[rank]][k];
        bool now_beats = false;

        // Bomb detection: for non-bomb M any bomb beats; for bomb M only higher-rank bombs.
        bool bomb_eligible = !is_bomb || (rank > bomb_lo_idx);
        if (bomb_eligible && ((rank < 11 && k >= 4) || (rank == 11 && k >= 3)))
            now_beats = true;

        // Rank-local same-type beat (singles/doubles/triples only).
        if (!now_beats && same_type_lo >= 0 && rank > same_type_lo && k >= same_type_min)
            now_beats = true;

        enumerate(rank + 1, remaining - k, upper, suffix_cap, new_pw, h,
                  is_bomb, bomb_lo_idx, same_type_lo, same_type_min,
                  m, now_beats, total_w, beat_w);
    }
}

// ---------------------------------------------------------------------------
// Per-move computation
// ---------------------------------------------------------------------------

static double compute(int move_id) {
    Move m(move_id);
    using C = Move::Combination;

    bool is_bomb    = (m.combination == C::kBomb);
    int  rank_idx   = rv_to_idx(m.rank);
    int  bomb_lo    = is_bomb ? rank_idx : -1;

    int same_type_lo  = -1;
    int same_type_min = -1;
    switch (m.combination) {
    case C::kSingle: same_type_lo = rank_idx; same_type_min = 1; break;
    case C::kDouble: same_type_lo = rank_idx; same_type_min = 2; break;
    case C::kTriple: same_type_lo = rank_idx; same_type_min = 3; break;
    default: break;
    }

    std::array<int, 13> upper;
    for (int r = 0; r < 13; ++r)
        upper[r] = max_cards_in_deck_for_rank(r) - MOVE_TO_CARDS[move_id][r];

    std::array<int, 14> suffix_cap;
    suffix_cap[13] = 0;
    for (int r = 12; r >= 0; --r)
        suffix_cap[r] = suffix_cap[r + 1] + upper[r];

    std::array<int, 13> h{};
    long double total_w = 0, beat_w = 0;

    enumerate(0, 16, upper, suffix_cap, 1LL, h,
              is_bomb, bomb_lo, same_type_lo, same_type_min,
              m, false, total_w, beat_w);

    return (total_w > 0) ? (double)(beat_w / total_w) : 0.0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    init_binom();

    double table[LEGAL_MOVES_SIZE] = {};

    std::cout << std::left << std::fixed << std::setprecision(6);
    std::cout << std::setw(6)  << "ID"
              << std::setw(22) << "Move"
              << "P(beat)\n";
    std::cout << std::string(36, '-') << "\n";

    auto wall0 = std::chrono::high_resolution_clock::now();

    for (int mid = 1; mid < LEGAL_MOVES_SIZE; ++mid) {
        auto t0 = std::chrono::high_resolution_clock::now();
        table[mid] = compute(mid);
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        std::cout << std::setw(6)  << mid
                  << std::setw(22) << move_name(mid)
                  << std::setw(10) << table[mid]
                  << "  (" << std::setprecision(1) << ms << " ms)"
                  << std::setprecision(6) << "\n";
    }

    double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - wall0).count();

    std::cout << "\nTotal time: " << std::setprecision(1) << total_ms << " ms\n\n";

    // Compact C++ initializer for the lookup table
    std::cout << "// --- Lookup table (copy into hint_table.h) ---\n";
    std::cout << "static const float kHintTable[" << LEGAL_MOVES_SIZE << "] = {\n";
    std::cout << "    0.0f, // 0 = pass\n";
    for (int mid = 1; mid < LEGAL_MOVES_SIZE; ++mid) {
        std::cout << "    " << std::setprecision(6) << (float)table[mid] << "f,";
        if (mid % 8 == 0) std::cout << " // IDs " << (mid - 7) << "-" << mid;
        std::cout << "\n";
    }
    std::cout << "};\n";

    return 0;
}
