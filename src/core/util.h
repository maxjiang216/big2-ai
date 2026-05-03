#ifndef UTIL_H
#define UTIL_H

#include "move.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#define LEGAL_MOVES_SIZE 468

const int kPASS = 0;
const int kSINGLE_START = 1;
const int kDOUBLE_START = kSINGLE_START + 13;
const int kTRIPLE_START = kDOUBLE_START + 12;
const int kFULL_HOUSE_START = kTRIPLE_START + 11;
const int kBOMB_START = kFULL_HOUSE_START + 132;

const int kSTRAIGHT5_START = kBOMB_START + 156;
const int kSTRAIGHT6_START = kSTRAIGHT5_START + 10;
const int kSTRAIGHT7_START = kSTRAIGHT6_START + 9;
const int kSTRAIGHT8_START = kSTRAIGHT7_START + 8;
const int kSTRAIGHT9_START = kSTRAIGHT8_START + 7;
const int kSTRAIGHT10_START = kSTRAIGHT9_START + 6;
const int kSTRAIGHT11_START = kSTRAIGHT10_START + 5;
const int kSTRAIGHT12_START = kSTRAIGHT11_START + 4;
const int kSTRAIGHT13_START = kSTRAIGHT12_START + 3;

const int kDOUBLESTRAIGHT2_START = kSTRAIGHT13_START + 1;
const int kDOUBLESTRAIGHT3_START = kDOUBLESTRAIGHT2_START + 11;
const int kDOUBLESTRAIGHT4_START = kDOUBLESTRAIGHT3_START + 10;
const int kDOUBLESTRAIGHT5_START = kDOUBLESTRAIGHT4_START + 9;
const int kDOUBLESTRAIGHT6_START = kDOUBLESTRAIGHT5_START + 8;
const int kDOUBLESTRAIGHT7_START = kDOUBLESTRAIGHT6_START + 7;

const int kTRIPLESTRAIGHT2_START = kDOUBLESTRAIGHT7_START + 6;
const int kTRIPLESTRAIGHT3_START = kTRIPLESTRAIGHT2_START + 10; // highest rank K (ace excluded)
const int kTRIPLESTRAIGHT4_START = kTRIPLESTRAIGHT3_START + 9;
const int kTRIPLESTRAIGHT5_START = kTRIPLESTRAIGHT4_START + 8;

const int kDOUBLESTRAIGHT8_START = kTRIPLESTRAIGHT5_START + 7;

#include "move_to_cards.inc"

inline constexpr int max_cards_in_deck_for_rank(int rank_index) {
  if (rank_index == 11)
    return 3;
  if (rank_index == 12)
    return 1;
  return 4;
}

// ---------------------------------------------------------------------------
// HandBits — cumulative bitmask representation of a player's hand.
// at_n: bit r is set iff the player holds ≥n cards of rank r (0-based).
// Invariant: at4 ⊆ at3 ⊆ at2 ⊆ at1 (bits 0-12 used; bits 13-15 always 0).
// ---------------------------------------------------------------------------
struct HandBits {
    uint16_t at1{}, at2{}, at3{}, at4{};
};

inline HandBits hand_bits_from_counts(const std::array<int, 13> &c) {
    HandBits h;
    for (int r = 0; r < 13; ++r) {
        if (c[r] >= 1) h.at1 |= static_cast<uint16_t>(1u << r);
        if (c[r] >= 2) h.at2 |= static_cast<uint16_t>(1u << r);
        if (c[r] >= 3) h.at3 |= static_cast<uint16_t>(1u << r);
        if (c[r] >= 4) h.at4 |= static_cast<uint16_t>(1u << r);
    }
    return h;
}

inline std::array<int, 13> hand_bits_to_counts(const HandBits &h) {
    std::array<int, 13> c{};
    for (int r = 0; r < 13; ++r)
        c[r] = ((h.at1 >> r) & 1) + ((h.at2 >> r) & 1) +
               ((h.at3 >> r) & 1) + ((h.at4 >> r) & 1);
    return c;
}

// Remove `cost` cards of rank r (modifies h in place; called from apply_move).
inline void hand_bits_remove(HandBits &h, int r, int cost) {
    const int old_c = ((h.at1 >> r) & 1) + ((h.at2 >> r) & 1) +
                      ((h.at3 >> r) & 1) + ((h.at4 >> r) & 1);
    const int new_c = old_c - cost;
    const uint16_t bit = static_cast<uint16_t>(1u << r);
    if (new_c < 4) h.at4 &= static_cast<uint16_t>(~bit);
    if (new_c < 3) h.at3 &= static_cast<uint16_t>(~bit);
    if (new_c < 2) h.at2 &= static_cast<uint16_t>(~bit);
    if (new_c < 1) h.at1 &= static_cast<uint16_t>(~bit);
}

// True if every rank appears at most once (no pair, triple, or bomb in hand).
inline bool hand_is_only_singles(const std::array<int, 13> &hand) {
  for (int i = 0; i < 13; ++i) {
    if (hand[i] > 1)
      return false;
  }
  return true;
}

char rankToChar(int rank);

// Precomputed table of all 468 Move objects, indexed by legal move ID.
// Eliminates repeated Move(int) construction in hot loops.
const std::array<Move, LEGAL_MOVES_SIZE> &all_moves();

std::vector<int> compute_legal_moves(const HandBits &hand,
                                     const Move &last_move);
// Compatibility shim: constructs HandBits from count array, then delegates.
std::vector<int> compute_legal_moves(const std::array<int, 13> &hand,
                                     const Move &last_move);
// Integer-move-id overload (convenience; used by tests and find_forced_win callers).
std::vector<int> compute_legal_moves(const std::array<int, 13> &hand, int move_id);

// Fills `out` in place (avoids vector reallocation at call sites that already hold a buf).
void compute_legal_moves_into(const std::array<int, 13> &hand, int move_id,
                              std::vector<int> &out);

// Fill `out` with the straight move IDs playable from the given single/double/triple
// presence bitmasks. Exposed for testing equivalence against brute-force reference.
void straight_moves_for_pass_masks_into(uint16_t hs, uint16_t hp, uint16_t ht,
                                        std::vector<int> &out);

// kPASS lookup table for small hands — not implemented (set to 0 to skip table path).
extern int g_kpass_legal_max_cards;
constexpr int KPASS_LEGAL_TABLE_MAX_CARDS_BUILT = 0;

std::vector<int> compute_possible_moves(const std::array<int, 13> &player_hand,
                                        const std::array<int, 13> &discard_pile,
                                        int opponent_card_count,
                                        const Move &last_move,
                                        bool exclude_bombs);

// For each non-pass move_id, the list of move_ids that can beat it
// (same combination type with higher rank, or any bomb).
// Computed once on first call; O(LEGAL_MOVES_SIZE^2) pre-processing.
const std::vector<std::vector<int>> &get_beating_moves();

// True if the opponent might still hold at least one response to move_id,
// given our hand, the discard pile, and the opponent's known card count.
// Bomb responses are checked using bare bombs only (no auxiliary / kicker):
// same rank, strictly fewer cards, same beating power.
bool opponent_can_respond(int move_id,
                          const std::array<int, 13> &hand,
                          const std::array<int, 13> &discard,
                          int opp_count);

// From a new-trick (we have the lead) position, search for a sequence of
// moves that guarantees we empty our hand:
//   - A hand-emptying move is always a win (opponent's response is irrelevant).
//   - An unbeatable move (opponent must pass) keeps the lead; recurse.
// Depth-first over legal moves sorted by combination size (most cards first;
// ties broken by lower move_id). Returns the *first* complete winning sequence
// found in that order, not an exhaustive comparison of all winning lines.
// Returns nullopt if no such sequence exists. Only the first element is played;
// the rest is informational.
std::optional<std::vector<int>>
find_forced_win(const std::array<int, 13> &hand,
                const std::array<int, 13> &discard,
                int opp_count);

#endif
