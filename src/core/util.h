#ifndef UTIL_H
#define UTIL_H

#include "move.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#define LEGAL_MOVES_SIZE 468

// kPASS precomputed legal table: set BUILD to true to populate maps at
// MoveTables init; lookup still requires g_kpass_legal_max_cards > 0.
inline constexpr bool KPASS_LEGAL_TABLE_BUILD = false;
inline constexpr int KPASS_LEGAL_TABLE_MAX_CARDS_BUILT = 9;
// Primary map holds [1, KPASS_LEGAL_PRIMARY_MAX_CARDS]; 8–9 use kpass_legal_hi.
inline constexpr int KPASS_LEGAL_PRIMARY_MAX_CARDS = 7;

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

// True if every rank appears at most once (no pair, triple, or bomb in hand).
inline bool hand_is_only_singles(const std::array<int, 13> &hand) {
  for (int i = 0; i < 13; ++i) {
    if (hand[i] > 1)
      return false;
  }
  return true;
}

char rankToChar(int rank);

std::vector<int> compute_legal_moves(const std::array<int, 13> &hand,
                                     int last_move_id);

// Fills `out` with the same ids as compute_legal_moves (after clear()). Reuses
// `out` capacity to avoid per-call allocations on hot paths.
void compute_legal_moves_into(const std::array<int, 13> &hand, int last_move_id,
                              std::vector<int> &out);

// 0 = never use kPASS table (default while KPASS_LEGAL_TABLE_BUILD is false).
// Otherwise cap by sum(hand) <= this value (max KPASS_LEGAL_TABLE_MAX_CARDS_BUILT).
extern int g_kpass_legal_max_cards;

// kPASS straight-family move ids only: threshold bitmasks hs (>=1 card), hp (>=2),
// ht (>=3) per rank. Clears `out` then appends ids in rolling-scan order.
void straight_moves_for_pass_masks_into(uint16_t hs, uint16_t hp, uint16_t ht,
                                          std::vector<int> &out);

std::vector<int> compute_possible_moves(const std::array<int, 13> &player_hand,
                                        const std::array<int, 13> &discard_pile,
                                        int opponent_card_count,
                                        int last_move_id,
                                        bool exclude_bombs);

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
