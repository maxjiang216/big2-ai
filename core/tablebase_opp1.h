#ifndef TABLEBASE_OPP1_H
#define TABLEBASE_OPP1_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>

// Tablebase for: we have the lead, opponent has exactly 1 card.
// Precomputed file lists hands where playing a specific straight first beats
// the default strategy; each entry is only the first straight. Chained lookups
// (play straight, re-query) recover the full plan.

struct Opp1Result {
  // Metric after optimal straight-prefix play (0..12 rank index, 15 = won /
  // trivial). Only meaningful when the hand is in the table.
  uint8_t score{0};
  // Encoded move id of the straight to play first, or 0 = use default strategy.
  int first_move_id{0};
};

// Load binary table from disk (e.g. from `make tablebase_opp1_gen`, default
// `tablebase_opp1.bin` next to the binary). Safe to call
// once at startup; map is read-only after load. If the file is missing or
// invalid, lookups behave as "not found".
void load_tablebase_opp1(const std::string &path);

// Returns {0,0} if the hand is absent from the table (use default strategy).
Opp1Result lookup_opp1(const std::array<int, 13> &hand);

// Default endgame line: bomb (min move id) → triple → double → longest
// straight → lowest single. Used when lookup returns first_move_id == 0.
std::optional<int> opp1_default_strategy_move(const std::array<int, 13> &hand);

#endif
