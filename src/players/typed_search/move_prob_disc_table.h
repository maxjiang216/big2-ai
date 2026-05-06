#ifndef TYPED_SEARCH_MOVE_PROB_DISC_TABLE_H
#define TYPED_SEARCH_MOVE_PROB_DISC_TABLE_H

#include "move_prob_table.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace typed_search {

// Discard-conditioned move-prob table. Adds a per-rank "opp-availability"
// bitmap to the key, only for moves where the response distribution depends
// substantially on what opp could still hold.
//
// Eligible moves: singles 3-K, doubles 3-K, triples 3-Q (32 moves total).
// Excluded: single A/2 and double A (no plausible higher response except a
// bomb), triple K/A (only triple-A above triple-K is the A-bomb), full
// houses / bombs / straights / DS / TS (mostly PASS-dominated by analysis).
//
// Bitmap: 11 bits, bit i = "opp could still hold ≥3 of rank idx (i+1)".
// (rank idx 1..11 = face 4..A; we don't track rank 2 = idx 12.)
//
// For a given player_move with rank idx R, only bits R..10 are relevant
// (lower bits encode ranks opp can't beat with this combo type, masked out
// before keying).
//
// Query is shrinkage-blended with `mp_main` as the prior: if the disc cell
// has D total observations, the result is
//     P[m] = (kappa * mp_main_prob[m] + disc_count[m]) / (kappa + D)
// for each candidate response m.
class MoveProbDiscardTable {
public:
  static constexpr int kNumMoves = 468;
  static constexpr int kBitmapWidth = 11;
  static constexpr int kBitmapMax = 1 << kBitmapWidth;

  // counts[y]: # of times opp played y when y was deduced-feasible.
  // trials[y]: # of times y was deduced-feasible (whether played or not).
  struct Entry {
    std::array<float, kNumMoves> counts{};
    std::array<float, kNumMoves> trials{};
  };

  static bool is_eligible(int move_id);

  // Compute the full bitmap from the move-player's POV after the move:
  //   bit i = 1 iff min(max_cards_in_deck[i+1] - hand[i+1] - discard[i+1],
  //                      opp_count) >= 3.
  static std::uint16_t compute_bitmap(const std::array<int, 13> &hand_after,
                                        const std::array<int, 13> &discard_after,
                                        int opp_count);

  // Mask the bitmap to the bits relevant to `move_id` (lower bits zeroed).
  static std::uint16_t mask_relevant(int move_id, std::uint16_t bitmap);

  void load(const std::string &path);
  void save(const std::string &path) const;

  // Shrinkage-blended query against mp_main as prior.
  void query(int move_id, int opp_count, int our_hand_size_bucket,
             std::uint16_t bitmap_full, const std::vector<int> &legal_moves,
             const MoveProbTable &mp_main,
             std::vector<float> &out_probs) const;

  // Training. No-op for ineligible moves. For each y in feasible_set, increment
  // trials by weight; for played_response also increment its count.
  void add_observation(int move_id, int opp_count, int our_hand_size_bucket,
                       std::uint16_t bitmap_full,
                       const std::vector<int> &feasible_set,
                       int played_response, float weight = 1.0f);
  void decay(float alpha);

  std::size_t size() const { return entries_.size(); }

private:
  static std::uint32_t make_key(int move_id, int opp_count, int our_b,
                                 std::uint16_t bitmap_relevant) {
    // ((move * 17 + opp_count) * 4 + our_b) * 2048 + bitmap (11 bits)
    return ((((static_cast<std::uint32_t>(move_id) * 17u +
                 static_cast<std::uint32_t>(opp_count)) *
                4u) +
              static_cast<std::uint32_t>(our_b)) *
             static_cast<std::uint32_t>(kBitmapMax)) +
            static_cast<std::uint32_t>(bitmap_relevant);
  }

  std::unordered_map<std::uint32_t, Entry> entries_;
};

}  // namespace typed_search

#endif
