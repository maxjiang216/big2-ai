#ifndef HINT_COMPUTE_H
#define HINT_COMPUTE_H

#include "util.h"

#include <algorithm>
#include <array>
#include <random>
#include <vector>

// MC estimate of the probability that a random opponent hand (drawn uniformly
// from the max-possible distribution consistent with the player's view) contains
// at least one move that beats move_id.
//
// Returns 0.0 if move_id is kPASS (player passed, no trick to contest).
// Returns 1.0 if opponent_can_respond() confirms NO max-possible hand can beat
//   move_id (opponent definitely locked).
// Otherwise samples n_samples random hands and returns the fraction that can
// respond.  Uses a thread_local RNG so callers don't need to manage state.
//
// player_hand, discard: state BEFORE move_id was played (max_possible is
// invariant to which move was played, so before == after here).
inline float compute_hint(int move_id,
                          const std::array<int, 13> &player_hand,
                          const std::array<int, 13> &discard,
                          int opp_count,
                          int n_samples = 200) {
  if (move_id == kPASS)
    return 0.0f;

  // Deterministic upper-bound check: can any max-possible hand respond?
  if (!opponent_can_respond(move_id, player_hand, discard, opp_count))
    return 1.0f; // opponent is definitely locked

  // Build pool of unseen cards (not in player hand, not in discard).
  // These represent cards that could be in the opponent's hand.
  std::vector<int> pool;
  pool.reserve(opp_count + 16);
  for (int r = 0; r < 13; ++r) {
    int max_p = max_cards_in_deck_for_rank(r) - player_hand[r] - discard[r];
    for (int k = 0; k < max_p; ++k)
      pool.push_back(r);
  }

  if (static_cast<int>(pool.size()) < opp_count)
    return 0.5f; // degenerate state, shouldn't occur in valid play

  const auto &beating = get_beating_moves()[move_id];

  thread_local std::mt19937 rng(std::random_device{}());

  int respond_count = 0;
  std::vector<int> sample;
  sample.reserve(opp_count);

  for (int s = 0; s < n_samples; ++s) {
    sample.clear();
    std::sample(pool.begin(), pool.end(), std::back_inserter(sample),
                opp_count, rng);

    std::array<int, 13> hand{};
    for (int r : sample)
      hand[r]++;

    for (int beat_id : beating) {
      bool fits = true;
      for (int r = 0; r < 13; ++r) {
        if (MOVE_TO_CARDS[beat_id][r] > hand[r]) {
          fits = false;
          break;
        }
      }
      if (fits) {
        respond_count++;
        break;
      }
    }
  }

  return static_cast<float>(respond_count) / static_cast<float>(n_samples);
}

#endif
