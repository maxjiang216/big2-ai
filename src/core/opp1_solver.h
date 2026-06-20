#ifndef OPP1_SOLVER_H
#define OPP1_SOLVER_H

#include "series.h"

#include <array>
#include <vector>

// Exact endgame solver for "WE have the lead and the opponent holds exactly 1
// card." (The response-position variant, point 7, reduces to this.)
//
// Theory. Opp's single can only ever respond to a single we lead, and only by a
// strictly higher rank (suits are abstracted in the rank-count model; a bomb is
// impossible with 1 card). If opp ever beats a single, opp empties and wins. So
// every multi-card move we make is unbeatable, and the only cards at risk are
// the singles we are forced to lead one at a time.
//
// Let S = the multiset of EXPOSED singles (singles not absorbed by any combo /
// straight / bomb-auxiliary). For an opp card of rank X, with
//   k = #{ s in S : s < X },
//   k <= 1  -> we WIN (the lowest exposed single is played last = free),
//   k >= 2  -> we LOSE by (k - 1) cards.
// Only single straights and bomb auxiliaries can change S (pairs/triples/
// full-houses/double-/triple-straights merely shed >=2-count ranks), so the
// brute force is over which single straights to play first.

namespace opp1 {

// Belief over the rank of the opponent's hidden single. prob[r] for rank index
// r in 0..12 (rank value r+3). Sum to 1 for EV; irrelevant for dominance.
using Belief = std::array<double, 13>;

// One candidate line of play.
struct Plan {
  std::vector<int> moves;    // full ordered move-id sequence
  std::vector<int> exposed;  // exposed single rank VALUES (3..15), ascending
};

// k(X): number of exposed singles strictly below opp rank X (rank value).
int k_below(const Plan &p, int opp_rank);

// Our series win-probability if opp's card has rank X. Uses the series table
// when t.loaded; otherwise binary (win=1.0, loss=0.0).
double value_vs_rank(const Plan &p, int opp_rank, const SeriesTable &t,
                     int my_pts, int opp_pts);

// EV over the belief.
double ev(const Plan &p, const Belief &b, const SeriesTable &t, int my_pts,
          int opp_pts);

// True iff plan a is never worse than b for any opp rank (k_a(X) <= k_b(X) ∀X).
bool weakly_dominates(const Plan &a, const Plan &b);

// All candidate lines: the trivial combo line plus one per single-straight
// packing played first.
std::vector<Plan> enumerate_plans(const std::array<int, 13> &hand);

struct Solution {
  std::vector<int> moves;  // chosen line (empty hand -> empty)
  bool by_dominance{false};  // a single plan dominated every other
};

// Choose the line. If one plan k-dominates all others, play it (objective,
// belief-free). Otherwise pick the max-EV plan under the belief + series state.
Solution solve_lead(const std::array<int, 13> &hand, const Belief &b,
                    const SeriesTable &t, int my_pts, int opp_pts);

// Response position (point 7): opponent just played `last_move_id` and now holds
// exactly 1 card; we must respond. A multi-card / bomb response regains the lead
// (continuation = solve_lead on the remainder); a single response of rank sr is
// beaten iff opp's card X > sr, else regains the lead. Returns the best first
// response move (re-solve as a lead position afterwards). moves is empty iff we
// have no legal response (forced pass -> we lose).
Solution solve_response(const std::array<int, 13> &hand, int last_move_id,
                        const Belief &b, const SeriesTable &t, int my_pts,
                        int opp_pts);

}  // namespace opp1

#endif
