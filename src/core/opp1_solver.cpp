#include "opp1_solver.h"

#include "move.h"
#include "util.h"

#include <algorithm>
#include <set>

namespace opp1 {

namespace {

using C = Move::Combination;

bool is_straight_combo(C c) { return c >= C::kStraight5 && c <= C::kStraight13; }

// Footprint of a move id as rank counts (MOVE_TO_CARDS[id][0..12]).
const std::array<int, 14> &footprint(int id) { return MOVE_TO_CARDS[id]; }

bool playable(const std::array<int, 13> &rem, int id) {
  const auto &c = footprint(id);
  for (int r = 0; r < 13; ++r)
    if (c[r] > rem[r])
      return false;
  return true;
}

// Build the trivial line for `rem`: bombs (with low loose singles buried as
// auxiliaries), then triples/doubles, then exposed singles highest-first.
Plan build_trivial(const std::array<int, 13> &rem) {
  Plan p;
  // Loose singles (count == 1), ascending by rank value.
  std::vector<int> loose;             // rank values 3..15
  std::vector<int> loose_idx;         // rank indices 0..12
  for (int r = 0; r < 13; ++r)
    if (rem[r] == 1) {
      loose.push_back(r + 3);
      loose_idx.push_back(r);
    }

  // Bomb ranks (rank VALUE): four-of-a-kind (idx 0..10) or the ace bomb (3 aces).
  std::vector<int> bomb_ranks;
  for (int r = 0; r <= 10; ++r)
    if (rem[r] == 4)
      bomb_ranks.push_back(r + 3);
  bool ace_bomb = (rem[11] == 3);
  if (ace_bomb)
    bomb_ranks.push_back(14);

  // Bury the lowest loose singles EXCEPT the global-lowest (kept free for the
  // last play). With b bombs we bury loose[1 .. b]. (Burying loose[0] instead is
  // outcome-equivalent, but keeping it free is the canonical choice.)
  int nb = static_cast<int>(bomb_ranks.size());
  std::set<int> buried;  // rank values
  std::vector<int> buried_list;
  for (int i = 1; i < static_cast<int>(loose.size()) && static_cast<int>(buried_list.size()) < nb; ++i) {
    buried.insert(loose[i]);
    buried_list.push_back(loose[i]);
  }

  // Emit bombs, each carrying one buried single as auxiliary (aux 0 if none).
  size_t ti = 0;
  for (int br : bomb_ranks) {
    int aux = (ti < buried_list.size()) ? buried_list[ti++] : 0;
    p.moves.push_back(encodeMove(Move(C::kBomb, br, aux)));
  }
  // Triples (non-ace; the ace triple, if any, was used as a bomb above).
  for (int r = 0; r <= 10; ++r)
    if (rem[r] == 3)
      p.moves.push_back(encodeMove(Move(C::kTriple, r + 3)));
  // Doubles.
  for (int r = 0; r < 13; ++r)
    if (rem[r] == 2)
      p.moves.push_back(encodeMove(Move(C::kDouble, r + 3)));

  // Exposed singles = loose minus buried; play highest-first, lowest last.
  std::vector<int> exposed;
  for (int v : loose)
    if (!buried.count(v))
      exposed.push_back(v);
  std::sort(exposed.begin(), exposed.end());  // ascending
  p.exposed = exposed;
  for (auto it = exposed.rbegin(); it != exposed.rend(); ++it)
    p.moves.push_back(encodeMove(Move(C::kSingle, *it)));
  return p;
}

constexpr int kNodeCap = 200000;

void enumerate_rec(const std::array<int, 13> &rem, const std::vector<int> &straights,
                   size_t start, std::vector<int> &chosen,
                   std::set<std::array<int, 13>> &seen_remainders,
                   std::vector<Plan> &out, int &nodes) {
  if (nodes++ > kNodeCap)
    return;
  // Emit the line that stops adding straights here.
  Plan t = build_trivial(rem);
  Plan p;
  p.moves = chosen;
  p.moves.insert(p.moves.end(), t.moves.begin(), t.moves.end());
  p.exposed = t.exposed;
  out.push_back(std::move(p));

  for (size_t i = start; i < straights.size(); ++i) {
    int s = straights[i];
    if (!playable(rem, s))
      continue;
    std::array<int, 13> rem2 = rem;
    const auto &c = footprint(s);
    for (int r = 0; r < 13; ++r)
      rem2[r] -= c[r];
    // Dedup identical remainders reached via the same straight-prefix budget.
    if (!seen_remainders.insert(rem2).second)
      continue;
    chosen.push_back(s);
    enumerate_rec(rem2, straights, i, chosen, seen_remainders, out, nodes);
    chosen.pop_back();
  }
}

}  // namespace

int k_below(const Plan &p, int opp_rank) {
  int k = 0;
  for (int s : p.exposed)
    if (s < opp_rank)
      ++k;
  return k;
}

double value_vs_rank(const Plan &p, int opp_rank, const SeriesTable &t,
                     int my_pts, int opp_pts) {
  int k = k_below(p, opp_rank);
  if (k <= 1) {  // we win; opp keeps its 1 card -> we score points_for_cards(1)=1
    if (!t.loaded)
      return 1.0;
    return series_value_after_win(t, my_pts, opp_pts, 1);
  }
  // we lose by (k-1) cards; opp scores points_for_cards(k-1)
  if (!t.loaded)
    return 0.0;
  return 1.0 - series_value_after_win(t, opp_pts, my_pts, k - 1);
}

double ev(const Plan &p, const Belief &b, const SeriesTable &t, int my_pts,
          int opp_pts) {
  double e = 0.0;
  for (int r = 0; r < 13; ++r)
    if (b[r] > 0.0)
      e += b[r] * value_vs_rank(p, r + 3, t, my_pts, opp_pts);
  return e;
}

bool weakly_dominates(const Plan &a, const Plan &b) {
  for (int x = 3; x <= 15; ++x)
    if (k_below(a, x) > k_below(b, x))
      return false;
  return true;
}

std::vector<Plan> enumerate_plans(const std::array<int, 13> &hand) {
  std::vector<int> straights;
  for (int id : compute_legal_moves(hand, Move(C::kPass)))
    if (is_straight_combo(all_moves()[id].combination))
      straights.push_back(id);
  std::sort(straights.begin(), straights.end());

  std::vector<Plan> out;
  std::vector<int> chosen;
  std::set<std::array<int, 13>> seen;
  int nodes = 0;
  enumerate_rec(hand, straights, 0, chosen, seen, out, nodes);
  return out;
}

Solution solve_lead(const std::array<int, 13> &hand, const Belief &b,
                    const SeriesTable &t, int my_pts, int opp_pts) {
  std::vector<Plan> plans = enumerate_plans(hand);
  Solution sol;
  if (plans.empty())
    return sol;

  // A dominator achieves the pointwise-min k(X) for every opp rank.
  std::array<int, 16> kmin;
  kmin.fill(1 << 30);
  for (const Plan &p : plans)
    for (int x = 3; x <= 15; ++x)
      kmin[x] = std::min(kmin[x], k_below(p, x));
  for (const Plan &p : plans) {
    bool dom = true;
    for (int x = 3; x <= 15; ++x)
      if (k_below(p, x) != kmin[x]) {
        dom = false;
        break;
      }
    if (dom) {
      sol.moves = p.moves;
      sol.by_dominance = true;
      return sol;
    }
  }

  // No dominator: maximise EV under the belief + series state.
  double best = -1.0;
  for (const Plan &p : plans) {
    double e = ev(p, b, t, my_pts, opp_pts);
    if (e > best) {
      best = e;
      sol.moves = p.moves;
    }
  }
  sol.by_dominance = false;
  return sol;
}

namespace {

double win_val(const SeriesTable &t, int my_pts, int opp_pts) {
  // We empty; opp keeps its 1 card -> we score points_for_cards(1) = 1.
  return t.loaded ? series_value_after_win(t, my_pts, opp_pts, 1) : 1.0;
}
double loss_val(const SeriesTable &t, int my_pts, int opp_pts, int our_cards) {
  return t.loaded ? 1.0 - series_value_after_win(t, opp_pts, my_pts, our_cards)
                  : 0.0;
}

// Exposed singles of an already-chosen line (the kSingle moves), from `hand`.
std::vector<int> exposed_of(const std::array<int, 13> &hand,
                            const std::vector<int> &moves) {
  std::vector<int> ex;
  for (int mid : moves)
    if (all_moves()[mid].combination == C::kSingle)
      ex.push_back(all_moves()[mid].rank);
  std::sort(ex.begin(), ex.end());
  (void)hand;
  return ex;
}

}  // namespace

Solution solve_response(const std::array<int, 13> &hand, int last_move_id,
                        const Belief &b, const SeriesTable &t, int my_pts,
                        int opp_pts) {
  Solution sol;
  int our_size = 0;
  for (int r = 0; r < 13; ++r) our_size += hand[r];

  std::vector<int> legal = compute_legal_moves(hand, Move(last_move_id));
  double best_ev = -1.0;
  for (int r : legal) {
    if (r == kPASS) continue;
    const auto &c = footprint(r);
    std::array<int, 13> rem = hand;
    int rem_total = our_size;
    for (int i = 0; i < 13; ++i) rem[i] -= c[i];
    rem_total -= c[13];

    double ev_r;
    if (rem_total == 0) {
      ev_r = win_val(t, my_pts, opp_pts);  // our response empties our hand -> win
    } else {
      // Optimal continuation after we (likely) regain the lead.
      Solution cont = solve_lead(rem, b, t, my_pts, opp_pts);
      Plan cp;
      cp.exposed = exposed_of(rem, cont.moves);
      Move m(r);
      bool is_single = (m.combination == C::kSingle);
      ev_r = 0.0;
      for (int xi = 0; xi < 13; ++xi) {
        if (b[xi] <= 0.0) continue;
        int X = xi + 3;
        double v;
        if (is_single && X > m.rank)
          v = loss_val(t, my_pts, opp_pts, rem_total);  // opp beats our single
        else
          v = value_vs_rank(cp, X, t, my_pts, opp_pts);
        ev_r += b[xi] * v;
      }
    }
    if (ev_r > best_ev) {
      best_ev = ev_r;
      sol.moves.assign(1, r);
    }
  }
  // No legal response -> forced pass -> opp leads its card and wins; moves empty.
  sol.by_dominance = false;
  return sol;
}

}  // namespace opp1
