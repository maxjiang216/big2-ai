// opp-1 lead endgame: proven-optimal solver vs the straight-blind heuristic.
//
// Current az_search uses opp1_series_move only when the hand has NO straight
// (then it is optimal); straight-holding hands fall through to MCTS. This
// measures the gap if that fallback were straight-blind: regret = optimal EV
// (solver) - heuristic EV, where the heuristic is opp1_series_move applied to
// every hand. It is an UPPER BOUND on the lead-case gain (MCTS may already find
// some straights). Belief over opp's card: proportional to remaining deck count.
//
// Usage: ./bin/opp1_compare [--per N] [--seed S]

#include "move.h"
#include "opp1_solver.h"
#include "series.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 13> kCap = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1};

// Realistic weighting: our hand-size distribution at trigger states (greedy
// self-play, from bin/opp1_freq), index = size.
constexpr std::array<double, 17> kSizeWeight = {
    0,      .1848, .1732, .1423, .1062, .0795, .0640, .0500, .0466,
    .0368,  .0317, .0277, .0241, .0214, .0104, .0014, 0};

std::array<int, 13> sample_hand(int n, std::mt19937 &rng) {
  std::vector<int> pool;
  for (int r = 0; r < 13; ++r)
    for (int c = 0; c < kCap[r]; ++c) pool.push_back(r);
  std::shuffle(pool.begin(), pool.end(), rng);
  std::array<int, 13> h{};
  for (int i = 0; i < n; ++i) h[pool[i]]++;
  return h;
}

// Belief over opp's single rank ~ remaining deck count per rank.
opp1::Belief belief_for(const std::array<int, 13> &h) {
  opp1::Belief b{};
  double tot = 0;
  for (int r = 0; r < 13; ++r) { b[r] = std::max(0, kCap[r] - h[r]); tot += b[r]; }
  for (int r = 0; r < 13; ++r) b[r] = tot > 0 ? b[r] / tot : 0;
  return b;
}

// Exposed-singles set of the straight-blind heuristic line.
opp1::Plan heuristic_plan(std::array<int, 13> h) {
  opp1::Plan p;
  for (;;) {
    int mid = opp1_series_move(h);
    if (mid == kPASS) break;
    Move m(mid);
    if (m.combination == Move::Combination::kSingle) p.exposed.push_back(m.rank);
    const auto &c = MOVE_TO_CARDS[mid];
    for (int r = 0; r < 13; ++r) h[r] -= c[r];
  }
  std::sort(p.exposed.begin(), p.exposed.end());
  return p;
}

bool has_straight(const std::array<int, 13> &h) {
  for (int id : compute_legal_moves(h, Move(Move::Combination::kPass)))
    if (all_moves()[id].combination >= Move::Combination::kStraight5) return true;
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  int per = 20000;
  unsigned seed = 7;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--per" && i + 1 < argc) per = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoul(argv[++i]);
  }
  SeriesTable bt;  // binary -> EV is win probability

  std::printf("size   optimalWP  heurWP   regret   diffMove  (straight hands only:)  regretStr\n");
  double w_opt = 0, w_heur = 0, w_reg = 0, wsum = 0;
  for (int s = 2; s <= 16; ++s) {
    std::mt19937 rng(seed + s);
    double opt = 0, heur = 0, diff = 0, reg_str = 0;
    long long n = 0, nstr = 0;
    for (int g = 0; g < per; ++g) {
      auto h = sample_hand(s, rng);
      auto bel = belief_for(h);
      auto sol = opp1::solve_lead(h, bel, bt, 0, 0);
      // Solver chosen plan = first plan whose move-seq matches sol.moves; recompute
      // its EV directly by rebuilding via enumerate + matching exposed is overkill,
      // so re-evaluate using the chosen plan's exposed via a fresh solve value.
      // Simpler: optimal EV = EV of the solver's chosen line.
      // Rebuild chosen plan's exposed by replaying sol.moves.
      opp1::Plan chosen;
      {
        std::array<int, 13> hh = h;
        for (int mid : sol.moves) {
          Move m(mid);
          if (m.combination == Move::Combination::kSingle) chosen.exposed.push_back(m.rank);
          const auto &c = MOVE_TO_CARDS[mid];
          for (int r = 0; r < 13; ++r) hh[r] -= c[r];
        }
        std::sort(chosen.exposed.begin(), chosen.exposed.end());
      }
      auto hp = heuristic_plan(h);
      double vo = opp1::ev(chosen, bel, bt, 0, 0);
      double vh = opp1::ev(hp, bel, bt, 0, 0);
      opt += vo; heur += vh;
      if (chosen.exposed != hp.exposed) diff += 1;
      if (has_straight(h)) { nstr++; reg_str += (vo - vh); }
      n++;
    }
    double regret = (opt - heur) / n;
    std::printf("%4d    %7.4f  %7.4f  %7.4f   %6.2f%%              %7.4f\n",
                s, opt / n, heur / n, regret, 100.0 * diff / n,
                nstr ? reg_str / nstr : 0.0);
    double wt = kSizeWeight[s];
    w_opt += wt * opt / n; w_heur += wt * heur / n; w_reg += wt * regret; wsum += wt;
  }
  std::printf("\nWeighted by trigger size dist: optimalWP=%.4f heurWP=%.4f "
              "regret=%.4f (coverage %.2f)\n",
              w_opt / wsum, w_heur / wsum, w_reg / wsum, wsum);
  return 0;
}
