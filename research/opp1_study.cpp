// How decidable is the opp-1 lead endgame? For random hands at each size (we
// lead, opp has exactly 1 card), measure:
//   dom      : a k-dominating line exists (majorization settles it, belief-free)
//   trivial  : the no-straight line is itself that dominator (straights moot)
//   straight+: a straight STRICTLY helps (trivial line is not optimal)
//   needEV   : no dominator -> must fall back to expected series value
//   plans    : mean number of candidate lines enumerated
//
// Usage: ./bin/opp1_study [--per N] [--seed S]

#include "move.h"
#include "opp1_solver.h"
#include "series.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <random>
#include <vector>

namespace {

std::array<int, 13> sample_hand(int n, std::mt19937 &rng) {
  static const std::array<int, 13> cap = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1};
  std::vector<int> pool;
  for (int r = 0; r < 13; ++r)
    for (int c = 0; c < cap[r]; ++c)
      pool.push_back(r);
  std::shuffle(pool.begin(), pool.end(), rng);
  std::array<int, 13> h{};
  for (int i = 0; i < n; ++i)
    h[pool[i]]++;
  return h;
}

bool has_straight(const std::array<int, 13> &h) {
  for (int id : compute_legal_moves(h, Move(Move::Combination::kPass)))
    if (all_moves()[id].combination >= Move::Combination::kStraight5)
      return true;
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  int per = 20000;
  unsigned seed = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--per" && i + 1 < argc) per = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = std::stoul(argv[++i]);
  }
  SeriesTable bt;  // binary: dominance is belief-free anyway
  opp1::Belief uni;
  uni.fill(1.0 / 13.0);

  std::printf("size      n  hasStr   dom  trivial  straight+  needEV   plans\n");
  for (int s = 2; s <= 16; ++s) {
    std::mt19937 rng(seed + s);
    long long n = 0, hs = 0, dom = 0, triv = 0, strp = 0, need = 0, plans_sum = 0;
    for (int g = 0; g < per; ++g) {
      auto h = sample_hand(s, rng);
      auto plans = opp1::enumerate_plans(h);
      plans_sum += (long long)plans.size();
      bool hstr = has_straight(h);
      hs += hstr;

      // dominator?
      std::array<int, 16> kmin; kmin.fill(1 << 30);
      for (auto &p : plans)
        for (int x = 3; x <= 15; ++x)
          kmin[x] = std::min(kmin[x], opp1::k_below(p, x));
      int dom_idx = -1;
      for (int i = 0; i < (int)plans.size(); ++i) {
        bool d = true;
        for (int x = 3; x <= 15; ++x)
          if (opp1::k_below(plans[i], x) != kmin[x]) { d = false; break; }
        if (d) { dom_idx = i; break; }
      }
      if (dom_idx >= 0) {
        dom++;
        // plans[0] is the trivial (no-straight) line.
        bool trivial_dom = opp1::weakly_dominates(plans[0], plans[dom_idx]) &&
                           opp1::weakly_dominates(plans[dom_idx], plans[0]);
        if (trivial_dom) triv++;
        // straight strictly helps if the dominator beats trivial somewhere.
        bool strict = false;
        for (int x = 3; x <= 15; ++x)
          if (opp1::k_below(plans[dom_idx], x) < opp1::k_below(plans[0], x)) { strict = true; break; }
        if (strict) strp++;
      } else {
        need++;
      }
      n++;
    }
    auto pc = [&](long long x) { return 100.0 * x / n; };
    std::printf("%4d %7lld  %5.1f%% %5.1f%%  %5.1f%%   %5.1f%%   %5.1f%%  %6.1f\n",
                s, n, pc(hs), pc(dom), pc(triv), pc(strp), pc(need),
                (double)plans_sum / n);
  }
  return 0;
}
