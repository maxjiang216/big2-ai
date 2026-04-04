// Count information states at "my turn" for the 2-player Big 2 variant (see RULES.md).
// Deck caps match src/core/util.h::max_cards_in_deck_for_rank.
//
// Usage:
//   ./build/research/count_turn_states --discard 0,0,...,0 --k 16
//   ./build/research/count_turn_states --discard 0,0,...,0   # all k 1..16
//   ./build/research/count_turn_states --summary

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kRanks = 13;
constexpr std::array<int, kRanks> CAP = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1};
constexpr int kTotalDeck = 48;
constexpr int kUnused = 16;
constexpr int kPlayersCards = kTotalDeck - kUnused;  // 32

bool can_assign_unused(const std::array<int, kRanks> &rem, int need) {
  if (need < 0) return false;
  int s = 0;
  for (int r = 0; r < kRanks; ++r) s += rem[r];
  if (s < need) return false;
  uint32_t bits = 1u;
  const uint32_t mask = need < 31 ? ((1u << (need + 1)) - 1u) : 0xffffffffu;
  for (int r = 0; r < kRanks; ++r) {
    const int mx = std::min(rem[r], need);
    uint32_t new_bits = 0;
    for (int u = 0; u <= mx; ++u) new_bits |= bits << u;
    bits = new_bits & mask;
  }
  return (bits >> need) & 1u;
}

uint64_t count_multisets_of_size(const std::array<int, kRanks> &limits, int k) {
  if (k < 0) return 0;
  std::vector<uint64_t> dp(static_cast<size_t>(k) + 1, 0), next(static_cast<size_t>(k) + 1);
  dp[0] = 1;
  for (int i = 0; i < kRanks; ++i) {
    std::fill(next.begin(), next.end(), 0);
    for (int s = 0; s <= k; ++s) {
      if (dp[static_cast<size_t>(s)] == 0) continue;
      const int hi = std::min(limits[i], k - s);
      for (int x = 0; x <= hi; ++x) {
        next[static_cast<size_t>(s + x)] += dp[static_cast<size_t>(s)];
      }
    }
    dp.swap(next);
  }
  return dp[static_cast<size_t>(k)];
}

int sum_d(const std::array<int, kRanks> &d) {
  int s = 0;
  for (int r = 0; r < kRanks; ++r) s += d[r];
  return s;
}

uint64_t count_hands_given_discard(int k, const std::array<int, kRanks> &d) {
  if (k < 0 || k > 16) return 0;
  const int sd = sum_d(d);
  for (int r = 0; r < kRanks; ++r) {
    if (d[r] > CAP[r]) return 0;
  }
  if (k + sd > kPlayersCards) return 0;
  const int o = kPlayersCards - k - sd;
  if (o < 0) return 0;

  std::array<int, kRanks> limits{};
  for (int r = 0; r < kRanks; ++r) {
    limits[r] = CAP[r] - d[r];
    if (limits[r] < 0) return 0;
  }

  // Empty discard: multiset count equals feasibility-inclusive count (same as Python).
  if (sd == 0) return count_multisets_of_size(limits, k);

  const int target_rem_sum = o + kUnused;
  std::array<int, kRanks> acc{};

  std::function<uint64_t(int, int)> rec;
  rec = [&](int i, int k_left) -> uint64_t {
    if (i == kRanks) {
      if (k_left != 0) return 0;
      std::array<int, kRanks> rem{};
      int sum_rem = 0;
      for (int r = 0; r < kRanks; ++r) {
        rem[r] = CAP[r] - acc[r] - d[r];
        if (rem[r] < 0) return 0;
        sum_rem += rem[r];
      }
      if (sum_rem != target_rem_sum) return 0;
      return can_assign_unused(rem, kUnused) ? UINT64_C(1) : UINT64_C(0);
    }
    uint64_t total = 0;
    const int max_m = std::min(CAP[i] - d[i], k_left);
    for (int m = 0; m <= max_m; ++m) {
      acc[i] = m;
      total += rec(i + 1, k_left - m);
    }
    acc[i] = 0;
    return total;
  };

  return rec(0, k);
}

template <typename F>
void iter_discard_multisets(int max_sum, F &&fn) {
  std::array<int, kRanks> cur{};
  std::function<void(int, int)> gen;
  gen = [&](int i, int left) {
    if (i == kRanks) {
      fn(cur);
      return;
    }
    const int hi = std::min(CAP[i], left);
    for (int x = 0; x <= hi; ++x) {
      cur[i] = x;
      gen(i + 1, left - x);
    }
  };
  gen(0, max_sum);
}

bool parse_discard(const char *s, std::array<int, kRanks> &out) {
  std::string str(s);
  for (char &c : str)
    if (c == ',') c = ' ';
  std::istringstream iss(str);
  for (int r = 0; r < kRanks; ++r) {
    if (!(iss >> out[r])) return false;
  }
  int extra = 0;
  if (iss >> extra) return false;
  return true;
}

void print_usage(const char *argv0) {
  std::cerr << "Usage:\n"
            << "  " << argv0 << " --discard c0,c1,...,c12 [--k N]\n"
            << "  " << argv0 << " --summary\n"
            << "Ranks 0..12: 3..K (4 each), aces (3), deuce (1).\n";
}

void print_u128(std::ostream &os, unsigned __int128 x) {
  if (x == 0) {
    os << 0;
    return;
  }
  char buf[64];
  int p = 0;
  while (x) {
    buf[p++] = char('0' + int(x % 10));
    x /= 10;
  }
  while (p--) os << buf[p];
}

}  // namespace

int main(int argc, char **argv) {
  bool summary = false;
  bool have_discard = false;
  bool have_k = false;
  int k_val = 0;
  std::array<int, kRanks> discard{};

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--summary") == 0) {
      summary = true;
    } else if (std::strcmp(argv[i], "--discard") == 0 && i + 1 < argc) {
      if (!parse_discard(argv[++i], discard)) {
        std::cerr << "Expected 13 rank counts after --discard\n";
        return 1;
      }
      have_discard = true;
    } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      k_val = std::atoi(argv[++i]);
      have_k = true;
    } else {
      print_usage(argv[0]);
      return 1;
    }
  }

  if (summary) {
    unsigned __int128 grand = 0;
    std::cout << "k  discard_patterns  hand_discard_pairs\n";
    for (int k = 1; k <= 16; ++k) {
      const int max_discard = kPlayersCards - k;
      uint64_t n_d = 0;
      unsigned __int128 npairs = 0;
      iter_discard_multisets(max_discard, [&](const std::array<int, kRanks> &d) {
        ++n_d;
        npairs += static_cast<unsigned __int128>(count_hands_given_discard(k, d));
      });
      grand += npairs;
      std::cout << k << "  " << n_d << "  ";
      print_u128(std::cout, npairs);
      std::cout << "\n";
    }
    std::cout << "total_pairs_all_k=";
    print_u128(std::cout, grand);
    std::cout << "\n";
    return 0;
  }

  if (!have_discard) {
    print_usage(argv[0]);
    return 1;
  }

  if (have_k) {
    std::cout << count_hands_given_discard(k_val, discard) << "\n";
    return 0;
  }

  std::cout << "k  num_hands\n";
  for (int k = 1; k <= 16; ++k) {
    std::cout << k << "  " << count_hands_given_discard(k, discard) << "\n";
  }
  return 0;
}
