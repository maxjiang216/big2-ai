// Standalone precompute: opponent has 1 card, maximize second-smallest single.
// Build: make tablebase_opp1_gen   (from repo root)
// Run:   ./bin/tablebase_opp1_gen [out.bin] [samples.txt]

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define LEGAL_MOVES_SIZE 472

const int kSTRAIGHT5_START = 325;
const int kSTRAIGHT6_START = kSTRAIGHT5_START + 10;
const int kSTRAIGHT7_START = kSTRAIGHT6_START + 9;
const int kSTRAIGHT8_START = kSTRAIGHT7_START + 8;
const int kSTRAIGHT9_START = kSTRAIGHT8_START + 7;
const int kSTRAIGHT10_START = kSTRAIGHT9_START + 6;
const int kSTRAIGHT11_START = kSTRAIGHT10_START + 5;
const int kSTRAIGHT12_START = kSTRAIGHT11_START + 4;
const int kSTRAIGHT13_START = kSTRAIGHT12_START + 3;
const int kDOUBLESTRAIGHT2_START = kSTRAIGHT13_START + 1;

#include "../core/move_to_cards.inc"

namespace {

constexpr int RANKS = 13;
constexpr int MAX_CARDS = 15;
constexpr std::array<uint8_t, RANKS> MAX_COUNTS{4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                4, 3, 1};

using Hand = std::array<uint8_t, RANKS>;

struct HandHash {
  std::size_t operator()(const Hand &h) const noexcept {
    std::size_t hash = 14695981039346656037ull;
    for (uint8_t c : h) {
      hash ^= c;
      hash *= 1099511628211ull;
    }
    return hash;
  }
};

struct Result {
  uint8_t score{};
  uint16_t first_move_id{};
};

std::unordered_map<Hand, Result, HandHash> MEMO;

bool can_subtract(const Hand &h, int mid) {
  for (int i = 0; i < 13; ++i) {
    if (h[i] < MOVE_TO_CARDS[mid][i])
      return false;
  }
  return true;
}

void subtract(Hand &h, int mid) {
  for (int i = 0; i < 13; ++i)
    h[i] -= static_cast<uint8_t>(MOVE_TO_CARDS[mid][i]);
}

// Second-smallest rank index among remaining cards, or 15 if <=1 card left.
uint8_t second_smallest_metric(const Hand &h) {
  std::vector<int> ranks;
  ranks.reserve(13);
  for (int r = 0; r < RANKS; ++r) {
    if (h[r] > 0)
      ranks.push_back(r);
  }
  if (ranks.size() <= 1)
    return 15;
  std::sort(ranks.begin(), ranks.end());
  return static_cast<uint8_t>(ranks[1]);
}

void remove_bombs_with_aux(Hand &h) {
  for (;;) {
    int br = -1;
    for (int r = 0; r <= 10; ++r) {
      if (h[r] == 4) {
        br = r;
        break;
      }
    }
    if (br == -1 && h[11] == 3)
      br = 11;
    if (br == -1)
      break;

    if (br == 11)
      h[11] -= 3;
    else
      h[br] -= 4;

    int aux = -1;
    for (int s = 0; s < RANKS; ++s) {
      if (h[s] == 1) {
        aux = s;
        break;
      }
    }
    if (aux != -1)
      h[aux] -= 1;
  }
}

void remove_all_triples(Hand &h) {
  for (int r = 0; r <= 10; ++r) {
    if (h[r] == 3)
      h[r] = 0;
  }
}

void remove_all_pairs(Hand &h) {
  for (int r = 0; r < RANKS; ++r) {
    if (h[r] == 2)
      h[r] = 0;
  }
}

void greedy_longest_straights(Hand &h) {
  for (;;) {
    int best_mid = -1;
    int best_len = 0;
    for (int mid = kSTRAIGHT5_START; mid < kDOUBLESTRAIGHT2_START; ++mid) {
      if (!can_subtract(h, mid))
        continue;
      int len = MOVE_TO_CARDS[mid][13];
      if (len > best_len) {
        best_len = len;
        best_mid = mid;
      } else if (len == best_len && (best_mid == -1 || mid < best_mid)) {
        best_mid = mid;
      }
    }
    if (best_mid == -1)
      break;
    subtract(h, best_mid);
  }
}

uint8_t default_evaluate(Hand h) {
  remove_bombs_with_aux(h);
  remove_all_triples(h);
  remove_all_pairs(h);
  greedy_longest_straights(h);
  return second_smallest_metric(h);
}

std::vector<int> straight_move_ids() {
  std::vector<int> mids;
  mids.reserve(kDOUBLESTRAIGHT2_START - kSTRAIGHT5_START);
  for (int mid = kSTRAIGHT5_START; mid < kDOUBLESTRAIGHT2_START; ++mid)
    mids.push_back(mid);
  return mids;
}

const std::vector<int> kStraightMids = straight_move_ids();

// High card rank (Move::rank) for a regular straight move id; matches move.cpp.
int straight_high_rank(int mid) {
  if (mid >= kSTRAIGHT5_START && mid < kSTRAIGHT6_START)
    return mid - kSTRAIGHT5_START + 6;
  if (mid >= kSTRAIGHT6_START && mid < kSTRAIGHT7_START)
    return mid - kSTRAIGHT6_START + 7;
  if (mid >= kSTRAIGHT7_START && mid < kSTRAIGHT8_START)
    return mid - kSTRAIGHT7_START + 8;
  if (mid >= kSTRAIGHT8_START && mid < kSTRAIGHT9_START)
    return mid - kSTRAIGHT8_START + 9;
  if (mid >= kSTRAIGHT9_START && mid < kSTRAIGHT10_START)
    return mid - kSTRAIGHT9_START + 10;
  if (mid >= kSTRAIGHT10_START && mid < kSTRAIGHT11_START)
    return mid - kSTRAIGHT10_START + 11;
  if (mid >= kSTRAIGHT11_START && mid < kSTRAIGHT12_START)
    return mid - kSTRAIGHT11_START + 12;
  if (mid >= kSTRAIGHT12_START && mid < kSTRAIGHT13_START)
    return mid - kSTRAIGHT12_START + 13;
  if (mid == kSTRAIGHT13_START)
    return 15;
  return -1;
}

int straight_len(int mid) { return MOVE_TO_CARDS[mid][13]; }

// When two straights yield the same optimal score, prefer longer, then lower
// high-card rank (weaker top card).
bool better_straight(int mid_new, int mid_cur) {
  int ln = straight_len(mid_new), lc = straight_len(mid_cur);
  if (ln != lc)
    return ln > lc;
  return straight_high_rank(mid_new) < straight_high_rank(mid_cur);
}

Result fun(const Hand &hand) {
  if (auto it = MEMO.find(hand); it != MEMO.end())
    return it->second;

  const uint8_t def = default_evaluate(hand);
  uint8_t best_score = def;
  uint16_t best_first = 0;

  for (int mid : kStraightMids) {
    if (!can_subtract(hand, mid))
      continue;
    Hand h2 = hand;
    subtract(h2, mid);
    Result sub = fun(h2);
    if (sub.score > best_score) {
      best_score = sub.score;
      best_first = static_cast<uint16_t>(mid);
    } else if (sub.score == best_score && best_score > def && best_first != 0) {
      if (better_straight(mid, static_cast<int>(best_first)))
        best_first = static_cast<uint16_t>(mid);
    }
  }

  if (best_score == def)
    best_first = 0;

  Result res{best_score, best_first};
  MEMO.emplace(hand, res);
  return res;
}

void gen_rec(int idx, int total, Hand &hand, std::vector<Hand> &out) {
  if (idx == RANKS) {
    out.push_back(hand);
    return;
  }
  const uint8_t cap = MAX_COUNTS[idx];
  for (uint8_t c = 0; c <= cap; ++c) {
    int new_total = total + c;
    if (new_total > MAX_CARDS)
      break;
    hand[idx] = c;
    gen_rec(idx + 1, new_total, hand, out);
  }
  hand[idx] = 0;
}

// Rank index i = 0..12 → 3,4,…,10,J,Q,K,A,2 (readable labels).
std::string rank_label(int idx) {
  const int rank = idx + 3;
  if (rank == 15)
    return "2";
  if (rank >= 3 && rank <= 9)
    return std::string(1, static_cast<char>('0' + rank));
  if (rank == 10)
    return "10";
  if (rank == 11)
    return "J";
  if (rank == 12)
    return "Q";
  if (rank == 13)
    return "K";
  if (rank == 14)
    return "A";
  return "?";
}

// All cards in hand, space-separated (multiset order 3..2).
std::string hand_spaced(const Hand &h) {
  std::string s;
  for (int i = 0; i < RANKS; ++i) {
    for (int c = 0; c < h[i]; ++c) {
      if (!s.empty())
        s += ' ';
      s += rank_label(i);
    }
  }
  return s;
}

// Cards in the straight move (from MOVE_TO_CARDS), space-separated.
std::string straight_cards_spaced(int mid) {
  std::string s;
  for (int i = 0; i < RANKS; ++i) {
    int n = MOVE_TO_CARDS[mid][i];
    for (int k = 0; k < n; ++k) {
      if (!s.empty())
        s += ' ';
      s += rank_label(i);
    }
  }
  return s;
}

// Metric: second-smallest rank index 0..12, or 15 = ≤1 card left / won line.
std::string metric_pretty(uint8_t m) {
  if (m == 15)
    return "win (≤1 single left)";
  return std::string("second-smallest = ") + rank_label(static_cast<int>(m)) +
         " (index " + std::to_string(static_cast<int>(m)) + ")";
}

// A row is exported iff playing the stored straight first strictly beats going
// to default_evaluate immediately. Runtime lookup: play straight, re-query new
// hand; if not in table, switch to default_evaluate order.
bool strictly_needs_straight_prefix(const Hand &hd, const Result &r) {
  const uint8_t def = default_evaluate(hd);
  if (r.score <= def)
    return false;
  if (r.first_move_id == 0)
    return false;
  if (!can_subtract(hd, static_cast<int>(r.first_move_id)))
    return false;
  return true;
}

void write_samples(const std::vector<std::pair<Hand, Result>> &entries,
                   const char *path, std::size_t max_lines) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "Warning: cannot open sample file " << path << "\n";
    return;
  }
  f << "# Generated by: ./bin/tablebase_opp1_gen [out.bin] [this_file]\n"
       "#\n"
       "# Not the same as research/endgame_straight_dp_sample.txt from\n"
       "# endgame.cpp (that file is comma-separated rank counts + score).\n"
       "#\n"
       "# Each block: a hand where playing the listed straight first strictly\n"
       "# beats using the full default order (bombs→triples→pairs→straights)\n"
       "# immediately. Metrics are the DP objective: second-smallest rank after\n"
       "# optimal play-through, or \"win\" if at most one single remains.\n"
       "#\n\n";

  std::size_t n = std::min(max_lines, entries.size());
  for (std::size_t i = 0; i < n; ++i) {
    const auto &[hd, r] = entries[i];
    const uint8_t def = default_evaluate(hd);
    const int mid = static_cast<int>(r.first_move_id);
    f << "Hand: " << hand_spaced(hd) << "\n";
    f << "  if default order now: " << metric_pretty(def) << "\n";
    f << "  if optimal line:       " << metric_pretty(r.score) << "\n";
    f << "  play straight first:   " << straight_cards_spaced(mid) << "\n";
    f << "\n";
  }
  f.close();
  std::cerr << "Wrote " << n << " sample cases to " << path << "\n";
}

int run(int argc, char **argv) {
  const char *out_path =
      (argc >= 2) ? argv[1] : "tablebase_opp1.bin";
  const char *sample_path =
      (argc >= 3) ? argv[2] : "tablebase_opp1_samples.txt";

  std::vector<Hand> hands;
  Hand h{};
  gen_rec(0, 0, h, hands);

  std::cerr << "Enumerated " << hands.size() << " hands. Running DP...\n";

  std::array<std::uint64_t, 16> score_hist{};
  std::vector<std::pair<Hand, Result>> entries;
  entries.reserve(500000);

  for (const auto &hd : hands) {
    Result r = fun(hd);
    if (r.score < 16)
      ++score_hist[r.score];
    if (strictly_needs_straight_prefix(hd, r))
      entries.emplace_back(hd, r);
  }

  std::sort(entries.begin(), entries.end(),
            [](const std::pair<Hand, Result> &a,
               const std::pair<Hand, Result> &b) { return a.first < b.first; });

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "Cannot open " << out_path << "\n";
    return 1;
  }

  const char magic[4] = {'O', 'P', 'P', '1'};
  out.write(magic, 4);

  std::uint32_t n = static_cast<std::uint32_t>(entries.size());
  out.write(reinterpret_cast<const char *>(&n), 4);

  for (const auto &[hd, r] : entries) {
    out.write(reinterpret_cast<const char *>(hd.data()), 13);
    char buf[3];
    buf[0] = static_cast<char>(r.score);
    buf[1] = static_cast<char>(r.first_move_id & 0xFF);
    buf[2] = static_cast<char>((r.first_move_id >> 8) & 0xFF);
    out.write(buf, 3);
  }

  out.close();

  std::cerr << "Wrote " << n
            << " non-trivial entries (strict: straight-prefix strictly improves "
               "vs default) to "
            << out_path << "\n";
  std::cerr << "MEMO size: " << MEMO.size() << "\n";
  std::cerr << "Score histogram (all states, metric 0..15):\n";
  for (int s = 0; s <= 15; ++s)
    std::cerr << "  " << s << ": " << score_hist[s] << "\n";

  write_samples(entries, sample_path, 400);

  return 0;
}

} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
