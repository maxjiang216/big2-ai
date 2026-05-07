// Analyze the move-prob table at gen-N.
// Currently keyed by (player_move, opp_count) with 17·468 = 7956 states.
// Each state stores 468 floats (response distribution).
//
// Reports:
//  - Hit distribution: how often is each player_move queried during play?
//  - Per-state entropy and max-prob: how peaked are the distributions?
//  - Variance across opp_count for the same player_move: does opp_count
//    actually matter for response prediction, or could we collapse it?
//  - Top responses for representative player_moves.

#include "core/move.h"
#include "core/util.h"
#include "typed_search/move_prob_table.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using typed_search::MoveProbTable;

namespace {

// Load main table directly by parsing the binary format (we don't have a
// public iterator on the table). Format:
//   "MPRB" magic + u32 count + n × (u16 player_move, u8 opp_count, u8 reserved,
//                                    468 × f32 counts)
struct Entry {
  int player_move;
  int opp_count;
  std::array<float, 468> counts;
};

std::vector<Entry> load_main(const std::string &path) {
  std::vector<Entry> out;
  std::ifstream in(path, std::ios::binary);
  if (!in) return out;
  char magic[4];
  in.read(magic, 4);
  if (magic[0] != 'M' || magic[1] != 'P' || magic[2] != 'R' ||
      magic[3] != 'B') {
    fprintf(stderr, "bad magic in %s\n", path.c_str());
    return out;
  }
  uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), 4);
  out.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    Entry e;
    uint16_t pm;
    uint8_t oc, reserved;
    in.read(reinterpret_cast<char *>(&pm), 2);
    in.read(reinterpret_cast<char *>(&oc), 1);
    in.read(reinterpret_cast<char *>(&reserved), 1);
    in.read(reinterpret_cast<char *>(e.counts.data()),
             sizeof(float) * 468);
    if (!in) break;
    e.player_move = pm;
    e.opp_count = oc;
    out.push_back(std::move(e));
  }
  return out;
}

float total(const std::array<float, 468> &c) {
  float s = 0;
  for (float x : c) s += x;
  return s;
}

float entropy(const std::array<float, 468> &c) {
  float s = total(c);
  if (s <= 0) return 0;
  float h = 0;
  for (float x : c) {
    if (x <= 0) continue;
    float p = x / s;
    h -= p * std::log2(p);
  }
  return h;
}

float max_prob(const std::array<float, 468> &c) {
  float s = total(c);
  if (s <= 0) return 0;
  float m = 0;
  for (float x : c) {
    float p = x / s;
    if (p > m) m = p;
  }
  return m;
}

std::string move_str(int mid) {
  if (mid == kPASS) return "PASS";
  Move m(mid);
  std::ostringstream os;
  os << m;
  return os.str();
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string dir = "data/typed_search_v6";
  if (argc >= 2) dir = argv[1];

  auto entries = load_main(dir + "/mp_main.bin");
  fprintf(stderr, "Loaded %zu mp_main entries\n", entries.size());

  // Aggregate across all entries: total visits per player_move (sum of
  // counts across all opp_counts and all responses).
  std::map<int, double> visits_by_pm;
  std::map<int, double> visits_by_pair;  // key = pm*32 + oc
  for (const auto &e : entries) {
    double s = total(e.counts);
    visits_by_pm[e.player_move] += s;
    visits_by_pair[e.player_move * 32 + e.opp_count] += s;
  }

  printf("=== Top 25 player_moves by total visits ===\n");
  std::vector<std::pair<int, double>> pm_list(visits_by_pm.begin(),
                                                visits_by_pm.end());
  std::sort(pm_list.begin(), pm_list.end(),
             [](auto &a, auto &b) { return a.second > b.second; });
  for (int i = 0; i < std::min<int>(25, pm_list.size()); ++i) {
    int pm = pm_list[i].first;
    double v = pm_list[i].second;
    printf("#%-3d move=%-12s total_visits=%.0f\n", i + 1,
            move_str(pm).c_str(), v);
  }

  printf("\n=== Per-state entropy + max-prob distribution ===\n");
  // Bucket entries by entropy band.
  int H_buckets[6] = {};   // [0,1), [1,2), [2,3), [3,4), [4,5), [5+
  for (const auto &e : entries) {
    if (total(e.counts) < 5) continue;
    float h = entropy(e.counts);
    int b = std::min(5, static_cast<int>(h));
    ++H_buckets[b];
  }
  printf("  Entropy histogram (entries with ≥5 total counts):\n");
  for (int b = 0; b < 6; ++b) {
    printf("    [%d, %d): %d\n", b, b + 1, H_buckets[b]);
  }
  // Max-prob histogram.
  int MP_buckets[10] = {};
  for (const auto &e : entries) {
    if (total(e.counts) < 5) continue;
    float mp = max_prob(e.counts);
    int b = std::min(9, static_cast<int>(mp * 10));
    ++MP_buckets[b];
  }
  printf("  Max-prob histogram (entries with ≥5 total counts):\n");
  for (int b = 0; b < 10; ++b) {
    printf("    [%.1f, %.1f): %d\n", b * 0.1, (b + 1) * 0.1, MP_buckets[b]);
  }

  printf("\n=== Does opp_count matter? Variance across opp_count for same "
          "player_move ===\n");
  // For each player_move, compute the L1 distance of the response
  // distribution between opp_count buckets. If small, opp_count is not
  // very informative for that move — we could collapse opp_count buckets.
  // Group opp_count into 4 buckets like our hand-size buckets.
  auto oc_bucket = [](int oc) {
    if (oc <= 4) return 0;
    if (oc <= 7) return 1;
    if (oc <= 11) return 2;
    return 3;
  };
  std::map<int, std::array<std::array<double, 468>, 4>> agg;
  std::map<int, std::array<double, 4>> bucket_visits;
  for (const auto &e : entries) {
    int b = oc_bucket(e.opp_count);
    auto &dst = agg[e.player_move][b];
    for (int i = 0; i < 468; ++i) dst[i] += e.counts[i];
    bucket_visits[e.player_move][b] += total(e.counts);
  }
  // For top-visited player_moves, report mean L1 distance across the
  // 4 opp_count buckets.
  printf("  player_move (visits)   mean L1(p_b - p_avg) across opp_count buckets\n");
  for (int i = 0; i < std::min<int>(20, pm_list.size()); ++i) {
    int pm = pm_list[i].first;
    auto &buckets = agg[pm];
    auto &visits = bucket_visits[pm];
    double total_v = 0;
    for (int b = 0; b < 4; ++b) total_v += visits[b];
    if (total_v <= 0) continue;
    // Compute average distribution across all buckets, then L1 distance from each
    // bucket to that average.
    std::array<double, 468> avg{};
    for (int b = 0; b < 4; ++b) {
      double bv = visits[b];
      if (bv <= 0) continue;
      for (int j = 0; j < 468; ++j) avg[j] += buckets[b][j];
    }
    for (int j = 0; j < 468; ++j) avg[j] /= total_v;
    // Compute L1 per bucket.
    double l1_sum = 0;
    int n_buckets = 0;
    for (int b = 0; b < 4; ++b) {
      double bv = visits[b];
      if (bv <= 0) continue;
      double l1 = 0;
      for (int j = 0; j < 468; ++j) {
        l1 += std::abs(buckets[b][j] / bv - avg[j]);
      }
      l1_sum += l1;
      ++n_buckets;
    }
    double mean_l1 = (n_buckets > 0) ? l1_sum / n_buckets : 0;
    printf("    %-12s (%.0f) -> mean L1=%.3f across %d buckets\n",
            move_str(pm).c_str(), pm_list[i].second, mean_l1, n_buckets);
  }
  return 0;
}
