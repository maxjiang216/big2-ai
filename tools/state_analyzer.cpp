// Analyze the extended-eval table: which feature combos are hit most, which
// have the most "off" values relative to the coarser main table, where the
// finer granularity actually buys real signal vs where it's wasted.
//
// Decodes state IDs back into the (init, opp_b, our_b, has_bomb, st_tier,
// ds_tier, singles[4], doubles[3], triples[2]) tuple and reports.
//
// Usage: state_analyzer [tables_dir]   (default data/typed_search_v6)

#include "core/util.h"
#include "typed_search/eval_features.h"
#include "typed_search/eval_table.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using typed_search::EvalTable;

namespace {

// Mixed-radix encoding used by extended_state_id (in this exact order):
//   init(2) opp_b(4) our_b(4) hb(2)
//   s_3(2) s_4(2) s_57(4) s_med(3) s_lg(3) s_top(3)        -- singles
//   doubles(3)                                              -- all doubles
//   trip_sm(3) trip_lg(3)                                   -- triples (2 regions)
//   opp_2(2) opp_A(3) opp_high(3)                           -- opp threat features
constexpr int kRadices[16] = {2, 4, 4, 2, 2, 2, 4, 3, 3, 3, 3, 3, 3, 2, 3, 3};
constexpr const char *kNames[16] = {"init", "opp_b", "our_b", "bomb",
                                      "s_3", "s_4", "s_57", "s_med", "s_lg",
                                      "s_top", "dbl",
                                      "trp_sm", "trp_lg",
                                      "opp_2", "opp_A", "opp_high"};

// Decode ext state ID to a 16-vector of feature bucket values.
void decode_ext(uint32_t s, int out[16]) {
  for (int i = 15; i >= 0; --i) {
    out[i] = static_cast<int>(s % kRadices[i]);
    s /= kRadices[i];
  }
}

// Encode the same 16-vector to a state ID (round-trip check).
uint32_t encode_ext(const int v[16]) {
  uint32_t s = 0;
  for (int i = 0; i < 16; ++i) s = s * kRadices[i] + v[i];
  return s;
}

const char *bucket_label(int idx, int v) {
  static const char *init_n[] = {"opp_pass", "we_pass"};
  static const char *size_n[] = {"1-4", "5-7", "8-11", "12-16"};
  static const char *yn_n[] = {"no", "yes"};
  static const char *bk3[] = {"0", "1", "2+"};
  static const char *bk4[] = {"0", "1", "2", "3+"};
  switch (idx) {
    case 0: return init_n[v];                  // init
    case 1: case 2: return size_n[v];          // opp_b, our_b
    case 3: return yn_n[v];                    // bomb
    case 4: case 5: return yn_n[v];            // s_3, s_4
    case 6: return bk4[v];                     // s_57
    case 13: return yn_n[v];                   // opp_2
    default: return bk3[v];                    // radix-3 features
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string dir = "data/typed_search_v6";
  if (argc >= 2) dir = argv[1];

  EvalTable ext, main_t, fb;
  ext.load(dir + "/eval_extended.bin");
  main_t.load(dir + "/eval_main.bin");
  fb.load(dir + "/eval_fallback.bin");

  fprintf(stderr, "Loaded ext=%zu main=%zu fb=%zu from %s\n",
           ext.size(), main_t.size(), fb.size(), dir.c_str());

  // Snapshot all entries.
  struct Row {
    uint32_t sid;
    float visits;
    float win_prob;
    int features[16];
  };
  std::vector<Row> rows;
  rows.reserve(ext.size());
  for (const auto &kv : ext.entries()) {
    Row r;
    r.sid = kv.first;
    r.visits = kv.second.visit_count;
    r.win_prob = (r.visits > 0) ? kv.second.total_wins / r.visits : 0.0f;
    decode_ext(r.sid, r.features);
    rows.push_back(r);
  }

  printf("=== Top 25 ext cells by visit count ===\n");
  std::sort(rows.begin(), rows.end(),
             [](const Row &a, const Row &b) { return a.visits > b.visits; });
  for (int i = 0; i < std::min<int>(25, rows.size()); ++i) {
    const Row &r = rows[i];
    printf("#%-3d sid=%-9u visits=%7.0f wp=%.3f  ", i + 1, r.sid, r.visits,
            r.win_prob);
    for (int j = 0; j < 16; ++j) {
      printf("%s=%s ", kNames[j], bucket_label(j, r.features[j]));
    }
    printf("\n");
  }

  printf("\n=== Per-feature variance: how much does each feature dimension "
          "split the value? ===\n");
  // For each feature dimension f and each value v of that dimension, group
  // ext cells (≥100 visits) by f=v and compute mean win_prob. The spread
  // across values of f tells us whether dimension f matters.
  for (int f = 0; f < 16; ++f) {
    double sum_per_v[5] = {};
    double cnt_per_v[5] = {};
    double sum_visits_per_v[5] = {};
    int radix = kRadices[f];
    for (const Row &r : rows) {
      if (r.visits < 100) continue;
      int v = r.features[f];
      sum_per_v[v] += r.win_prob * r.visits;  // visit-weighted
      cnt_per_v[v] += 1.0;
      sum_visits_per_v[v] += r.visits;
    }
    printf("  %-10s ", kNames[f]);
    for (int v = 0; v < radix; ++v) {
      if (cnt_per_v[v] == 0) {
        printf("v=%d: --        ", v);
      } else {
        double wp = sum_per_v[v] / sum_visits_per_v[v];
        printf("v=%d: wp=%.3f n=%-4.0f  ", v, wp, cnt_per_v[v]);
      }
    }
    // Spread metric: max-min across values of f.
    double minw = 1.0, maxw = 0.0;
    for (int v = 0; v < radix; ++v) {
      if (cnt_per_v[v] == 0) continue;
      double wp = sum_per_v[v] / sum_visits_per_v[v];
      if (wp < minw) minw = wp;
      if (wp > maxw) maxw = wp;
    }
    printf("[Δ=%.3f]\n", maxw - minw);
  }
  return 0;
}
