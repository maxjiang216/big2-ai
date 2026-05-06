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
//   init(2) opp_b(4) our_b(4) hb(2) st(3) ds(3)
//   sm(4) med(3) lg(3) top(3)            -- singles
//   dsm(3) dmed(3) dlg(3)                -- doubles
//   tsm(3) tlg(3)                        -- triples
constexpr int kRadices[15] = {2, 4, 4, 2, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3};
constexpr const char *kNames[15] = {"init", "opp_b", "our_b", "bomb", "str",
                                      "ds_ts", "sm",   "med",   "lg",  "top",
                                      "dbl_sm", "dbl_med", "dbl_lg",
                                      "trp_sm", "trp_lg"};

// Decode ext state ID to a 15-vector of feature bucket values.
void decode_ext(uint32_t s, int out[15]) {
  for (int i = 14; i >= 0; --i) {
    out[i] = static_cast<int>(s % kRadices[i]);
    s /= kRadices[i];
  }
}

// Encode the same 15-vector to a state ID (round-trip check).
uint32_t encode_ext(const int v[15]) {
  uint32_t s = 0;
  for (int i = 0; i < 15; ++i) s = s * kRadices[i] + v[i];
  return s;
}

// Map each ext bucket index to a coarsened main-table bucket. Main has:
//   sm: 0/1/2+   (radix 3 — clamp ext sm 0-3 -> 0,1,2,2)
//   med/lg/top:  0/1+ (radix 2 — clamp 0/1/2 -> 0,1,1)
//   dsm/dmed/dlg: 0/1+ (clamp)
//   tsm/tlg:      0/1+ (clamp)
// The doubles boundary differs: ext medium=8-J, main medium=8-10. We can't
// recover that without the original hand, so we just clamp here. Result is a
// best-effort "main proxy" bucket; useful for spotting divergence.
void coarsen_to_main(const int ext[15], int main_v[15]) {
  for (int i = 0; i < 6; ++i) main_v[i] = ext[i];
  // ext singles_small (radix 4) -> main (radix 3): clamp 3 -> 2
  main_v[6] = std::min(ext[6], 2);
  // The remaining 8 features are radix-3 in ext, radix-2 in main: clamp 2 -> 1
  for (int i = 7; i < 15; ++i) main_v[i] = std::min(ext[i], 1);
}

// Encode 15-vector with main radices (different from ext).
constexpr int kMainRadices[15] = {2, 4, 4, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2};
uint32_t encode_main(const int v[15]) {
  uint32_t s = 0;
  for (int i = 0; i < 15; ++i) s = s * kMainRadices[i] + v[i];
  return s;
}

const char *bucket_label(int idx, int v) {
  // Compact human-readable bucket value names.
  static const char *init_n[] = {"opp_pass", "we_pass"};
  static const char *size_n[] = {"1-4", "5-7", "8-11", "12-16"};
  static const char *bomb_n[] = {"no", "yes"};
  static const char *tier_n[] = {"none", "weak", "strong"};
  static const char *bk3[] = {"0", "1", "2+"};
  static const char *bk4[] = {"0", "1", "2", "3+"};
  switch (idx) {
    case 0: return init_n[v];
    case 1: case 2: return size_n[v];
    case 3: return bomb_n[v];
    case 4: case 5: return tier_n[v];
    case 6: return bk4[v];
    default: return bk3[v];
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
    int features[15];
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
    for (int j = 0; j < 15; ++j) {
      printf("%s=%s ", kNames[j], bucket_label(j, r.features[j]));
    }
    printf("\n");
  }

  printf("\n=== Most divergent ext cells vs main proxy (visits>=100) ===\n");
  // For each ext cell with >=100 visits, compute the main-proxy state ID and
  // look up main's win_prob there. Sort by |ext_wp - main_wp| desc.
  struct DivRow {
    uint32_t sid;
    float visits;
    float ext_wp;
    float main_wp;
    float main_visits;
    int features[15];
  };
  std::vector<DivRow> divs;
  for (const Row &r : rows) {
    if (r.visits < 100) continue;
    int main_v[15];
    coarsen_to_main(r.features, main_v);
    uint32_t main_sid = encode_main(main_v);
    auto m = main_t.lookup(main_sid);
    DivRow d;
    d.sid = r.sid;
    d.visits = r.visits;
    d.ext_wp = r.win_prob;
    d.main_wp = m.found ? m.win_prob : 0.5f;
    d.main_visits = m.found ? m.visit_count : 0.0f;
    std::memcpy(d.features, r.features, sizeof(int) * 15);
    divs.push_back(d);
  }
  std::sort(divs.begin(), divs.end(), [](const DivRow &a, const DivRow &b) {
    return std::abs(a.ext_wp - a.main_wp) > std::abs(b.ext_wp - b.main_wp);
  });
  printf("%-3s %-9s %-7s %-6s %-6s %-7s  features\n", "#", "sid", "visits",
          "ext_wp", "main_wp", "Δ");
  for (int i = 0; i < std::min<int>(25, divs.size()); ++i) {
    const DivRow &d = divs[i];
    printf("%-3d %-9u %7.0f %.3f  %.3f  %+.3f  ", i + 1, d.sid, d.visits,
            d.ext_wp, d.main_wp, d.ext_wp - d.main_wp);
    for (int j = 0; j < 15; ++j) {
      printf("%s=%s ", kNames[j], bucket_label(j, d.features[j]));
    }
    printf("(main_visits=%.0f)\n", d.main_visits);
  }

  printf("\n=== Per-feature variance: how much does each feature dimension "
          "split the value? ===\n");
  // For each feature dimension f and each value v of that dimension, group
  // ext cells (≥100 visits) by f=v and compute mean win_prob. The spread
  // across values of f tells us whether dimension f matters.
  for (int f = 0; f < 15; ++f) {
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
