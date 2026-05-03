#include "util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <vector>

// Define BIG2_PROFILE_LEGAL_MOVES_SECTIONS when profiling with Callgrind /
// KCachegrind: splits PASS / response sections into separate noinline symbols.
#if defined(BIG2_PROFILE_LEGAL_MOVES_SECTIONS) && \
    (defined(__GNUC__) || defined(__clang__))
#define BIG2_LEGAL_SEC static __attribute__((noinline))
#else
#define BIG2_LEGAL_SEC static inline
#endif

char rankToChar(int rank) {
  if (rank == 2 || rank == 15) {
    return '2';
  }
  if (rank < 10) {
    return '0' + rank;
  }
  if (rank == 10) {
    return '0';
  }
  if (rank == 11) {
    return 'J';
  }
  if (rank == 12) {
    return 'Q';
  }
  if (rank == 13) {
    return 'K';
  }
  return 'A';
}

int g_kpass_legal_max_cards =
    KPASS_LEGAL_TABLE_BUILD ? KPASS_LEGAL_PRIMARY_MAX_CARDS : 0;

// ======================================================================
// Precomputed move metadata
// ======================================================================
namespace {

void append_pass_straights_for_dup_helper(
    const int16_t (&straight_id_by_mask_req)[8192][4], uint32_t dup13, int req,
    std::vector<int> &legal) {
  int min_len = 2;
  int max_len = 13;
  if (req == 1) {
    min_len = 5;
    max_len = 13;
  } else if (req == 2) {
    min_len = 2;
    max_len = 8;
  } else {
    min_len = 2;
    max_len = 5;
  }
  for (int s = 0; s < 13; ++s) {
    for (int len = min_len; len <= max_len; ++len) {
      if (len == 13 && s != 0)
        continue;
      if (s + len > 26)
        break;
      const uint32_t need = ((1u << len) - 1u) << s;
      if ((dup13 & need) != need)
        continue;
      uint16_t circ_mask = 0;
      for (int k = 0; k < len; ++k)
        circ_mask |= static_cast<uint16_t>(1u << ((s + k) % 13));
      const int16_t mid = straight_id_by_mask_req[circ_mask][req];
      if (mid >= 0)
        legal.push_back(static_cast<int>(mid));
    }
  }
}

struct MoveTables;

static uint64_t pack_hand_kpass(const std::array<int, 13> &hand) {
  uint64_t k = 0;
  for (int i = 0; i < 13; ++i)
    k |= static_cast<uint64_t>(static_cast<unsigned>(hand[i]) & 0xFu)
         << (4 * static_cast<unsigned>(i));
  return k;
}

template <typename F>
static void enumerate_hands_total_cards_rec(int rank, int rem,
                                            std::array<int, 13> &h, F &&fn) {
  if (rank == 12) {
    if (rem >= 0 && rem <= max_cards_in_deck_for_rank(12)) {
      h[12] = rem;
      fn(h);
    }
    return;
  }
  const int mx = max_cards_in_deck_for_rank(rank);
  for (int c = 0; c <= rem && c <= mx; ++c) {
    h[rank] = c;
    enumerate_hands_total_cards_rec(rank + 1, rem - c, h, fn);
  }
}

struct MoveTables {
  // Bomb kicker hand-index for kicker offset k (1..12) at bomb rank r (0..11).
  // Access: bomb_kicker[r * 12 + (k - 1)].  Always a valid index (never equals r).
  int8_t bomb_kicker[144];

  // Per-straight-entry required bitmask of hand indices and required count.
  // Indexed 0..142 (kSTRAIGHT5_START + i).
  uint16_t straight_mask[143];
  int8_t straight_req[143];  // 1 = single straight, 2 = double, 3 = triple

  // Rolling straight helper: `straight_id_by_mask_req[m][r]` is the move id for
  // straight row with rank bitmask `m` and per-rank multiplicity class `r`
  // (1=single, 2=double, 3=triple). Built from MOVE_TO_CARDS; -1 if invalid.
  int16_t straight_id_by_mask_req[8192][4];

  // kPASS straight ids by threshold mask alone: pass_str_off[k][m]..[m+1) in
  // pass_str_ids[k] for m = hs/hp/ht (k = 0/1/2, req = k+1). Built once.
  std::array<std::array<uint32_t, 8193>, 3> pass_str_off{};
  std::array<std::vector<uint16_t>, 3> pass_str_ids{};

  // kPASS legal move ids for hands with 1..KPASS_LEGAL_TABLE_MAX_CARDS_BUILT cards.
  std::vector<uint16_t> kpass_legal_blob;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint16_t>> kpass_legal;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint16_t>> kpass_legal_hi;

  void build_kpass_legal_table();

  MoveTables() {
    // --- Bomb kickers ---
    // From move.cpp: bombRank = r+3, rem = k (offset 1..12).
    //   aux = (rem+2 < bombRank) ? rem+2 : rem+3
    //       = (k < r+1)         ? k+2   : k+3
    //   kicker_hand_idx = aux - 3
    for (int r = 0; r < 12; ++r) {
      for (int k = 1; k <= 12; ++k) {
        const int kv = (k <= r) ? k + 2 : k + 3;
        bomb_kicker[r * 12 + (k - 1)] = static_cast<int8_t>(kv - 3);
      }
    }

    // --- Straights ---
    for (auto &rowm : straight_id_by_mask_req)
      for (int16_t &c : rowm)
        c = -1;
    for (int i = 0; i < 143; ++i) {
      const auto &row = MOVE_TO_CARDS[kSTRAIGHT5_START + i];
      uint16_t mask = 0;
      int req = 0;
      for (int r = 0; r < 13; ++r) {
        if (row[r] > 0) {
          mask |= static_cast<uint16_t>(1u << r);
          if (row[r] > req) req = row[r];
        }
      }
      straight_mask[i] = mask;
      straight_req[i] = static_cast<int8_t>(req);
      int16_t &cell = straight_id_by_mask_req[mask][req];
      const int16_t id = static_cast<int16_t>(kSTRAIGHT5_START + i);
      if (cell != -1 && cell != id) {
        std::abort();
      }
      cell = id;
    }

    std::vector<int> tmp;
    tmp.reserve(64);
    for (int k = 0; k < 3; ++k) {
      const int req = k + 1;
      pass_str_ids[k].clear();
      for (int m = 0; m < 8192; ++m) {
        pass_str_off[k][static_cast<std::size_t>(m)] =
            static_cast<uint32_t>(pass_str_ids[k].size());
        tmp.clear();
        const uint16_t hm = static_cast<uint16_t>(m & 0x1FFFu);
        const uint32_t dup =
            static_cast<uint32_t>(hm) | (static_cast<uint32_t>(hm) << 13u);
        append_pass_straights_for_dup_helper(straight_id_by_mask_req, dup, req,
                                             tmp);
        for (const int id : tmp)
          pass_str_ids[k].push_back(static_cast<uint16_t>(id));
      }
      pass_str_off[k][8192] =
          static_cast<uint32_t>(pass_str_ids[k].size());
    }
    if (KPASS_LEGAL_TABLE_BUILD)
      build_kpass_legal_table();
  }
};

const MoveTables &get_move_tables() {
  static const MoveTables tables;
  return tables;
}

// Enumerate all bomb move IDs that are legal given `hand`, starting at rank
// index `min_rank_idx` (inclusive).  Bare bombs are added first, then kicker
// variants in encoding order.
static void add_bombs_from_hand(const MoveTables &mt,
                                 const std::array<int, 13> &hand,
                                 std::vector<int> &out, int min_rank_idx) {
  const int8_t *kt = mt.bomb_kicker;
  for (int r = min_rank_idx; r < 12; ++r) {
    const int thr = (r == 11) ? 3 : 4;
    if (hand[r] < thr) continue;
    out.push_back(kBOMB_START + r * 13);  // bare bomb
    for (int k = 1; k <= 12; ++k) {
      if (hand[static_cast<int>(kt[r * 12 + (k - 1)])] >= 1)
        out.push_back(kBOMB_START + r * 13 + k);
    }
  }
}

// Enumerate possible bomb move IDs from the opponent's perspective.
// Uses unseen-card counts rather than our own hand counts.
static void add_possible_bombs(const std::array<int, 13> &player_hand,
                                const std::array<int, 13> &discard,
                                int opp_count, std::vector<int> &out,
                                int min_rank_idx) {
  const int8_t *kt = get_move_tables().bomb_kicker;
  for (int r = min_rank_idx; r < 12; ++r) {
    const int thr = (r == 11) ? 3 : 4;
    if (opp_count < thr) continue;
    const int unseen_r =
        max_cards_in_deck_for_rank(r) - player_hand[r] - discard[r];
    if (unseen_r < thr) continue;
    out.push_back(kBOMB_START + r * 13);  // bare bomb
    if (opp_count < thr + 1) continue;
    for (int k = 1; k <= 12; ++k) {
      const int ki = static_cast<int>(kt[r * 12 + (k - 1)]);
      const int unseen_ki =
          max_cards_in_deck_for_rank(ki) - player_hand[ki] - discard[ki];
      if (unseen_ki >= 1)
        out.push_back(kBOMB_START + r * 13 + k);
    }
  }
}

// Full beating rows (all mid2 that beat mid), then packed into flat arrays:
// - combo: same-combination beaters + straights; bombs excluded (see add_bombs).
// - opponent: combo + bare bombs only (kicker bombs omitted for reachability).
struct BeatingFlat {
  std::vector<int> combo_ids;
  std::array<uint32_t, LEGAL_MOVES_SIZE + 1> combo_off{};
  std::vector<int> opponent_ids;
  std::array<uint32_t, LEGAL_MOVES_SIZE + 1> opponent_off{};
};

static const BeatingFlat &beating_flat() {
  static const BeatingFlat T = [] {
    BeatingFlat out;
    std::vector<std::vector<int>> full(static_cast<std::size_t>(LEGAL_MOVES_SIZE));
    for (int mid = kSINGLE_START; mid < LEGAL_MOVES_SIZE; ++mid) {
      Move m(mid);
      for (int mid2 = kSINGLE_START; mid2 < LEGAL_MOVES_SIZE; ++mid2) {
        Move m2(mid2);
        bool beats;
        if (m2.combination == Move::Combination::kBomb)
          beats = (m.combination != Move::Combination::kBomb) ||
                  (m2.rank > m.rank);
        else
          beats = (m2.combination == m.combination) && (m2.rank > m.rank);
        if (beats)
          full[static_cast<std::size_t>(mid)].push_back(mid2);
      }
    }
    for (int mid = 0; mid < LEGAL_MOVES_SIZE; ++mid) {
      out.combo_off[static_cast<std::size_t>(mid)] =
          static_cast<uint32_t>(out.combo_ids.size());
      out.opponent_off[static_cast<std::size_t>(mid)] =
          static_cast<uint32_t>(out.opponent_ids.size());
      for (int id : full[static_cast<std::size_t>(mid)]) {
        if (id < kBOMB_START || id >= kSTRAIGHT5_START)
          out.combo_ids.push_back(id);
        if (!(id >= kBOMB_START && id < kSTRAIGHT5_START &&
              (id - kBOMB_START) % 13 != 0))
          out.opponent_ids.push_back(id);
      }
    }
    out.combo_off[LEGAL_MOVES_SIZE] =
        static_cast<uint32_t>(out.combo_ids.size());
    out.opponent_off[LEGAL_MOVES_SIZE] =
        static_cast<uint32_t>(out.opponent_ids.size());
    return out;
  }();
  return T;
}

BIG2_LEGAL_SEC void append_pass_singles_doubles_triples(uint16_t hs, uint16_t hp,
                                                        uint16_t ht,
                                                        std::vector<int> &legal) {
  for (int r = 0; r < 13; ++r)
    if ((hs >> r) & 1) legal.push_back(kSINGLE_START + r);
  for (int r = 0; r < 12; ++r)
    if ((hp >> r) & 1) legal.push_back(kDOUBLE_START + r);
  for (int r = 0; r < 11; ++r)
    if ((ht >> r) & 1) legal.push_back(kTRIPLE_START + r);
}

// Full houses: each legal full house is a triple rank t (hand[t] >= 3) and a
// distinct pair rank p (hand[p] >= 2); move index i satisfies fh_triple[i]=t,
// fh_pair[i]=p with i = 11*t + (p < t ? p : p - 1).
BIG2_LEGAL_SEC void append_pass_full_houses_sparse(uint16_t ht, uint16_t hp,
                                                   std::vector<int> &legal) {
  // Full-house encoding uses triple hand-index t in 0..11 only (same 12 ranks
  // as the triple move slice); pair hand-index p is also < 12 in the 132 IDs.
  constexpr uint16_t kMask = 0xFFFu;
  ht = static_cast<uint16_t>(ht & kMask);
  hp = static_cast<uint16_t>(hp & kMask);
  for (uint16_t tm = ht; tm;) {
    const int t = __builtin_ctz(static_cast<unsigned>(tm));
    tm = static_cast<uint16_t>(tm & static_cast<uint16_t>(tm - 1));
    for (uint16_t pm = hp; pm;) {
      const int p = __builtin_ctz(static_cast<unsigned>(pm));
      pm = static_cast<uint16_t>(pm & static_cast<uint16_t>(pm - 1));
      if (p == t) continue;
      legal.push_back(kFULL_HOUSE_START + 11 * t + (p < t ? p : p - 1));
    }
  }
}

BIG2_LEGAL_SEC void append_pass_straights_for_dup(const MoveTables &mt,
                                                  uint32_t dup13, int req,
                                                  std::vector<int> &legal) {
  append_pass_straights_for_dup_helper(mt.straight_id_by_mask_req, dup13, req,
                                       legal);
}

BIG2_LEGAL_SEC void append_pass_straights(const MoveTables &mt, uint16_t hs,
                                          uint16_t hp, uint16_t ht,
                                          std::vector<int> &legal) {
  const uint16_t ms = static_cast<uint16_t>(hs & 0x1FFFu);
  const uint16_t mp = static_cast<uint16_t>(hp & 0x1FFFu);
  const uint16_t mt_ = static_cast<uint16_t>(ht & 0x1FFFu);
  for (int k = 0; k < 3; ++k) {
    const uint16_t hm = (k == 0) ? ms : (k == 1) ? mp : mt_;
    const uint32_t lo = mt.pass_str_off[static_cast<std::size_t>(k)][hm];
    const uint32_t hi =
        mt.pass_str_off[static_cast<std::size_t>(k)][hm + 1u];
    const std::vector<uint16_t> &ids =
        mt.pass_str_ids[static_cast<std::size_t>(k)];
    legal.insert(legal.end(), ids.begin() + static_cast<std::ptrdiff_t>(lo),
                 ids.begin() + static_cast<std::ptrdiff_t>(hi));
  }
}

static void legal_moves_fill_pass_lead(const MoveTables &mt,
                                       const std::array<int, 13> &hand,
                                       std::vector<int> &out) {
  uint16_t hs = 0, hp = 0, ht = 0;
  for (int r = 0; r < 13; ++r) {
    const int c = hand[r];
    if (c >= 1) hs |= static_cast<uint16_t>(1u << r);
    if (c >= 2) hp |= static_cast<uint16_t>(1u << r);
    if (c >= 3) ht |= static_cast<uint16_t>(1u << r);
  }
  append_pass_singles_doubles_triples(hs, hp, ht, out);
  append_pass_full_houses_sparse(ht, hp, out);
  add_bombs_from_hand(mt, hand, out, 0);
  append_pass_straights(mt, hs, hp, ht, out);
}

void MoveTables::build_kpass_legal_table() {
  kpass_legal_blob.clear();
  kpass_legal.clear();
  kpass_legal_hi.clear();
  kpass_legal.reserve(68000);
  if (KPASS_LEGAL_TABLE_MAX_CARDS_BUILT > KPASS_LEGAL_PRIMARY_MAX_CARDS)
    kpass_legal_hi.reserve(330000);
  std::array<int, 13> h{};
  std::vector<int> tmp;
  tmp.reserve(static_cast<std::size_t>(LEGAL_MOVES_SIZE));
  const auto append_row = [&](const std::array<int, 13> &hh,
                               std::unordered_map<uint64_t,
                                                  std::pair<uint32_t, uint16_t>>
                                   &into_map) {
    tmp.clear();
    legal_moves_fill_pass_lead(*this, hh, tmp);
    const uint32_t off = static_cast<uint32_t>(kpass_legal_blob.size());
    for (const int id : tmp)
      kpass_legal_blob.push_back(static_cast<uint16_t>(id));
    const auto len = static_cast<uint16_t>(kpass_legal_blob.size() - off);
    const uint64_t key = pack_hand_kpass(hh);
    if (!into_map.emplace(key, std::make_pair(off, len)).second)
      std::abort();
  };
  const int primary_u = std::min(KPASS_LEGAL_PRIMARY_MAX_CARDS,
                                 KPASS_LEGAL_TABLE_MAX_CARDS_BUILT);
  for (int X = 1; X <= primary_u; ++X) {
    enumerate_hands_total_cards_rec(
        0, X, h, [&](const std::array<int, 13> &hh) {
          append_row(hh, kpass_legal);
        });
  }
  for (int X = primary_u + 1; X <= KPASS_LEGAL_TABLE_MAX_CARDS_BUILT; ++X) {
    enumerate_hands_total_cards_rec(
        0, X, h, [&](const std::array<int, 13> &hh) {
          append_row(hh, kpass_legal_hi);
        });
  }
}

BIG2_LEGAL_SEC void append_response_beats_and_pass(const std::array<int, 13> &hand,
                                                   int last_move_id,
                                                   std::vector<int> &legal) {
  legal.push_back(kPASS);
  const auto &Bf = beating_flat();
  const uint32_t clo = Bf.combo_off[static_cast<std::size_t>(last_move_id)];
  const uint32_t chi =
      Bf.combo_off[static_cast<std::size_t>(last_move_id) + 1u];
  for (uint32_t i = clo; i < chi; ++i) {
    const int beat_id = Bf.combo_ids[i];
    bool ok = true;
    for (int r = 0; r < 13; ++r) {
      if (hand[r] < MOVE_TO_CARDS[beat_id][r]) {
        ok = false;
        break;
      }
    }
    if (ok) legal.push_back(beat_id);
  }
}

static bool try_kpass_legal_copy(const MoveTables &mt,
                                 const std::array<int, 13> &hand,
                                 std::vector<int> &out) {
  if (g_kpass_legal_max_cards <= 0)
    return false;
  int tot = 0;
  for (int c : hand)
    tot += c;
  if (tot == 0 || tot > g_kpass_legal_max_cards ||
      tot > KPASS_LEGAL_TABLE_MAX_CARDS_BUILT)
    return false;
  const uint64_t key = pack_hand_kpass(hand);
  const auto &tab = (tot <= KPASS_LEGAL_PRIMARY_MAX_CARDS) ? mt.kpass_legal
                                                          : mt.kpass_legal_hi;
  const auto it = tab.find(key);
  if (it == tab.end())
    return false;
  const uint32_t lo = it->second.first;
  const uint16_t len = it->second.second;
  out.clear();
  if (out.capacity() < static_cast<std::size_t>(LEGAL_MOVES_SIZE))
    out.reserve(static_cast<std::size_t>(LEGAL_MOVES_SIZE));
  out.resize(len);
  for (uint16_t i = 0; i < len; ++i)
    out[i] = static_cast<int>(mt.kpass_legal_blob[lo + i]);
  return true;
}

void legal_moves_fill(const std::array<int, 13> &hand, int last_move_id,
                      std::vector<int> &out) {
  const MoveTables &mt = get_move_tables();
  if (last_move_id == kPASS && try_kpass_legal_copy(mt, hand, out))
    return;

  out.clear();
  // Every legal id is in [0, LEGAL_MOVES_SIZE); reuse capacity across calls
  // (hot PIMC path) to avoid realloc + memmoves while appending.
  if (out.capacity() < static_cast<std::size_t>(LEGAL_MOVES_SIZE))
    out.reserve(static_cast<std::size_t>(LEGAL_MOVES_SIZE));

  if (last_move_id == kPASS)
    legal_moves_fill_pass_lead(mt, hand, out);
  else {
    append_response_beats_and_pass(hand, last_move_id, out);
    if (last_move_id >= kBOMB_START && last_move_id < kSTRAIGHT5_START)
      add_bombs_from_hand(mt, hand, out,
                          (last_move_id - kBOMB_START) / 13 + 1);
    else
      add_bombs_from_hand(mt, hand, out, 0);
  }
}

} // namespace

void straight_moves_for_pass_masks_into(uint16_t hs, uint16_t hp, uint16_t ht,
                                        std::vector<int> &out) {
  out.clear();
  append_pass_straights(get_move_tables(), hs, hp, ht, out);
}

// ======================================================================
// compute_legal_moves
// ======================================================================
void compute_legal_moves_into(const std::array<int, 13> &hand, int last_move_id,
                              std::vector<int> &out) {
  legal_moves_fill(hand, last_move_id, out);
}

std::vector<int> compute_legal_moves(const std::array<int, 13> &hand,
                                     int last_move_id) {
  std::vector<int> v;
  compute_legal_moves_into(hand, last_move_id, v);
  return v;
}

// ======================================================================
// opponent_can_respond
// ======================================================================
bool opponent_can_respond(int move_id, const std::array<int, 13> &hand,
                          const std::array<int, 13> &discard, int opp_count) {
  int unseen[13];
  for (int r = 0; r < 13; ++r)
    unseen[r] =
        max_cards_in_deck_for_rank(r) - hand[r] - discard[r];
  const auto &Bf = beating_flat();
  const uint32_t olo = Bf.opponent_off[static_cast<std::size_t>(move_id)];
  const uint32_t ohi =
      Bf.opponent_off[static_cast<std::size_t>(move_id) + 1u];
  for (uint32_t i = olo; i < ohi; ++i) {
    const int mid2 = Bf.opponent_ids[i];
    const auto &cost = MOVE_TO_CARDS[mid2];
    if (opp_count < cost[13]) continue;
    bool feasible = true;
    for (int r = 0; r < 13; ++r) {
      if (unseen[r] < cost[r]) {
        feasible = false;
        break;
      }
    }
    if (feasible) return true;
  }
  return false;
}

// ======================================================================
// find_forced_win  (with transposition table + O(1) push_back)
// ======================================================================
namespace {

struct HDKey {
  std::array<uint8_t, 26> data;
  bool operator==(const HDKey &o) const noexcept { return data == o.data; }
};
struct HDKeyHash {
  std::size_t operator()(const HDKey &k) const noexcept {
    std::size_t h = 14695981039346656037ull;
    for (uint8_t c : k.data) {
      h ^= c;
      h *= 1099511628211ull;
    }
    return h;
  }
};
using FWTable =
    std::unordered_map<HDKey, std::optional<std::vector<int>>, HDKeyHash>;

inline HDKey make_fw_key(const std::array<int, 13> &hand,
                         const std::array<int, 13> &discard) {
  HDKey k;
  for (int i = 0; i < 13; ++i)
    k.data[i] = static_cast<uint8_t>(hand[i]);
  for (int i = 0; i < 13; ++i)
    k.data[13 + i] = static_cast<uint8_t>(discard[i]);
  return k;
}

// Builds the winning sequence in reverse (push_back); the public wrapper
// reverses once after the top-level call returns.
static std::optional<std::vector<int>>
find_forced_win_impl(const std::array<int, 13> &hand,
                     const std::array<int, 13> &discard, int opp_count,
                     FWTable &table) {
  const HDKey key = make_fw_key(hand, discard);
  auto it = table.find(key);
  if (it != table.end()) return it->second;

  int hand_size = 0;
  for (int c : hand) hand_size += c;

  std::vector<int> legal;
  legal.reserve(64);
  compute_legal_moves_into(hand, kPASS, legal);
  std::sort(legal.begin(), legal.end(), [](int a, int b) {
    const int ca = MOVE_TO_CARDS[a][13], cb = MOVE_TO_CARDS[b][13];
    return (ca != cb) ? (ca > cb) : (a < b);
  });

  for (int mid : legal) {
    const int cards = MOVE_TO_CARDS[mid][13];

    if (cards == hand_size) {
      auto result = std::make_optional(std::vector<int>{mid});
      table[key] = result;
      return result;
    }

    if (!opponent_can_respond(mid, hand, discard, opp_count)) {
      const auto &cost = MOVE_TO_CARDS[mid];
      std::array<int, 13> new_hand = hand;
      std::array<int, 13> new_discard = discard;
      for (int r = 0; r < 13; ++r) {
        new_hand[r] -= cost[r];
        new_discard[r] += cost[r];
      }
      auto rest = find_forced_win_impl(new_hand, new_discard, opp_count, table);
      if (rest) {
        rest->push_back(mid);
        table[key] = rest;
        return rest;
      }
    }
  }

  table[key] = std::nullopt;
  return std::nullopt;
}

} // namespace

std::optional<std::vector<int>>
find_forced_win(const std::array<int, 13> &hand,
                const std::array<int, 13> &discard, int opp_count) {
  FWTable table;
  auto result = find_forced_win_impl(hand, discard, opp_count, table);
  if (result) std::reverse(result->begin(), result->end());
  return result;
}

// ======================================================================
// compute_possible_moves
// ======================================================================
std::vector<int> compute_possible_moves(const std::array<int, 13> &player_hand,
                                        const std::array<int, 13> &discard_pile,
                                        int opponent_card_count,
                                        int last_move_id,
                                        bool exclude_bombs) {
  std::vector<int> possible;
  possible.reserve(32);

  auto check_move = [&](int move_id) {
    bool ok = opponent_card_count >= MOVE_TO_CARDS[move_id][13];
    if (ok) {
      for (int r = 0; r < 13; ++r) {
        const int unseen = max_cards_in_deck_for_rank(r) - player_hand[r] -
                           discard_pile[r];
        if (unseen < MOVE_TO_CARDS[move_id][r]) { ok = false; break; }
      }
    }
    if (ok) possible.push_back(move_id);
  };

  if (last_move_id != kPASS) {
    possible.push_back(kPASS);
    const auto &Bf = beating_flat();
    const uint32_t clo = Bf.combo_off[static_cast<std::size_t>(last_move_id)];
    const uint32_t chi =
        Bf.combo_off[static_cast<std::size_t>(last_move_id) + 1u];
    for (uint32_t i = clo; i < chi; ++i)
      check_move(Bf.combo_ids[i]);
    if (!exclude_bombs) {
      if (last_move_id >= kBOMB_START && last_move_id < kSTRAIGHT5_START)
        add_possible_bombs(player_hand, discard_pile, opponent_card_count,
                           possible,
                           (last_move_id - kBOMB_START) / 13 + 1);
      else
        add_possible_bombs(player_hand, discard_pile, opponent_card_count,
                           possible, 0);
    }
  } else {
    // PASS: iterate all non-bomb non-pass move IDs, then handle bombs separately
    for (int mid = kSINGLE_START; mid < kBOMB_START; ++mid)
      check_move(mid);
    if (!exclude_bombs)
      add_possible_bombs(player_hand, discard_pile, opponent_card_count,
                         possible, 0);
    for (int mid = kSTRAIGHT5_START; mid < LEGAL_MOVES_SIZE; ++mid)
      check_move(mid);
  }

  return possible;
}
