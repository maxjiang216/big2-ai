#include "game.h"
#include "partial_game.h"
#include "util.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// Dense 132-index full-house enumeration (reference for sparse equivalence).
static std::vector<int> full_houses_dense(uint16_t ht, uint16_t hp) {
  constexpr uint16_t kMask = 0xFFFu;
  ht = static_cast<uint16_t>(ht & kMask);
  hp = static_cast<uint16_t>(hp & kMask);
  std::vector<int> v;
  for (int i = 0; i < 132; ++i) {
    const int t = i / 11;
    const int rem = i % 11;
    const int p = (rem < t) ? rem : rem + 1;
    if (((ht >> t) & 1) && ((hp >> p) & 1))
      v.push_back(kFULL_HOUSE_START + i);
  }
  return v;
}

static std::vector<int> full_houses_sparse(uint16_t ht, uint16_t hp) {
  constexpr uint16_t kMask = 0xFFFu;
  ht = static_cast<uint16_t>(ht & kMask);
  hp = static_cast<uint16_t>(hp & kMask);
  std::vector<int> v;
  for (uint16_t tm = ht; tm;) {
    const int t = __builtin_ctz(static_cast<unsigned>(tm));
    tm = static_cast<uint16_t>(tm & static_cast<uint16_t>(tm - 1));
    for (uint16_t pm = hp; pm;) {
      const int p = __builtin_ctz(static_cast<unsigned>(pm));
      pm = static_cast<uint16_t>(pm & static_cast<uint16_t>(pm - 1));
      if (p == t)
        continue;
      v.push_back(kFULL_HOUSE_START + 11 * t + (p < t ? p : p - 1));
    }
  }
  return v;
}

static void random_position(std::mt19937 &rng, std::array<int, 13> *hand,
                            int *last_move_id, int *current_player) {
  Game g;
  g.shuffle_deal(rng);
  const int n_step = static_cast<int>(rng() % 20u);
  for (int s = 0; s < n_step && !g.is_over(); ++s) {
    std::vector<int> legal = g.get_legal_moves();
    if (legal.empty())
      break;
    g.apply_move(legal[rng() % legal.size()]);
  }
  *current_player = g.current_player();
  *hand = g.player_hand(*current_player);
  *last_move_id = g.last_move_id();
}

static void pass_straights_bruteforce(uint16_t hs, uint16_t hp, uint16_t ht,
                                      std::vector<int> &out) {
  out.clear();
  for (int i = 0; i < 143; ++i) {
    const auto &row = MOVE_TO_CARDS[kSTRAIGHT5_START + i];
    uint16_t mask = 0;
    int req = 0;
    for (int r = 0; r < 13; ++r) {
      if (row[r] > 0) {
        mask |= static_cast<uint16_t>(1u << r);
        if (row[r] > req)
          req = row[r];
      }
    }
    const uint16_t hm =
        (req == 1) ? hs : (req == 2) ? hp : ht;
    if ((hm & mask) == mask)
      out.push_back(kSTRAIGHT5_START + i);
  }
}

static void enumerate_hands_kpass_test(int rank, int rem, std::array<int, 13> &h,
                                       const std::function<void(
                                           const std::array<int, 13> &)> &fn) {
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
    enumerate_hands_kpass_test(rank + 1, rem - c, h, fn);
  }
}

// Move id of the bare (auxiliary-free) bomb of a given face rank, or -1.
static int bare_bomb_id(int face_rank) {
  const auto &mvs = all_moves();
  for (int i = 0; i < LEGAL_MOVES_SIZE; ++i)
    if (mvs[i].combination == Move::Combination::kBomb &&
        mvs[i].auxiliary == 0 && mvs[i].rank == face_rank)
      return i;
  return -1;
}

static bool affordable(const std::array<int, 13> &hand, int mid) {
  const auto &cost = MOVE_TO_CARDS[mid];
  for (int r = 0; r < 13; ++r)
    if (hand[r] < cost[r])
      return false;
  return true;
}

void run_legal_moves_tests() {
  std::mt19937 rng(42);

  // ---- Bomb responses beat lower bombs (regression: util.cpp compared a face
  // rank against a rank *index*, so bombs of Q and K read as unbeatable and
  // every bomb wrongly blocked the three ranks above it). ----
  {
    // Bare bombs of 5, 9, K and the ace bomb (three aces; one ace is removed
    // from the deck, so rank 14 is a triple -- see RULES.md:90).
    std::array<int, 13> hand{};
    hand[2] = 4;   // 5s
    hand[6] = 4;   // 9s
    hand[10] = 4;  // Ks
    hand[11] = 3;  // ace bomb

    struct Case {
      int last_rank;               // face rank of the bomb just played
      std::vector<int> expect;     // face ranks of our legal bomb responses
    };
    // There is no bomb of 2s: RULES.md:9 removes three of the four 2s.
    const std::vector<Case> cases = {
        {4, {5, 9, 13, 14}},  // low bomb: everything higher answers it
        {5, {9, 13, 14}},     // equal rank does not beat
        {12, {13, 14}},       // the Q case that read as unbeatable
        {13, {14}},           // the K case that read as unbeatable
        {14, {}},             // ace bomb genuinely is unbeatable
    };

    for (const Case &c : cases) {
      const int last_id = bare_bomb_id(c.last_rank);
      if (last_id < 0) {
        std::cerr << "no bare bomb for rank " << c.last_rank << "\n";
        std::abort();
      }
      std::vector<int> got;
      for (int mid : compute_legal_moves(hand, last_id)) {
        const Move &m = all_moves()[mid];
        if (m.combination == Move::Combination::kBomb && m.auxiliary == 0)
          got.push_back(m.rank);
      }
      std::sort(got.begin(), got.end());
      std::vector<int> want = c.expect;
      std::sort(want.begin(), want.end());
      if (got != want) {
        std::cerr << "bomb response mismatch after bomb of rank "
                  << c.last_rank << ": got {";
        for (int r : got)
          std::cerr << r << ",";
        std::cerr << "} want {";
        for (int r : want)
          std::cerr << r << ",";
        std::cerr << "}\n";
        std::abort();
      }
    }
  }

  // ---- compute_legal_moves agrees with the get_beating_moves relation.
  // These are two independent encodings of "what beats what" (the browser
  // consumes the latter via az_moves.json, the search the former); nothing
  // else cross-checks them. ----
  for (int it = 0; it < 20'000; ++it) {
    std::array<int, 13> hand{};
    int lid = 0;
    int cp = 0;
    random_position(rng, &hand, &lid, &cp);
    (void)cp;
    if (lid == kPASS)
      continue;  // lead positions are not a beat relation

    std::vector<int> got;
    for (int mid : compute_legal_moves(hand, lid))
      if (mid != kPASS)
        got.push_back(mid);
    std::vector<int> want;
    for (int mid : get_beating_moves()[lid])
      if (affordable(hand, mid))
        want.push_back(mid);
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got != want) {
      std::cerr << "compute_legal_moves vs get_beating_moves mismatch, last="
                << lid << " (" << all_moves()[lid] << ")\n";
      std::vector<int> diff;
      std::set_symmetric_difference(got.begin(), got.end(), want.begin(),
                                    want.end(), std::back_inserter(diff));
      for (int mid : diff)
        std::cerr << "  differs: " << mid << " (" << all_moves()[mid] << ")\n";
      std::abort();
    }
  }

  // kPASS straight enumeration: circular window path vs 143-loop reference
  for (int it = 0; it < 50'000; ++it) {
    const uint16_t hs = static_cast<uint16_t>(rng() & 0x1FFFu);
    const uint16_t hp = static_cast<uint16_t>(rng() & 0x1FFFu);
    const uint16_t ht = static_cast<uint16_t>(rng() & 0x1FFFu);
    std::vector<int> a, b;
    pass_straights_bruteforce(hs, hp, ht, a);
    straight_moves_for_pass_masks_into(hs, hp, ht, b);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) {
      std::cerr << "pass straights circular vs bruteforce mismatch\n";
      std::abort();
    }
  }
  for (unsigned m = 0; m < 8192u; ++m) {
    const uint16_t hs = static_cast<uint16_t>(m & 0x1FFFu);
    std::vector<int> a, b;
    pass_straights_bruteforce(hs, 0, 0, a);
    straight_moves_for_pass_masks_into(hs, 0, 0, b);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) {
      std::cerr << "pass straights exhaustive hs mismatch\n";
      std::abort();
    }
  }

  // kPASS lookup table vs generic path for all deck-capped hands with
  // 1..KPASS_LEGAL_TABLE_MAX_CARDS_BUILT cards
  {
    const int saved_cap = g_kpass_legal_max_cards;
    std::array<int, 13> h{};
    for (int X = 1; X <= KPASS_LEGAL_TABLE_MAX_CARDS_BUILT; ++X) {
      enumerate_hands_kpass_test(0, X, h, [&](const std::array<int, 13> &hh) {
        g_kpass_legal_max_cards = 0;
        std::vector<int> ref = compute_legal_moves(hh, kPASS);
        std::sort(ref.begin(), ref.end());
        for (int cap = 0; cap <= KPASS_LEGAL_TABLE_MAX_CARDS_BUILT; ++cap) {
          g_kpass_legal_max_cards = cap;
          std::vector<int> got = compute_legal_moves(hh, kPASS);
          std::sort(got.begin(), got.end());
          if (got != ref) {
            std::cerr << "kPASS table mismatch cap=" << cap << "\n";
            std::abort();
          }
        }
      });
    }
    g_kpass_legal_max_cards = saved_cap;
  }

  // Random bitmasks: full house sparse vs dense
  for (int it = 0; it < 30'000; ++it) {
    uint16_t ht = static_cast<uint16_t>(rng() & 0x1FFF);
    uint16_t hp = static_cast<uint16_t>(rng() & 0x1FFF);
    auto a = full_houses_dense(ht, hp);
    auto b = full_houses_sparse(ht, hp);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) {
      std::cerr << "full house sparse/dense mismatch\n";
      std::abort();
    }
  }

  // Full compute_legal_moves: into vs return-by-value, random table positions
  for (int it = 0; it < 8'000; ++it) {
    std::array<int, 13> hand{};
    int lid = 0;
    int cp = 0;
    random_position(rng, &hand, &lid, &cp);
    (void)cp;

    std::vector<int> v1 = compute_legal_moves(hand, lid);
    std::vector<int> v2;
    compute_legal_moves_into(hand, lid, v2);
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    if (v1 != v2) {
      std::cerr << "compute_legal_moves vs into mismatch\n";
      std::abort();
    }
  }

  // PartialGame::get_legal_moves_into matches get_legal_moves
  for (int it = 0; it < 2'000; ++it) {
    Game g;
    g.shuffle_deal(rng);
    for (int s = 0; s < static_cast<int>(rng() % 12u) && !g.is_over(); ++s) {
      auto legal = g.get_legal_moves();
      if (legal.empty())
        break;
      g.apply_move(legal[rng() % legal.size()]);
    }
    int view_p = static_cast<int>(rng() % 2u);
    PartialGame pg(g, view_p);
    std::vector<int> a = pg.get_legal_moves();
    std::vector<int> b;
    pg.get_legal_moves_into(b);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) {
      std::cerr << "PartialGame get_legal_moves_into mismatch\n";
      std::abort();
    }
  }
}
