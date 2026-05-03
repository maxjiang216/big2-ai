#include "util.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>

// ---------------------------------------------------------------------------
// Precomputed tables for HandBits-based compute_legal_moves
// ---------------------------------------------------------------------------

// Interleaved rank+amount for bomb kicker checks.
struct SparseReq {
    int8_t count{};
    struct { int8_t rank, amount; } pairs[13];
};

// Check kicker requirements against HandBits.
static inline bool can_play_bits(const HandBits &h, const SparseReq &req) {
    for (int i = 0; i < req.count; ++i) {
        const int r = req.pairs[i].rank;
        const int n = req.pairs[i].amount;
        const uint16_t bit = static_cast<uint16_t>(1u << r);
        bool ok;
        if      (n >= 4) ok = (h.at4 & bit) != 0;
        else if (n >= 3) ok = (h.at3 & bit) != 0;
        else if (n >= 2) ok = (h.at2 & bit) != 0;
        else             ok = (h.at1 & bit) != 0;
        if (!ok) return false;
    }
    return true;
}

// Map Move::Combination enum value to straight level:
// 0 = single straights, 1 = double straights, 2 = triple straights, -1 = not a straight.
static inline int combo_to_straight_level(int combo) {
    if (combo >= 22) return 2;  // kTripleStraight2..kTripleStraight5 (22-25)
    if (combo >= 15) return 1;  // kDoubleStraight2..kDoubleStraight8 (15-21)
    if (combo >= 6)  return 0;  // kStraight5..kStraight13 (6-14)
    return -1;
}

struct HandBitsTables {
    // straight[level][at_mask] = sorted list of playable straight move IDs.
    // Level 0: single straights (at1), 1: double (at2), 2: triple (at3).
    std::vector<std::vector<uint16_t>> straight[3];

    // fh[t][k] = full house move ID with triple rank t and pair rank k (-1 if none).
    int16_t fh[13][13];

    // Bomb groups by base rank.
    struct BombGroup {
        int bare_id{-1};   // bare bomb (no kicker) move ID; -1 = rank can't make bombs
        int base_count{0}; // 4 for standard ranks, 3 for rank 11 if applicable
        struct BombAux { int move_id; SparseReq req; }; // kicker-only requirement
        std::vector<BombAux> aux;
    };
    BombGroup bombs[13];
};

static const HandBitsTables &hand_bits_tables() {
    static const HandBitsTables t = []() {
        HandBitsTables ht;

        // ---- Initialise ----
        for (int lv = 0; lv < 3; ++lv)
            ht.straight[lv].assign(1 << 13, std::vector<uint16_t>{});
        for (int ti = 0; ti < 13; ++ti)
            for (int ki = 0; ki < 13; ++ki)
                ht.fh[ti][ki] = -1;

        // ---- Straight tables ----
        // For each straight move, add its ID to every at-mask that is a
        // superset of its required rank bits.
        for (int mid = kSTRAIGHT5_START; mid < LEGAL_MOVES_SIZE; ++mid) {
            const auto &cost = MOVE_TO_CARDS[mid];
            uint16_t needed = 0;
            int cost_per_rank = 0;
            for (int r = 0; r < 13; ++r) {
                if (cost[r] > 0) {
                    needed |= static_cast<uint16_t>(1u << r);
                    cost_per_rank = cost[r];
                }
            }
            const int level = cost_per_rank - 1; // 0/1/2
            // Iterate all supersets of `needed` within 13 bits.
            const uint16_t complement = static_cast<uint16_t>((~needed) & 0x1FFFu);
            for (uint16_t sup = complement; ; sup = static_cast<uint16_t>((sup - 1) & complement)) {
                ht.straight[level][needed | sup].push_back(static_cast<uint16_t>(mid));
                if (sup == 0) break;
            }
        }

        // ---- Full house table ----
        for (int mid = kFULL_HOUSE_START; mid < kBOMB_START; ++mid) {
            const auto &cost = MOVE_TO_CARDS[mid];
            int t_rank = -1, k_rank = -1;
            for (int r = 0; r < 13; ++r) {
                if (cost[r] == 3) t_rank = r;
                else if (cost[r] == 2) k_rank = r;
            }
            assert(t_rank >= 0 && k_rank >= 0 && t_rank != k_rank);
            ht.fh[t_rank][k_rank] = static_cast<int16_t>(mid);
        }

        // ---- Bomb groups ----
        for (int mid = kBOMB_START; mid < kSTRAIGHT5_START; ++mid) {
            const auto &cost = MOVE_TO_CARDS[mid];
            // Base rank = rank with the most cards in this move.
            int base_rank = -1, base_count = 0;
            for (int r = 0; r < 13; ++r) {
                if (cost[r] > base_count) { base_count = cost[r]; base_rank = r; }
            }
            if (base_rank < 0 || base_count < 3) continue;

            const bool is_bare = (cost[13] == base_count); // no kicker
            auto &bg = ht.bombs[base_rank];
            if (is_bare) {
                bg.bare_id    = mid;
                bg.base_count = base_count;
            } else {
                SparseReq req{};
                for (int r = 0; r < 13; ++r) {
                    if (cost[r] > 0 && r != base_rank)
                        req.pairs[req.count++] = {static_cast<int8_t>(r),
                                                  static_cast<int8_t>(cost[r])};
                }
                bg.aux.push_back({mid, req});
            }
        }
        return ht;
    }();
    return t;
}

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

std::vector<int> compute_legal_moves(const HandBits &hand,
                                     const Move &last_move) {
  const auto &tbl = hand_bits_tables();
  std::vector<int> legal;
  legal.reserve(32);

  if (last_move.combination == Move::Combination::kPass) {
    // ---- Lead position ----
    // Singles: one card of any rank
    for (uint16_t b = hand.at1; b; b &= static_cast<uint16_t>(b - 1))
        legal.push_back(kSINGLE_START + __builtin_ctz(b));
    // Pairs: two of any rank (rank 12 has max=1 so at2 bit 12 is always 0)
    for (uint16_t b = hand.at2; b; b &= static_cast<uint16_t>(b - 1))
        legal.push_back(kDOUBLE_START + __builtin_ctz(b));
    // Standalone triples: only ranks 0-10 have triple play IDs
    for (uint16_t b = static_cast<uint16_t>(hand.at3 & 0x07FFu); b;
         b &= static_cast<uint16_t>(b - 1))
        legal.push_back(kTRIPLE_START + __builtin_ctz(b));
    // Full houses: triple rank (0-11) × pair rank (0-11, ≠ triple rank)
    for (uint16_t tb = hand.at3; tb; tb &= static_cast<uint16_t>(tb - 1)) {
        const int t = __builtin_ctz(tb);
        for (uint16_t kb = static_cast<uint16_t>(hand.at2 & ~(1u << t)); kb;
             kb &= static_cast<uint16_t>(kb - 1)) {
            const int16_t fh = tbl.fh[t][__builtin_ctz(kb)];
            if (fh >= 0) legal.push_back(fh);
        }
    }
    // Bombs: grouped by base rank
    for (int b = 0; b < 13; ++b) {
        const auto &bg = tbl.bombs[b];
        if (bg.bare_id < 0) continue;
        const uint16_t base_bit = static_cast<uint16_t>(1u << b);
        const bool avail = (bg.base_count >= 4) ? ((hand.at4 & base_bit) != 0)
                                                 : ((hand.at3 & base_bit) != 0);
        if (!avail) continue;
        legal.push_back(bg.bare_id);
        for (const auto &a : bg.aux)
            if (can_play_bits(hand, a.req)) legal.push_back(a.move_id);
    }
    // Straights: O(1) table lookup per type
    const uint16_t a1 = static_cast<uint16_t>(hand.at1 & 0x1FFFu);
    const uint16_t a2 = static_cast<uint16_t>(hand.at2 & 0x1FFFu);
    const uint16_t a3 = static_cast<uint16_t>(hand.at3 & 0x1FFFu);
    for (uint16_t mid : tbl.straight[0][a1]) legal.push_back(mid);
    for (uint16_t mid : tbl.straight[1][a2]) legal.push_back(mid);
    for (uint16_t mid : tbl.straight[2][a3]) legal.push_back(mid);
    return legal;
  }

  // ---- Response position ----
  legal.push_back(kPASS);

  const int lc = static_cast<int>(last_move.combination);
  const int lr = last_move.rank;
  const int bc = static_cast<int>(Move::Combination::kBomb);

  if (lc == static_cast<int>(Move::Combination::kSingle)) {
      for (uint16_t b = static_cast<uint16_t>(hand.at1 >> (lr + 1)); b;
           b &= static_cast<uint16_t>(b - 1))
          legal.push_back(kSINGLE_START + lr + 1 + __builtin_ctz(b));
  } else if (lc == static_cast<int>(Move::Combination::kDouble)) {
      for (uint16_t b = static_cast<uint16_t>(hand.at2 >> (lr + 1)); b;
           b &= static_cast<uint16_t>(b - 1))
          legal.push_back(kDOUBLE_START + lr + 1 + __builtin_ctz(b));
  } else if (lc == static_cast<int>(Move::Combination::kTriple)) {
      for (uint16_t b = static_cast<uint16_t>((hand.at3 & 0x07FFu) >> (lr + 1)); b;
           b &= static_cast<uint16_t>(b - 1))
          legal.push_back(kTRIPLE_START + lr + 1 + __builtin_ctz(b));
  } else if (lc == static_cast<int>(Move::Combination::kFullHouse)) {
      for (uint16_t tb = static_cast<uint16_t>(hand.at3 >> (lr + 1)); tb;
           tb &= static_cast<uint16_t>(tb - 1)) {
          const int t = lr + 1 + __builtin_ctz(tb);
          for (uint16_t kb = static_cast<uint16_t>(hand.at2 & ~(1u << t)); kb;
               kb &= static_cast<uint16_t>(kb - 1)) {
              const int16_t fh = tbl.fh[t][__builtin_ctz(kb)];
              if (fh >= 0) legal.push_back(fh);
          }
      }
  } else if (lc != bc) {
      // Straight response: table lookup + filter by rank > lr
      const int level = combo_to_straight_level(lc);
      if (level >= 0) {
          const uint16_t at = (level == 0) ? hand.at1 :
                              (level == 1) ? hand.at2 : hand.at3;
          const auto &mvs = all_moves();
          for (uint16_t mid : tbl.straight[level][at & 0x1FFFu])
              if (mvs[mid].rank > lr) legal.push_back(mid);
      }
  }

  // Bombs: any bomb beats a non-bomb; higher-rank bomb beats lower-rank bomb
  const int bomb_rank_min = (lc == bc) ? lr : -1;
  for (int b = 0; b < 13; ++b) {
      if (b <= bomb_rank_min) continue;
      const auto &bg = tbl.bombs[b];
      if (bg.bare_id < 0) continue;
      const uint16_t base_bit = static_cast<uint16_t>(1u << b);
      const bool avail = (bg.base_count >= 4) ? ((hand.at4 & base_bit) != 0)
                                               : ((hand.at3 & base_bit) != 0);
      if (!avail) continue;
      legal.push_back(bg.bare_id);
      for (const auto &a : bg.aux)
          if (can_play_bits(hand, a.req)) legal.push_back(a.move_id);
  }

  return legal;
}

std::vector<int> compute_legal_moves(const std::array<int, 13> &hand,
                                     const Move &last_move) {
    return compute_legal_moves(hand_bits_from_counts(hand), last_move);
}

std::vector<int> compute_legal_moves(const std::array<int, 13> &hand, int move_id) {
    return compute_legal_moves(hand_bits_from_counts(hand), Move(move_id));
}

void compute_legal_moves_into(const std::array<int, 13> &hand, int move_id,
                              std::vector<int> &out) {
    out = compute_legal_moves(hand, move_id);
}

void straight_moves_for_pass_masks_into(uint16_t hs, uint16_t hp, uint16_t ht,
                                        std::vector<int> &out) {
    const auto &tbl = hand_bits_tables();
    out.clear();
    for (uint16_t mid : tbl.straight[0][hs & 0x1FFFu]) out.push_back(mid);
    for (uint16_t mid : tbl.straight[1][hp & 0x1FFFu]) out.push_back(mid);
    for (uint16_t mid : tbl.straight[2][ht & 0x1FFFu]) out.push_back(mid);
}

int g_kpass_legal_max_cards = 0;

const std::vector<std::vector<int>> &get_beating_moves() {
  static const std::vector<std::vector<int>> table = [] {
    const auto &moves = all_moves();
    std::vector<std::vector<int>> t(LEGAL_MOVES_SIZE);
    for (int mid = kSINGLE_START; mid < LEGAL_MOVES_SIZE; ++mid) {
      const Move &m = moves[mid];
      for (int mid2 = kSINGLE_START; mid2 < LEGAL_MOVES_SIZE; ++mid2) {
        const Move &m2 = moves[mid2];
        bool beats;
        if (m2.combination == Move::Combination::kBomb)
          beats = (m.combination != Move::Combination::kBomb) ||
                  (m2.rank > m.rank);
        else
          beats = (m2.combination == m.combination) && (m2.rank > m.rank);
        if (beats)
          t[mid].push_back(mid2);
      }
    }
    return t;
  }();
  return table;
}

bool opponent_can_respond(int move_id,
                          const std::array<int, 13> &hand,
                          const std::array<int, 13> &discard,
                          int opp_count) {
  const auto &moves = all_moves();
  for (int mid2 : get_beating_moves()[move_id]) {
    const Move &m2 = moves[mid2];
    // Bombs with an auxiliary card beat the same things as the bare (no-kicker)
    // bomb of the same rank but use more cards. For "can the opponent
    // possibly respond?" it is enough to check bare bombs: if they cannot play
    // the minimal-card beating bomb, they cannot play the aux version either.
    if (m2.combination == Move::Combination::kBomb && m2.auxiliary != 0)
      continue;
    const auto &cost = MOVE_TO_CARDS[mid2];
    if (opp_count < cost[13])
      continue;
    bool feasible = true;
    for (int r = 0; r < 13; ++r) {
      int unseen = max_cards_in_deck_for_rank(r) - hand[r] - discard[r];
      if (unseen < cost[r]) {
        feasible = false;
        break;
      }
    }
    if (feasible)
      return true;
  }
  return false;
}

std::optional<std::vector<int>>
find_forced_win(const std::array<int, 13> &hand,
                const std::array<int, 13> &discard,
                int opp_count) {
  int hand_size = 0;
  for (int c : hand)
    hand_size += c;

  const Move pass_sentinel(Move::Combination::kPass);
  std::vector<int> legal = compute_legal_moves(hand, pass_sentinel);
  legal.erase(std::remove(legal.begin(), legal.end(), kPASS), legal.end());
  // DFS: try legal moves with larger combinations first (then lower move_id).
  std::sort(legal.begin(), legal.end(), [](int a, int b) {
    int ca = MOVE_TO_CARDS[a][13];
    int cb = MOVE_TO_CARDS[b][13];
    if (ca != cb)
      return ca > cb;
    return a < b;
  });

  const auto &moves = all_moves();
  for (int mid : legal) {
    int cards = moves[mid].numCards();

    // A hand-emptying move wins immediately — no need to check response.
    if (cards == hand_size)
      return std::vector<int>{mid};

    // An unbeatable move: opponent must pass, we keep the lead.
    if (!opponent_can_respond(mid, hand, discard, opp_count)) {
      const auto &cost = MOVE_TO_CARDS[mid];
      std::array<int, 13> new_hand = hand;
      std::array<int, 13> new_discard = discard;
      for (int r = 0; r < 13; ++r) {
        new_hand[r] -= cost[r];
        new_discard[r] += cost[r];
      }
      auto rest = find_forced_win(new_hand, new_discard, opp_count);
      if (rest) {
        rest->insert(rest->begin(), mid);
        return rest;
      }
    }
  }
  return std::nullopt;
}

std::vector<int> compute_possible_moves(const std::array<int, 13> &player_hand,
                                        const std::array<int, 13> &discard_pile,
                                        int opponent_card_count,
                                        const Move &last_move,
                                        bool exclude_bombs) {
  std::vector<int> possible_moves;
  possible_moves.reserve(32);
  if (last_move.combination != Move::Combination::kPass) {
    possible_moves.push_back(kPASS);
  }
  const auto &moves = all_moves();
  for (int move_id = kSINGLE_START; move_id < LEGAL_MOVES_SIZE; ++move_id) {
    const Move &move = moves[move_id];
    if (exclude_bombs && move.combination == Move::Combination::kBomb)
      continue;
    if (last_move.combination != Move::Combination::kPass) {
      if (move.combination != Move::Combination::kBomb) {
        if (move.combination != last_move.combination ||
            move.rank <= last_move.rank)
          continue;
      } else {
        if (last_move.combination == Move::Combination::kBomb &&
            move.rank <= last_move.rank)
          continue;
      }
    }
    bool is_legal = opponent_card_count >= MOVE_TO_CARDS[move_id][13];
    if (is_legal) {
      for (int rank = 0; rank < 13; ++rank) {
        const int unseen =
            max_cards_in_deck_for_rank(rank) - player_hand[rank] -
            discard_pile[rank];
        if (unseen < MOVE_TO_CARDS[move_id][rank]) {
          is_legal = false;
          break;
        }
      }
    }
    if (is_legal)
      possible_moves.push_back(move_id);
  }
  return possible_moves;
}
