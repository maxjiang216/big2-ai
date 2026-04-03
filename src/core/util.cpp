#include "util.h"

#include <algorithm>
#include <iostream>
#include <optional>

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

std::vector<int> compute_legal_moves(const std::array<int, 13> &hand,
                                     const Move &last_move) {
  std::vector<int> legal_moves;
  legal_moves.reserve(32);
  if (last_move.combination != Move::Combination::kPass) {
    legal_moves.push_back(kPASS);
  }
  for (int move_id = kSINGLE_START; move_id < LEGAL_MOVES_SIZE; ++move_id) {
    Move move(move_id);
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
    bool is_legal = true;
    for (int rank = 0; rank < 13; ++rank) {
      if (hand[rank] < MOVE_TO_CARDS[move_id][rank]) {
        is_legal = false;
        break;
      }
    }
    if (is_legal)
      legal_moves.push_back(move_id);
  }
  return legal_moves;
}

const std::vector<std::vector<int>> &get_beating_moves() {
  static const std::vector<std::vector<int>> table = [] {
    std::vector<std::vector<int>> t(LEGAL_MOVES_SIZE);
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
  for (int mid2 : get_beating_moves()[move_id]) {
    Move m2(mid2);
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

  for (int mid : legal) {
    int cards = Move(mid).numCards();

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
  for (int move_id = kSINGLE_START; move_id < LEGAL_MOVES_SIZE; ++move_id) {
    Move move(move_id);
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
