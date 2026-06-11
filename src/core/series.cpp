#include "series.h"

#include "move.h"
#include "util.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

int points_for_cards(int cards_remaining) {
  if (cards_remaining <= 0)
    return 0;
  if (cards_remaining <= 12)
    return cards_remaining;
  // 13 -> 20, 14 -> 30, 15 -> 40, 16 -> 50.
  return 10 * (cards_remaining - 11);
}

bool load_series_table(std::istream &in, SeriesTable &out) {
  out = SeriesTable{};
  std::string line;
  bool any = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    // Skip a header row like "a,b,v,natural_freq".
    if (!line.empty() && (line[0] < '0' || line[0] > '9'))
      continue;
    std::stringstream ss(line);
    std::string tok;
    int a, b;
    float v, nat = 0.0f;
    if (!std::getline(ss, tok, ',')) continue;
    a = std::stoi(tok);
    if (!std::getline(ss, tok, ',')) continue;
    b = std::stoi(tok);
    if (!std::getline(ss, tok, ',')) continue;
    v = std::stof(tok);
    if (std::getline(ss, tok, ','))
      nat = std::stof(tok);
    if (a < 0 || a >= kSeriesTarget || b < 0 || b >= kSeriesTarget)
      continue;
    out.v[a][b] = v;
    out.natural[a][b] = nat;
    any = true;
  }
  out.loaded = any;
  return any;
}

bool load_series_table(const std::string &csv_path, SeriesTable &out) {
  std::ifstream in(csv_path);
  if (!in.is_open()) {
    out = SeriesTable{};
    return false;
  }
  return load_series_table(in, out);
}

float series_value_after_win(const SeriesTable &t, int winner_pts, int loser_pts,
                             int loser_cards_remaining) {
  int p = points_for_cards(loser_cards_remaining);
  int next = winner_pts + p;
  if (next >= kSeriesTarget)
    return 1.0f;
  return t.v[next][loser_pts];
}

int sample_first_player_3s(const std::array<int, 13> &hand0,
                           const std::array<int, 13> &hand1,
                           std::mt19937 &rng) {
  // Ranks 3..K (indices 0..10) each have 4 suited copies. Assign suits within
  // each rank by shuffling its 4 copies across {hand0, hand1, unused}, then
  // walk the resulting suit->location map: spade pass over ranks 3..K, then
  // heart, diamond, club. The first pass that finds a copy in a player's hand
  // names the first player. (A specific suited copy in the unused pile means
  // that card was not dealt, so the rule falls through to the next suit/rank.)
  enum Loc { U = -1, P0 = 0, P1 = 1 };
  std::array<std::array<int, 4>, 11> suit_loc{}; // [rank][suit] -> loc
  for (int r = 0; r < 11; ++r) {
    std::array<int, 4> copies{};
    int idx = 0;
    for (int i = 0; i < hand0[r]; ++i) copies[idx++] = P0;
    for (int i = 0; i < hand1[r]; ++i) copies[idx++] = P1;
    while (idx < 4) copies[idx++] = U;
    std::shuffle(copies.begin(), copies.end(), rng);
    suit_loc[r] = copies; // copies[s] is the location of suit s for rank r
  }
  for (int s = 0; s < 4; ++s) {
    for (int r = 0; r < 11; ++r) {
      int loc = suit_loc[r][s];
      if (loc == P0) return 0;
      if (loc == P1) return 1;
    }
  }
  return 0; // unreachable in practice (16-card unused cannot hold all 44 copies)
}

bool hand_has_straight_lead(const std::array<int, 13> &hand) {
  std::vector<int> legal =
      compute_legal_moves(hand, Move(Move::Combination::kPass));
  const auto &moves = all_moves();
  for (int id : legal) {
    using C = Move::Combination;
    C c = moves[id].combination;
    if (c >= C::kStraight5) // every combination at/after kStraight5 is a straight
      return true;
  }
  return false;
}

int opp1_series_move(const std::array<int, 13> &hand) {
  using C = Move::Combination;
  // Smallest loose single (count == 1), used as a bomb auxiliary if present.
  int aux_rank = 0; // 0 = none
  for (int r = 0; r < 13; ++r) {
    if (hand[r] == 1) {
      aux_rank = r + 3;
      break;
    }
  }
  // 1) Bombs: four-of-a-kind (ranks 3..K) or the ace bomb (3 aces, index 11).
  for (int r = 0; r <= 10; ++r) {
    if (hand[r] == 4) {
      int bomb_rank = r + 3;
      int aux = (aux_rank != 0 && aux_rank != bomb_rank) ? aux_rank : 0;
      return encodeMove(Move(C::kBomb, bomb_rank, aux));
    }
  }
  if (hand[11] == 3) { // ace bomb (rank 14)
    int aux = (aux_rank != 0 && aux_rank != 14) ? aux_rank : 0;
    return encodeMove(Move(C::kBomb, 14, aux));
  }
  // 2) Triples (ranks 3..K; ace triple is the bomb above, no triple 2).
  for (int r = 0; r <= 10; ++r) {
    if (hand[r] == 3)
      return encodeMove(Move(C::kTriple, r + 3));
  }
  // 3) Doubles.
  for (int r = 0; r < 13; ++r) {
    if (hand[r] == 2)
      return encodeMove(Move(C::kDouble, r + 3));
  }
  // 4) Singles, highest rank first.
  for (int r = 12; r >= 0; --r) {
    if (hand[r] == 1)
      return encodeMove(Move(C::kSingle, r + 3));
  }
  return kPASS; // empty hand (caller should not reach here)
}
