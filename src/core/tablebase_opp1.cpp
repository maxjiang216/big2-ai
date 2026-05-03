#include "tablebase_opp1.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <unordered_map>

namespace {

struct HandHash {
  std::size_t operator()(const std::array<uint8_t, 13> &h) const noexcept {
    std::size_t hash = 14695981039346656037ull;
    for (uint8_t c : h) {
      hash ^= c;
      hash *= 1099511628211ull;
    }
    return hash;
  }
};

std::unordered_map<std::array<uint8_t, 13>, Opp1Result, HandHash> g_table;

} // namespace

void load_tablebase_opp1(const std::string &path) {
  g_table.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return;

  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || magic[0] != 'O' || magic[1] != 'P' || magic[2] != 'P' ||
      magic[3] != '1')
    return;

  std::uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), 4);
  if (!in)
    return;

  for (std::uint32_t i = 0; i < n; ++i) {
    std::array<uint8_t, 13> hand{};
    in.read(reinterpret_cast<char *>(hand.data()), 13);
    std::uint8_t score = 0;
    std::uint8_t lo = 0, hi = 0;
    in.read(reinterpret_cast<char *>(&score), 1);
    in.read(reinterpret_cast<char *>(&lo), 1);
    in.read(reinterpret_cast<char *>(&hi), 1);
    if (!in)
      break;
    int move_id = lo | (static_cast<int>(hi) << 8);
    g_table[hand] = Opp1Result{score, move_id};
  }
}

Opp1Result lookup_opp1(const std::array<int, 13> &hand) {
  std::array<uint8_t, 13> h{};
  for (int i = 0; i < 13; ++i)
    h[i] = static_cast<uint8_t>(hand[i]);
  auto it = g_table.find(h);
  if (it == g_table.end())
    return Opp1Result{};
  return it->second;
}

std::optional<int> opp1_default_strategy_move(const std::array<int, 13> &hand) {
  auto legal = compute_legal_moves(hand, kPASS);

  std::optional<int> best_bomb;
  for (int mid : legal) {
    if (mid == kPASS)
      continue;
    Move m(mid);
    if (m.combination == Move::Combination::kBomb) {
      if (!best_bomb || mid < *best_bomb)
        best_bomb = mid;
    }
  }
  if (best_bomb)
    return *best_bomb;

  std::optional<int> best_triple;
  for (int mid : legal) {
    if (mid == kPASS)
      continue;
    Move m(mid);
    if (m.combination == Move::Combination::kTriple) {
      if (!best_triple || mid < *best_triple)
        best_triple = mid;
    }
  }
  if (best_triple)
    return *best_triple;

  std::optional<int> best_double;
  for (int mid : legal) {
    if (mid == kPASS)
      continue;
    Move m(mid);
    if (m.combination == Move::Combination::kDouble) {
      if (!best_double || mid < *best_double)
        best_double = mid;
    }
  }
  if (best_double)
    return *best_double;

  std::optional<int> best_straight;
  int best_len = 0;
  for (int mid : legal) {
    if (mid == kPASS)
      continue;
    Move m(mid);
    if (m.combination >= Move::Combination::kStraight5 &&
        m.combination <= Move::Combination::kStraight13) {
      int len = m.numCards();
      if (len > best_len) {
        best_len = len;
        best_straight = mid;
      } else if (len == best_len &&
                 (!best_straight || mid < *best_straight)) {
        best_straight = mid;
      }
    }
  }
  if (best_straight)
    return *best_straight;

  std::optional<int> best_single;
  int best_rank = 99;
  for (int mid : legal) {
    if (mid == kPASS)
      continue;
    Move m(mid);
    if (m.combination == Move::Combination::kSingle) {
      if (!best_single || m.rank < best_rank) {
        best_rank = m.rank;
        best_single = mid;
      }
    }
  }
  if (best_single)
    return *best_single;

  return std::nullopt;
}
