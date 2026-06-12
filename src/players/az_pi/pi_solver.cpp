#include "az_pi/pi_solver.h"

#include "util.h"  // HandBits

#include <cstdint>
#include <unordered_map>

namespace az_pi {
namespace {

// Memo key: both hands (cumulative bitmasks) + last move + side to move.
// Memoryless and perfect-information, so this is the complete state.
struct Key {
  uint64_t h0, h1;
  uint32_t lm_side;
  bool operator==(const Key &o) const {
    return h0 == o.h0 && h1 == o.h1 && lm_side == o.lm_side;
  }
};

struct KeyHash {
  std::size_t operator()(const Key &k) const {
    uint64_t x = k.h0 * 1099511628211ull;
    x ^= k.h1 + 0x9e3779b97f4a7c15ull + (x << 6) + (x >> 2);
    x ^= static_cast<uint64_t>(k.lm_side) * 0x100000001b3ull;
    return static_cast<std::size_t>(x ^ (x >> 29));
  }
};

uint64_t pack(const HandBits &h) {
  return static_cast<uint64_t>(h.at1) | (static_cast<uint64_t>(h.at2) << 16) |
         (static_cast<uint64_t>(h.at3) << 32) |
         (static_cast<uint64_t>(h.at4) << 48);
}

Key make_key(const Game &g) {
  Key k;
  k.h0 = pack(g.player_hand_bits(0));
  k.h1 = pack(g.player_hand_bits(1));
  k.lm_side = static_cast<uint32_t>(g.last_move_id()) * 2u +
              static_cast<uint32_t>(g.current_player());
  return k;
}

struct Solver {
  std::unordered_map<Key, Proof, KeyHash> memo;
  long budget;

  // Returns WIN/LOSS for g.current_player(), or UNKNOWN if budget ran out.
  // Termination: every non-pass move strictly removes >=1 card; a pass leads to
  // a lead position where passing is illegal, so passes never chain. Total cards
  // therefore strictly decrease along any non-cyclic path.
  Proof rec(const Game &g) {
    if (budget <= 0) return Proof::UNKNOWN;
    --budget;

    const Key key = make_key(g);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;

    Proof result = Proof::LOSS;  // no winning move found yet
    bool any_unknown = false;
    for (int mv : g.get_legal_moves()) {
      Game child = g;
      child.apply_move(mv);
      if (child.is_over()) {  // emptied our hand -> immediate win
        result = Proof::WIN;
        break;
      }
      const Proof sub = rec(child);  // opponent to move in child
      if (sub == Proof::LOSS) {      // opponent loses => we win
        result = Proof::WIN;
        break;
      }
      if (sub == Proof::UNKNOWN) any_unknown = true;
    }

    // If we found no win but some line is unresolved, the result is unknown and
    // must not be memoized (more budget could still prove a win).
    if (result != Proof::WIN && any_unknown) return Proof::UNKNOWN;

    memo[key] = result;
    return result;
  }
};

}  // namespace

Proof solve(const Game &g, const SolverLimits &limits) {
  const int total =
      g.get_player_hand_size(0) + g.get_player_hand_size(1);
  if (total > limits.max_total_cards) return Proof::UNKNOWN;
  Solver s;
  s.budget = limits.node_budget;
  return s.rec(g);
}

}  // namespace az_pi
