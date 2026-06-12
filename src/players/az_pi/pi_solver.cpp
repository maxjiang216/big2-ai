#include "az_pi/pi_solver.h"

#include "az_pi/pi_prune.h"
#include "util.h"  // HandBits

#include <atomic>
#include <cstdint>
#include <memory>

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

// Persistent SHARED memo: the key is the complete PI state, so proven results
// stay valid across calls, trees, games, and threads.
//
// Flat lock-free transposition table (chess-TT style). Each entry is ONE
// atomic uint64: high 62 bits = key signature, low 2 bits = proof. Lookups
// and stores are relaxed atomics — no mutexes; a lost update or stale miss
// just costs a re-solve (it is a cache, never a correctness source). A
// signature match in 62 bits misidentifies a state with ~1e-12 probability
// per probe, far below any practical concern. Fixed footprint: 2^26 entries
// x 8 B = 512 MB, always-replace on collision (no eviction sweeps).
constexpr int kTableBits = 26;
constexpr std::size_t kTableSize = std::size_t(1) << kTableBits;
constexpr int kProbes = 4;  // linear probe window (half a cache line)
std::unique_ptr<std::atomic<uint64_t>[]> g_table(
    new std::atomic<uint64_t>[kTableSize]());

// Signature: mix independent from the index bits; never 0 when used because
// the stored word always carries a nonzero proof in the low bits.
inline uint64_t sig_of(const Key &k) {
  uint64_t x = k.h0 * 0x9e3779b97f4a7c15ull;
  x ^= (k.h1 + 0xc2b2ae3d27d4eb4full) * 0xff51afd7ed558ccdull;
  x ^= (x >> 33);
  x ^= static_cast<uint64_t>(k.lm_side) * 0xc4ceb9fe1a85ec53ull;
  x ^= (x >> 29);
  return x;
}

// Entry word layout: [sig:57][margin:5][proof:2]. Margin = the LOSER's final
// card count under the searched line (winner maximises it, loser minimises);
// a tie-break objective only, exact for losses, first-win-greedy for wins.
constexpr int kMarginBits = 5;
constexpr int kLowBits = kMarginBits + 2;

bool memo_find(const Key &k, Proof &out, int &margin) {
  const uint64_t sig = sig_of(k);
  const std::size_t base = sig & (kTableSize - 1);
  for (int p = 0; p < kProbes; ++p) {
    const uint64_t e = g_table[(base + p) & (kTableSize - 1)].load(
        std::memory_order_relaxed);
    if ((e >> kLowBits) == (sig >> kLowBits)) {
      out = static_cast<Proof>(e & 3u);
      margin = static_cast<int>((e >> 2) & 31u);
      return out != Proof::UNKNOWN;  // 0 only in never-written slots
    }
  }
  return false;
}

void memo_insert(const Key &k, Proof p, int margin) {
  const uint64_t sig = sig_of(k);
  const std::size_t base = sig & (kTableSize - 1);
  const uint64_t word = (sig & ~((uint64_t(1) << kLowBits) - 1)) |
                        (static_cast<uint64_t>(margin & 31) << 2) |
                        static_cast<uint64_t>(p);
  std::size_t victim = base;  // always-replace fallback: first probe slot
  for (int q = 0; q < kProbes; ++q) {
    const std::size_t i = (base + q) & (kTableSize - 1);
    const uint64_t e = g_table[i].load(std::memory_order_relaxed);
    if (e == 0 || (e >> kLowBits) == (sig >> kLowBits)) {
      victim = i;
      break;
    }
  }
  g_table[victim].store(word, std::memory_order_relaxed);
}

struct Solver {
  long budget;

  // Returns WIN/LOSS for g.current_player() (UNKNOWN if budget ran out) and
  // sets `margin` = loser's final card count on the proven line (see above).
  // Termination: every non-pass move strictly removes >=1 card; a pass leads to
  // a lead position where passing is illegal, so passes never chain. Total cards
  // therefore strictly decrease along any non-cyclic path.
  Proof rec(const Game &g, int &margin) {
    const Key key = make_key(g);
    Proof cached;
    if (memo_find(key, cached, margin)) return cached;

    if (budget <= 0) return Proof::UNKNOWN;
    --budget;

    Proof result = Proof::LOSS;  // no winning move found yet
    int loss_m = 32, win_m = -1;
    bool any_unknown = false;
    std::vector<int> legal = g.get_legal_moves();
    prune_dominated_attachments(g.player_hand(g.current_player()), legal);
    for (int mv : legal) {
      Game child = g;
      child.apply_move(mv);
      if (child.is_over()) {  // emptied our hand -> immediate win
        result = Proof::WIN;
        win_m = std::max(win_m,
                         g.get_player_hand_size(1 - g.current_player()));
        break;  // first proven win ends the scan (margin = best so far)
      }
      int sub_m = 0;
      const Proof sub = rec(child, sub_m);  // opponent to move in child
      if (sub == Proof::LOSS) {             // opponent loses => we win
        result = Proof::WIN;
        win_m = std::max(win_m, sub_m);
        break;
      }
      if (sub == Proof::WIN) loss_m = std::min(loss_m, sub_m);
      if (sub == Proof::UNKNOWN) any_unknown = true;
    }

    // If we found no win but some line is unresolved, the result is unknown and
    // must not be memoized (more budget could still prove a win).
    if (result != Proof::WIN && any_unknown) return Proof::UNKNOWN;

    margin = (result == Proof::WIN) ? win_m : loss_m;
    memo_insert(key, result, margin);
    return result;
  }
};

}  // namespace

Proof solve(const Game &g, const SolverLimits &limits, int *margin) {
  const int total =
      g.get_player_hand_size(0) + g.get_player_hand_size(1);
  if (total > limits.max_total_cards) return Proof::UNKNOWN;
  Solver s;
  s.budget = limits.node_budget;
  int m = 0;
  const Proof p = s.rec(g, m);
  if (margin) *margin = m;
  return p;
}

void solver_clear_memo() {
  for (std::size_t i = 0; i < kTableSize; ++i)
    g_table[i].store(0, std::memory_order_relaxed);
}

}  // namespace az_pi
