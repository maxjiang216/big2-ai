#ifndef AZ_PI_PI_SEARCH_H
#define AZ_PI_PI_SEARCH_H

// Perfect-information AlphaZero search: a plain two-player PUCT tree over full
// Game states (both hands visible). Torch-free — leaf evaluation is delegated to
// an abstract PiEvaluator (NN-backed in production, a stub in tests).
//
// Differences from src/players/az_search (the imperfect-information search):
//   * MEMORYLESS leaves: features are (mover hand, opp hand, trick) only — no
//     history tokens, no KV cache.
//   * Standard SUM/AVERAGE backup at BOTH sides (no max backup, no opponent-node
//     collapse). The opponent is modelled as the same net, perspective-flipped
//     each ply: value v at a node = P(side to move there wins), backed up as
//     1 - v per ply.
//   * MCTS-SOLVER: exact win/loss proofs (terminal moves + pi_solver endgames)
//     propagate game-theoretically — a proven child needs no NN eval and feeds
//     an exact 1/0 into the average; selection avoids provably losing edges.
//   * Hierarchical PUCT (family -> rank/high/kicker subgroup -> leaf), reusing
//     the az_search factored player head, for efficient deep search.
//
// Driver split (single-game and batched self-play): select_leaf() descends to an
// unexpanded non-terminal node and returns a request to evaluate it (or backs up
// a terminal/proven line and returns needs_eval=false); apply_eval() expands the
// pending leaf with the NN result and backs up; run() is the synchronous loop.

#include "az_pi/pi_eval.h"
#include "az_pi/pi_solver.h"
#include "az_search/considered_moves.h"
#include "game.h"
#include "series.h"

#include <cstdint>
#include <deque>
#include <random>
#include <utility>
#include <vector>

namespace az_pi {

struct PiSearchConfig {
  float c_puct = 1.5f;
  int sims = 400;
  unsigned seed = 0;
  // Self-play exploration (off for eval/play).
  bool root_noise = false;
  float dir_eps = 0.25f;
  float dir_alpha = 0.3f;
  // Endgame solver applied at leaf creation.
  SolverLimits solver{};
  // Series objective (optional). When `series` is set, proven/terminal nodes
  // back up P(win the SERIES) = series_value_after_win(margin) instead of 1/0,
  // conditioned on the per-seat scores in `pts`. Margin ordering in best_move
  // is unchanged: series points are monotone in the margin.
  const SeriesTable *series = nullptr;
  int pts[2] = {0, 0};  // series points by SEAT (not mover-relative)
};

struct PiNode;

struct PiEdge {
  int move_id = 0;
  float prior = 0.0f;
  PiNode *child = nullptr;        // lazily created, owned by this edge
  Proof proof = Proof::UNKNOWN;   // for the PARENT mover playing this edge
  // Loser's final card count along this edge's proven line (valid when proof
  // is known). Secondary objective: a winning mover maximises it, a losing
  // mover minimises it — best over the lines actually searched.
  int8_t margin = 0;
};

// Hierarchical selection groups, stored FLAT (no nested vectors) so recycled
// nodes keep their vector capacity — steady-state expansion allocates nothing.
// Level-1 family rows index a contiguous range of Level-2 subgroup rows, which
// index a contiguous range of edge indices in PiNode::sidx. Aggregate visit
// stats are kept per group, in the PARENT mover's perspective (same as the
// node's own W/N).
// int32/float stats throughout: visit counts stay far below 2^31 and W sums
// of [0,1] values lose nothing meaningful at float precision, while the rows
// shrink 32B -> 20B (more rows per cache line in the hot PUCT scan).
struct PiFam {
  int32_t key = 0;
  float prior_sum = 0.0f;
  int32_t N = 0;
  float W = 0.0f;
  int16_t sb = 0, se = 0;  // subgroup rows: subs[sb, se)
};
struct PiSub {
  int32_t key = 0;
  float prior_sum = 0.0f;
  int32_t N = 0;
  float W = 0.0f;
  int16_t ib = 0, ie = 0;  // edge indices: sidx[ib, ie)
};

struct PiNode {
  Game state;
  bool expanded = false;
  Proof proof = Proof::UNKNOWN;  // for the side to move at this node
  int8_t margin = 0;             // loser's final cards on the proven line
  bool terminal = false;         // game already over at this node
  float nn_value = 0.5f;         // leaf estimate, P(mover wins)
  int32_t N = 0;
  float W = 0.0f;                 // sum of mover-perspective values over visits
  std::vector<PiEdge> edges;
  std::vector<PiFam> fams;       // hierarchical view over edges (flat)
  std::vector<PiSub> subs;
  std::vector<int> sidx;
};

// Node pool, shareable between successive games (e.g. one per self-play slot
// or per shard — anything single-threaded). Recycled nodes keep the capacity
// of their edge/group vectors, so a warmed pool expands with zero mallocs.
struct PiNodePool {
  std::deque<PiNode> storage;
  std::vector<PiNode *> free_list;
};

struct PiLeafRequest {
  bool needs_eval = false;
  PiEvalFeatures feat{};
};

class PiSearch {
 public:
  // `pool` (optional) shares node storage across trees/games — must only be
  // used from one thread. Falls back to a private pool.
  PiSearch(const Game &root, const PiSearchConfig &cfg,
           PiNodePool *pool = nullptr);
  ~PiSearch();  // returns all nodes to the pool
  PiSearch(const PiSearch &) = delete;
  PiSearch &operator=(const PiSearch &) = delete;

  PiLeafRequest select_leaf();
  void apply_eval(const PiNetEval &e);  // expand pending leaf + back up
  void run(PiEvaluator &ev);            // synchronous convenience loop

  // Subtree reuse: keep the played child's subtree as the new root, free the
  // rest. Re-applies root noise when configured.
  void advance_root(int played_move);

  // Results.
  // Root proven win/loss: further sims add nothing — caller can stop early.
  bool root_proven() const { return root_->proof != Proof::UNKNOWN; }
  int best_move() const;  // proven-win edge first, else max visits
  float root_value() const;
  std::vector<std::pair<int, long>> root_visits() const;  // (move_id, visits)
  int sample_move(float temperature, std::mt19937 &rng) const;

  // Introspection (tests).
  long root_n() const { return root_->N; }
  std::size_t num_nodes() const { return live_nodes_; }

 private:
  // Exact value of a proven position, in the given mover's perspective:
  // series_value_after_win over (cfg_.series, cfg_.pts, margin), or 1/0
  // when no series table is configured.
  float exact_value(int mover_seat, Proof proof, int margin) const;
  float node_exact_value(const PiNode *n) const;
  PiNode *alloc_node(const Game &g);
  void release(PiNode *n);
  void expand(PiNode *n, const float *logits);
  void build_groups(PiNode *n);
  PiNode *resolve_child(PiNode *parent, PiEdge &e);
  void propagate_proof_up();                  // along path_ after a proven leaf
  void apply_root_noise();
  void backup(float leaf_value);              // along path_, parity-flipped

  // Selection bookkeeping for one simulation.
  struct Step { PiNode *node; int gi; int si; int ei; };

  PiSearchConfig cfg_;
  std::mt19937 rng_;
  PiNodePool own_pool_;            // fallback when no external pool given
  PiNodePool *pool_;               // node storage (external or own)
  std::size_t live_nodes_ = 0;
  PiNode *root_ = nullptr;
  PiNode *pending_ = nullptr;      // leaf awaiting apply_eval
  std::vector<Step> path_;         // internal-node selections this sim
  std::vector<float> tmp_f_;       // scratch (priors / noise), reused
  std::vector<int> tmp_fam_, tmp_sub_;  // scratch per-edge group keys
};

}  // namespace az_pi

#endif  // AZ_PI_PI_SEARCH_H
