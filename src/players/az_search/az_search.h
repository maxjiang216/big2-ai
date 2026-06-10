#ifndef AZ_SEARCH_AZ_SEARCH_H
#define AZ_SEARCH_AZ_SEARCH_H

// AlphaZero-style expectimax search over a PLAIN TREE, from the root player's
// imperfect-information point of view. Torch-free: leaf evaluation is delegated
// to an abstract Evaluator (NN-backed in production, a stub in tests).
//
// HISTORY-AWARE: the net conditions on the full move history, so each leaf
// request carries the move-id path from the search root to the leaf (the
// evaluator owns the game-history prefix + KV cache). Because evals depend on
// the history, transposition sharing is INCORRECT here — the former memo DAG
// is gone; edges own their children and re-rooting frees everything outside
// the kept subtree.
//
// Value convention: every node value is P(the ROOT player / searcher wins).
// Player (our-turn) nodes back up the MAX over expanded children; opponent-turn
// nodes back up the behavior-prior-weighted AVERAGE. The search is not truncated
// at trick boundaries — crossing a pass just flips the side to move.
//
// Driver split (supports both single-game and batched self-play):
//   select_leaf()  descends root->leaf, auto-handling terminals;
//                  returns a LeafRequest (needs_eval + features incl. path
//                  tokens) for a brand-new node, else needs_eval=false.
//   apply_eval()   expands the pending leaf with the unified NN result
//                  (player or opp heads chosen by the leaf's side) + backs up.
//   run()          synchronous convenience loop over a supplied Evaluator.

#include "evaluator.h"
#include "move.h"
#include "util.h"

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace az_search {

// side: 0 = Us (root player to move), 1 = Opp.
enum { kUs = 0, kOpp = 1 };

struct SearchState {
  std::array<int, 13> our_hand;  // root player's exact remaining cards
  int opp_size;                  // opponent's card count
  std::array<int, 13> discard;   // cards played so far (fixed on first visit)
  int last_move;                 // trick to beat; kPASS == lead position
  int side;                      // who moves at this node
};

struct SearchConfig {
  float c_puct = 1.5f;
  int sims = 200;
  unsigned seed = 0;  // seeds the weighted-random opponent representative draw
  // Opponent collapsed-class representative selection: training self-play draws
  // it weighted-random (exploration); evaluation/play takes the max-probability
  // member (deterministic, strongest line).
  bool training = true;
};

struct Node;

struct Edge {
  int move_id;
  float prior;
  Node *child = nullptr;    // owned by this edge (plain tree)
  bool fused_pass = false;  // transition fused a forced kPASS: the edge's token
                            // suffix is {move_id, kPASS} (at most one fuse —
                            // after a fused pass the side to move holds the
                            // lead, where passing is never forced)
  float move_value = 0.5f;  // opp-parent edges: NN per-move value q_a baseline
};

// Hierarchical selection grouping. Opponent nodes use a flat 2-level structure
// (group by trick type via `idx`). Player nodes use a 3-level factored tree:
// Level-1 family groups, each holding Level-2 `sub` subgroups (primary axis:
// rank / straight high card / bomb kicker), each holding leaf edge `idx`.
struct EdgeGroup {
  int key;
  std::vector<int> idx;        // leaf edge indices (Node::edges)
  std::vector<EdgeGroup> sub;  // Level-2 subgroups (player nodes only)
  float prior_sum = 0.0f;
};

struct Node {
  SearchState st;
  bool expanded = false;
  bool terminal = false;
  int forced_win_move = -1;  // player node we can win from outright (incl. DS8)
  float nn_value = 0.5f;     // leaf estimate, P(root wins)
  float value = 0.5f;        // backed-up expectimax value
  long N = 0;                // visit count
  std::vector<Edge> edges;
  std::vector<EdgeGroup> groups;  // hierarchical selection over `edges`
};

struct LeafRequest {
  bool needs_eval = false;
  EvalFeatures feat{};  // owner_to_move says which heads the search will read
};

class Search {
public:
  // `history` = move ids from the GAME START to (and excluding) the root
  // position — the evaluator's prefix. The search only ever appends to it.
  Search(const SearchState &root_state, std::vector<int> history,
         const SearchConfig &cfg);

  LeafRequest select_leaf();
  void apply_eval(const NetEval &e);  // expands the pending leaf + backs up
  void run(Evaluator &ev);
  // Post-search forced extensions. Always runs the (eval-free) forced-WIN proof.
  // When an evaluator is supplied (play/eval path), also runs the NN-valued
  // forced-move expansion: play our provably-unbeatable lead moves and raise the
  // root value to the max over every forced-reachable node (not just the leaves),
  // overriding the root move when a forced line beats the searched value. Self-
  // play passes none (win proof only), keeping its visit-based policy targets.
  void finalize(Evaluator *ev = nullptr);

  // Subtree reuse: walk the real-game move suffix (new_history minus our
  // recorded prefix) down the tree; on a full match the played-out child
  // becomes the root (its subtree + visits survive) and everything else is
  // freed. Any divergence (unresolved edge, opponent representative mismatch)
  // discards the whole tree and starts fresh at `true_next`.
  void advance_root(const SearchState &true_next,
                    const std::vector<int> &new_history);

  // Results.
  int best_move() const;                                 // max-visit (or forced win)
  float root_value() const { return root_->value; }
  std::vector<std::pair<int, long>> root_visits() const; // (move_id, visits)
  // Root policy PRIOR as pseudo-counts (move_id, scaled prior). Used by self-play
  // as the policy target / move-sampling distribution when the visit distribution
  // is degenerate (e.g. sims=1, where the single eval only expands the root and no
  // child is visited). Returns the forced-win one-hot when one exists.
  std::vector<std::pair<int, long>> root_prior() const;
  bool root_terminal() const { return root_->terminal; }

  // Introspection (tests).
  std::size_t num_nodes() const { return live_nodes_; }
  std::size_t free_count() const { return free_list_.size(); }
  long root_n() const { return root_->N; }
  const Node *root_node() const { return root_; }
  const std::vector<int> &history() const { return history_; }

private:
  Node *alloc_node(const SearchState &s);
  void finalize_terminal(Node *n);
  void expand_player(Node *n, const float *logits);
  void expand_opp(Node *n, const float *move_value, const float *logits);
  void build_groups(Node *n);
  void apply_root_forced_win();
  void forced_expand_root(Evaluator &ev);  // NN-valued unbeatable-move extension
  Edge *select_edge(Node *n);
  Node *resolve_child(Node *parent, Edge &e);
  void recompute_value(Node *n);
  void backup();
  void release(Node *n);  // frees n and its whole (detached) subtree

  SearchConfig cfg_;
  std::mt19937 rng_;                             // opponent representative sampling
  std::deque<Node> arena_;                       // stable-address node pool
  std::vector<Node *> free_list_;                // freed slots, reused by alloc_node
  std::size_t live_nodes_ = 0;
  Node *root_ = nullptr;
  Node *pending_ = nullptr;                      // leaf awaiting apply_eval
  std::vector<Node *> path_;                     // root..leaf for current sim
  std::vector<int> history_;                     // game start -> root (prefix)
  std::vector<int> path_tokens_;                 // root -> current leaf tokens
};

// State transition + forced-pass fusion (exposed for unit tests).
SearchState az_transition(const SearchState &s, int move_id);
void normalize_forced_pass(SearchState &s);
int hand_size(const std::array<int, 13> &h);

// Tablebase / forced-win oracle from the root player's POV. Returns a move when
// the position is provably handled without search: a hand-emptying move, the
// opp-has-1-card tablebase line, or a `find_forced_win` sequence. Used by the
// player for root-skip and (the opp1 part) for in-tree move fixing.
std::optional<int> az_definitive_move(const SearchState &s, bool allow_forced_win);

// True iff `s` is a lead position from which we can force a win (proven).
bool az_is_forced_win(const SearchState &s, int &first_move);

}  // namespace az_search

#endif  // AZ_SEARCH_AZ_SEARCH_H
