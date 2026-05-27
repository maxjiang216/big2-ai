#ifndef AZ_SEARCH_AZ_SEARCH_H
#define AZ_SEARCH_AZ_SEARCH_H

// AlphaZero-style expectimax search over a transposition DAG, from the root
// player's imperfect-information point of view. Torch-free: leaf evaluation is
// delegated to an abstract Evaluator (NN-backed in production, a stub in tests).
//
// Value convention: every node value is P(the ROOT player / searcher wins).
// Player (our-turn) nodes back up the MAX over expanded children; opponent-turn
// nodes back up the behavior-prior-weighted AVERAGE. The search is not truncated
// at trick boundaries — crossing a pass just flips the side to move.
//
// Driver split (supports both single-game and batched self-play):
//   select_leaf()  descends root->leaf, auto-handling terminals/transpositions;
//                  returns a LeafRequest (needs_eval + features) for a brand-new
//                  node, else needs_eval=false (already backed up).
//   apply_eval()   expands the pending leaf with an NN result and backs it up.
//   run()          synchronous convenience loop over a supplied Evaluator.

#include "evaluator.h"
#include "move.h"
#include "util.h"

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <unordered_map>
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
  Node *child = nullptr;  // resolved lazily (transposition-aware)
};

// Hierarchical selection grouping: edges sharing a trick type (full houses and
// bombs also split by rank, so the within-group choice is the auxiliary).
struct EdgeGroup {
  int key;
  std::vector<int> idx;  // indices into Node::edges
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
  bool is_player = false;
  PlayerFeatures pfeat{};
  OppFeatures ofeat{};
};

class Search {
public:
  Search(const SearchState &root_state, const SearchConfig &cfg);

  LeafRequest select_leaf();
  void apply_eval(float value, const float *logits);
  void run(Evaluator &ev);
  void finalize();  // post-search forced-win extension (run() calls it)

  // Subtree reuse: re-root onto the node for `true_next` (rebuilt from the real
  // game state, not the compressed key), reusing any prior search via the memo.
  void advance_root(const SearchState &true_next);

  // Results.
  int best_move() const;                                 // max-visit (or forced win)
  float root_value() const { return root_->value; }
  std::vector<std::pair<int, long>> root_visits() const; // (move_id, visits)
  bool root_terminal() const { return root_->terminal; }

  // Introspection (tests).
  std::size_t num_nodes() const { return memo_.size(); }
  long root_n() const { return root_->N; }
  const Node *root_node() const { return root_; }

private:
  Node *alloc_node(const SearchState &s);
  void finalize_terminal(Node *n);
  void expand(Node *n, float value, const float *logits);
  void expand_player(Node *n, const float *logits);
  void expand_opp(Node *n, const float *logits);
  void build_groups(Node *n);
  void apply_root_forced_win();
  Edge *select_edge(Node *n);
  Node *resolve_child(Node *parent, Edge &e);
  void recompute_value(Node *n);
  void backup();

  SearchConfig cfg_;
  std::mt19937 rng_;                             // opponent representative sampling
  std::deque<Node> arena_;                       // stable-address node pool
  std::unordered_map<uint64_t, Node *> memo_;    // transposition table
  Node *root_ = nullptr;
  Node *pending_ = nullptr;                      // leaf awaiting apply_eval
  std::vector<Node *> path_;                     // root..leaf for current sim
};

// State transition + key (exposed for unit tests).
SearchState az_transition(const SearchState &s, int move_id);
void normalize_forced_pass(SearchState &s);
uint64_t state_key(const SearchState &s);
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
