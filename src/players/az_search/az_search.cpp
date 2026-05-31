#include "az_search.h"

#include "considered_moves.h"
#include "features.h"
#include "tablebase_opp1.h"
#include "typed_search/move_grouping.h"

#include <algorithm>
#include <cmath>

namespace az_search {

int hand_size(const std::array<int, 13> &h) {
  int s = 0;
  for (int c : h) s += c;
  return s;
}

// ---------------------------------------------------------------------------
// Transition + key
// ---------------------------------------------------------------------------

SearchState az_transition(const SearchState &s, int move_id) {
  SearchState ns = s;
  const int mover = s.side;
  if (move_id == kPASS) {
    // Trick ends; the player who played the last move (the other side) wins it
    // and leads the next trick. Hands/discard unchanged.
    ns.side = 1 - mover;
    ns.last_move = kPASS;
    return ns;
  }
  const auto &cost = MOVE_TO_CARDS[move_id];
  for (int r = 0; r < 13; ++r) ns.discard[r] += cost[r];
  if (mover == kUs)
    for (int r = 0; r < 13; ++r) ns.our_hand[r] -= cost[r];
  else
    ns.opp_size -= cost[13];
  ns.last_move = move_id;
  ns.side = 1 - mover;
  return ns;
}

// If the side to move at a response position has no legal non-pass play, the
// pass is forced (not a decision) — collapse it into the post-pass lead node for
// the trick winner. Loops in the (rare) chained case.
void normalize_forced_pass(SearchState &s) {
  for (;;) {
    if (s.last_move == kPASS) return;            // lead position: must play
    if (hand_size(s.our_hand) == 0 || s.opp_size == 0) return;  // terminal

    bool has_play;
    const Move last(s.last_move);
    if (s.side == kUs) {
      auto legal = compute_legal_moves(s.our_hand, last);
      has_play = std::any_of(legal.begin(), legal.end(),
                             [](int m) { return m != kPASS; });
    } else {
      auto poss = compute_possible_moves(s.our_hand, s.discard, s.opp_size, last,
                                         /*exclude_bombs=*/false);
      has_play = std::any_of(poss.begin(), poss.end(),
                             [](int m) { return m != kPASS; });
    }
    if (has_play) return;
    // Forced pass: trick winner (other side) leads.
    s.side = 1 - s.side;
    s.last_move = kPASS;
  }
}

uint64_t state_key(const SearchState &s) {
  // 3 bits per rank count (0..4) x13 = 39 bits; opp_size 5; combo 5; rank 4;
  // side 1. Discard and the trick auxiliary are intentionally excluded.
  uint64_t k = 0;
  for (int r = 0; r < 13; ++r)
    k |= (uint64_t)(s.our_hand[r] & 0x7) << (3 * r);
  k |= (uint64_t)(s.opp_size & 0x1F) << 39;
  const Move lm(s.last_move);
  k |= (uint64_t)((int)lm.combination & 0x1F) << 44;
  k |= (uint64_t)(lm.rank & 0xF) << 49;
  k |= (uint64_t)(s.side & 0x1) << 53;
  return k;
}

// ---------------------------------------------------------------------------
// Tablebase / forced-win oracle (mirrors core tablebase_peek, on SearchState).
// ---------------------------------------------------------------------------

// opp-has-1-card endgame line (lead position only). Returns a move id or none.
static std::optional<int> opp1_move(const SearchState &s) {
  if (s.opp_size != 1 || s.last_move != kPASS) return std::nullopt;
  auto legal = compute_legal_moves(s.our_hand, Move(kPASS));
  // All singles -> play the highest single.
  int best_rank = -1;
  bool all_singles = true;
  for (int mid : legal) {
    Move m(mid);
    if (m.combination != Move::Combination::kSingle) { all_singles = false; break; }
    if (m.rank > best_rank) best_rank = m.rank;
  }
  if (all_singles && best_rank != -1)
    return encodeMove(Move(Move::Combination::kSingle, best_rank));

  Opp1Result o = lookup_opp1(s.our_hand);
  if (o.first_move_id != 0) {
    for (int mid : legal)
      if (mid == o.first_move_id) return mid;
  }
  if (auto def = opp1_default_strategy_move(s.our_hand)) return *def;
  return std::nullopt;
}

bool az_is_forced_win(const SearchState &s, int &first_move) {
  if (s.last_move != kPASS) return false;  // need the initiative
  auto seq = find_forced_win(s.our_hand, s.discard, s.opp_size);
  if (!seq) return false;
  first_move = (*seq)[0];
  return true;
}

std::optional<int> az_definitive_move(const SearchState &s, bool allow_forced_win) {
  const int our_size = hand_size(s.our_hand);
  // Hand-emptying move wins outright.
  for (int mid : compute_legal_moves(s.our_hand, Move(s.last_move)))
    if (mid != kPASS && MOVE_TO_CARDS[mid][13] == our_size) return mid;
  if (auto m = opp1_move(s)) return m;
  if (allow_forced_win) {
    int fm;
    if (az_is_forced_win(s, fm)) return fm;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

Search::Search(const SearchState &root_state, const SearchConfig &cfg)
    : cfg_(cfg), rng_(cfg.seed) {
  root_ = alloc_node(root_state);
  root_->in_edges = 1;  // synthetic ref: the live root is never GC'd
  memo_[state_key(root_state)] = root_;
  finalize_terminal(root_);
}

// Reuse a freed slot when available (the deque never relocates, so addresses
// stay stable for any still-live pointers); otherwise grow the arena.
Node *Search::alloc_node(const SearchState &s) {
  Node *n;
  if (!free_list_.empty()) {
    n = free_list_.back();
    free_list_.pop_back();
    *n = Node{};
  } else {
    arena_.push_back(Node{});
    n = &arena_.back();
  }
  n->st = s;
  return n;
}

void Search::finalize_terminal(Node *n) {
  if (hand_size(n->st.our_hand) == 0) {
    n->terminal = true; n->expanded = true; n->nn_value = n->value = 1.0f;
  } else if (n->st.opp_size == 0) {
    n->terminal = true; n->expanded = true; n->nn_value = n->value = 0.0f;
  }
}

void Search::expand_player(Node *n, const float *logits) {
  const int our_size = hand_size(n->st.our_hand);
  auto legal = compute_legal_moves(n->st.our_hand, Move(n->st.last_move));

  // Auto-win: any move that empties our hand wins outright (subsumes DS8, which
  // is never a policy choice). Prefer the largest such move, lowest id.
  int win = -1, win_cards = -1;
  for (int m : legal) {
    if (m == kPASS) continue;
    int c = MOVE_TO_CARDS[m][13];
    if (c == our_size && c > win_cards) { win = m; win_cards = c; }
  }
  if (win != -1) {
    n->terminal = true; n->forced_win_move = win;
    n->nn_value = n->value = 1.0f;
    return;
  }

  // Tablebase (opp-has-1-card) in-tree fix: pin the move without searching the
  // choice, but still evaluate the resulting leaf so a value propagates up.
  // (Cheap lookup; the general forced-win search is reserved for the root.)
  if (auto tb = opp1_move(n->st)) {
    n->edges.push_back(Edge{*tb, 1.0f, nullptr});
    build_groups(n);
    return;
  }

  // Factored / hierarchical policy prior: each concrete legal move's logit is
  // the SUM of its path components (family -> rank/high -> aux/low, see
  // considered_moves.h); softmax over the legal concrete set. This masked
  // softmax over composed logits is mathematically the nested softmax
  // P(family)*P(rank|family)*P(aux|family,rank). One edge per concrete move.
  float maxl = -1e30f;
  for (int m : legal) maxl = std::max(maxl, player_composed_logit(m, logits));
  float sum = 0.0f;
  n->edges.reserve(legal.size());
  for (int m : legal) {
    float e = std::exp(player_composed_logit(m, logits) - maxl);
    n->edges.push_back(Edge{m, e, nullptr});
    sum += e;
  }
  if (sum > 0.0f)
    for (auto &e : n->edges) e.prior /= sum;
  build_groups(n);
}

void Search::expand_opp(Node *n, const float *move_value, const float *logits) {
  auto poss = compute_possible_moves(n->st.our_hand, n->st.discard, n->st.opp_size,
                                     Move(n->st.last_move), /*exclude_bombs=*/false);
  // Behavior priors over the plausible set (masked softmax via the opp head).
  std::vector<float> probs;
  probs.reserve(poss.size());
  float maxl = -1e30f;
  for (int m : poss) maxl = std::max(maxl, logits[az_opp_head_index(m)]);
  float sum = 0.0f;
  for (int m : poss) {
    float e = std::exp(logits[az_opp_head_index(m)] - maxl);
    probs.push_back(e);
    sum += e;
  }
  if (sum > 0)
    for (float &p : probs) p /= sum;

  // Collapse response-equivalent opponent moves (full houses by triple rank,
  // bombs by rank, in-range same-type merges) into one edge each. The whole
  // class is treated as a single move with the summed class probability; the
  // representative is chosen ONCE and fixed for the rest of the search — we only
  // ever search that member and forget the others (a later turn that plays a
  // different member starts fresh). Training self-play draws the representative
  // weighted-random by the within-class probs (exploration); evaluation/play
  // takes the max-probability member (deterministic, strongest line).
  auto classes = typed_search::group_opp_moves(n->st.our_hand, poss, probs);
  n->edges.reserve(classes.size());
  for (const auto &c : classes) {
    if (c.moves.empty()) continue;
    float psum = 0.0f;
    for (float p : c.probs) psum += p;
    int rep = c.moves[0];
    if (cfg_.training && psum > 0.0f) {
      std::uniform_real_distribution<float> dist(0.0f, psum);
      const float r = dist(rng_);
      float acc = 0.0f;
      for (std::size_t i = 0; i < c.moves.size(); ++i) {
        acc += c.probs[i];
        if (r <= acc) { rep = c.moves[i]; break; }
      }
    } else if (!cfg_.training) {
      float best_p = -1.0f;
      for (std::size_t i = 0; i < c.moves.size(); ++i)
        if (c.probs[i] > best_p) { best_p = c.probs[i]; rep = c.moves[i]; }
    }
    n->edges.push_back(Edge{rep, psum, nullptr});
  }
  build_groups(n);

  // Per-edge value baseline q_a from the opp net's per-move value head, and the
  // node's leaf value as the prior-weighted mean Sum_a prior(a)*q_a (the
  // control-variate baseline — matches the priors, which is what makes the
  // partial-expansion backup unbiased). Priors already sum to 1 over edges.
  double v = 0.0;
  for (auto &e : n->edges) {
    e.move_value = move_value[az_opp_head_index(e.move_id)];
    v += (double)e.prior * (double)e.move_value;
  }
  n->nn_value = n->value = n->edges.empty() ? 0.5f : (float)v;
}

// Group key for hierarchical selection: trick type, with full houses / bombs
// also split by rank so the within-group choice is the auxiliary card.
static int group_key(int move_id) {
  const Move m(move_id);
  const int combo = static_cast<int>(m.combination);
  if (m.combination == Move::Combination::kFullHouse ||
      m.combination == Move::Combination::kBomb)
    return combo * 32 + m.rank;
  return combo;
}

void Search::build_groups(Node *n) {
  n->groups.clear();
  if (n->st.side != kUs) {
    // Opponent node: flat 2-level grouping by trick type (FH/bombs by rank).
    for (int i = 0; i < (int)n->edges.size(); ++i) {
      const int k = group_key(n->edges[i].move_id);
      EdgeGroup *g = nullptr;
      for (auto &eg : n->groups)
        if (eg.key == k) { g = &eg; break; }
      if (!g) { n->groups.push_back(EdgeGroup{k, {}, {}, 0.0f}); g = &n->groups.back(); }
      g->idx.push_back(i);
      g->prior_sum += n->edges[i].prior;
    }
    return;
  }
  // Player node: factored 3-level tree — Level-1 family, Level-2 subgroup
  // (primary axis: rank / straight high / bomb kicker), Level-3 leaf edges.
  for (int i = 0; i < (int)n->edges.size(); ++i) {
    const int fam = player_family_id(n->edges[i].move_id);
    const int sk = player_subgroup_key(n->edges[i].move_id);
    EdgeGroup *g = nullptr;
    for (auto &eg : n->groups)
      if (eg.key == fam) { g = &eg; break; }
    if (!g) { n->groups.push_back(EdgeGroup{fam, {}, {}, 0.0f}); g = &n->groups.back(); }
    g->prior_sum += n->edges[i].prior;
    EdgeGroup *sg = nullptr;
    for (auto &s : g->sub)
      if (s.key == sk) { sg = &s; break; }
    if (!sg) { g->sub.push_back(EdgeGroup{sk, {}, {}, 0.0f}); sg = &g->sub.back(); }
    sg->prior_sum += n->edges[i].prior;
    sg->idx.push_back(i);
  }
}

// Visit count an edge contributes (0 until its child is created).
static inline long edge_n(const Edge &e) { return e.child ? e.child->N : 0; }

// Opponent node: flat 2-level most-undersampled selection (prob/(visits+1)),
// over response-equivalence classes built by group_opp_moves.
static Edge *select_opp(Node *n) {
  const EdgeGroup *bg = nullptr;
  double bg_score = -1e30;
  for (const auto &g : n->groups) {
    long Ng = 0;
    for (int i : g.idx) Ng += edge_n(n->edges[i]);
    double score = (double)g.prior_sum / (double)(Ng + 1);
    if (score > bg_score) { bg_score = score; bg = &g; }
  }
  Edge *best = nullptr;
  double best_score = -1e30;
  for (int i : bg->idx) {
    Edge &e = n->edges[i];
    double score = (double)e.prior / (double)(edge_n(e) + 1);
    if (score > best_score) { best_score = score; best = &e; }
  }
  return best;
}

// Player node: hierarchical PUCT down the factored tree (family -> subgroup ->
// edge). Each level uses the CONDITIONAL prior (child prior_sum / parent
// prior_sum) and that level's summed visits, so e.g. the aux distribution within
// a full-house rank is a proper softmax over the aux logits that sums to 1.
static Edge *select_player(Node *n, double c_puct) {
  auto childQ = [&](const Edge &e) -> double {
    return (e.child && e.child->expanded) ? (double)e.child->value
                                          : (double)n->value;
  };
  // Level 1: family.
  long Ntot = 0;
  for (const auto &e : n->edges) Ntot += edge_n(e);
  const double sqTot = std::sqrt((double)Ntot + 1.0);
  const EdgeGroup *bg = nullptr;
  double bgs = -1e30;
  for (const auto &g : n->groups) {
    long Ng = 0;
    double Qg = -1.0;
    for (const auto &sg : g.sub)
      for (int i : sg.idx) { Ng += edge_n(n->edges[i]); Qg = std::max(Qg, childQ(n->edges[i])); }
    double score = Qg + c_puct * (double)g.prior_sum * sqTot / (1.0 + (double)Ng);
    if (score > bgs) { bgs = score; bg = &g; }
  }
  // Level 2: subgroup (conditional prior within the family).
  long Ng = 0;
  for (const auto &sg : bg->sub)
    for (int i : sg.idx) Ng += edge_n(n->edges[i]);
  const double sqG = std::sqrt((double)Ng + 1.0);
  const double invBg = bg->prior_sum > 0.0f ? 1.0 / (double)bg->prior_sum : 0.0;
  const EdgeGroup *bsg = nullptr;
  double bsgs = -1e30;
  for (const auto &sg : bg->sub) {
    long Ns = 0;
    double Qs = -1.0;
    for (int i : sg.idx) { Ns += edge_n(n->edges[i]); Qs = std::max(Qs, childQ(n->edges[i])); }
    double pcond = (double)sg.prior_sum * invBg;
    double score = Qs + c_puct * pcond * sqG / (1.0 + (double)Ns);
    if (score > bsgs) { bsgs = score; bsg = &sg; }
  }
  // Level 3: leaf edge (conditional prior within the subgroup).
  long Ns = 0;
  for (int i : bsg->idx) Ns += edge_n(n->edges[i]);
  const double sqS = std::sqrt((double)Ns + 1.0);
  const double invSg = bsg->prior_sum > 0.0f ? 1.0 / (double)bsg->prior_sum : 0.0;
  Edge *best = nullptr;
  double bes = -1e30;
  for (int i : bsg->idx) {
    Edge &e = n->edges[i];
    double pcond = (double)e.prior * invSg;
    double score = childQ(e) + c_puct * pcond * sqS / (1.0 + (double)edge_n(e));
    if (score > bes) { bes = score; best = &e; }
  }
  return best;
}

Edge *Search::select_edge(Node *n) {
  if (n->edges.empty()) return nullptr;
  if (n->groups.empty()) build_groups(n);  // safety (older nodes)
  return (n->st.side == kUs) ? select_player(n, cfg_.c_puct) : select_opp(n);
}

Node *Search::resolve_child(Node *parent, Edge &e) {
  if (e.child) return e.child;
  SearchState cs = az_transition(parent->st, e.move_id);
  normalize_forced_pass(cs);
  const uint64_t k = state_key(cs);
  auto it = memo_.find(k);
  Node *c;
  if (it != memo_.end()) {
    c = it->second;
  } else {
    c = alloc_node(cs);
    memo_[k] = c;
    finalize_terminal(c);
  }
  e.child = c;
  ++c->in_edges;  // this edge now references c (counted once, at first resolve)
  return c;
}

LeafRequest Search::select_leaf() {
  path_.clear();
  Node *cur = root_;
  for (;;) {
    path_.push_back(cur);
    if (cur->terminal) { backup(); return {}; }
    if (!cur->expanded) {
      pending_ = cur;
      LeafRequest req;
      req.needs_eval = true;
      const int our_size = hand_size(cur->st.our_hand);
      const auto opp_max = opp_max_counts(cur->st.our_hand, cur->st.discard);
      const auto trick = trick_counts(cur->st.last_move);
      if (cur->st.side == kUs) {
        req.is_player = true;
        req.pfeat = {cur->st.our_hand, opp_max, trick, cur->st.opp_size, our_size};
      } else {
        req.is_player = false;
        req.ofeat = {cur->st.our_hand, opp_max, trick, cur->st.opp_size, our_size};
      }
      return req;
    }
    Edge *e = select_edge(cur);
    if (!e) { backup(); return {}; }  // expanded but no edges (shouldn't happen)
    cur = resolve_child(cur, *e);
  }
}

void Search::recompute_value(Node *n) {
  if (n->terminal || !n->expanded) return;
  if (n->st.side == kUs) {
    float best = -1.0f;
    bool any = false;
    for (const auto &e : n->edges)
      if (e.child && e.child->expanded) { best = std::max(best, e.child->value); any = true; }
    n->value = any ? best : n->nn_value;
  } else {
    // Opponent (chance) node — control-variate backup. NO renormalization: each
    // unexpanded child keeps its NN per-move value q_a; expanded children use
    // their searched value. Priors sum to 1, so this stays an unbiased estimate
    // of Sum_a prior(a)*V_true(a) (renormalizing by expanded mass would bias it).
    if (n->edges.empty()) { n->value = n->nn_value; return; }
    double v = 0.0;
    for (const auto &e : n->edges)
      v += (double)e.prior *
           (double)((e.child && e.child->expanded) ? e.child->value : e.move_value);
    n->value = (float)v;
  }
}

void Search::backup() {
  for (Node *n : path_) n->N += 1;
  for (int i = (int)path_.size() - 1; i >= 0; --i) recompute_value(path_[i]);
}

void Search::apply_eval(const PlayerEval &e) {
  Node *n = pending_;
  n->expanded = true;
  n->nn_value = e.value;
  n->value = e.value;
  expand_player(n, e.logits.data());  // may override (auto-win / tablebase fix)
  backup();
  pending_ = nullptr;
}

void Search::apply_eval(const OppEval &e) {
  Node *n = pending_;
  n->expanded = true;
  expand_opp(n, e.move_value.data(), e.logits.data());  // derives nn_value/value
  backup();
  pending_ = nullptr;
}

void Search::run(Evaluator &ev) {
  for (int s = 0; s < cfg_.sims; ++s) {
    LeafRequest req = select_leaf();
    if (!req.needs_eval) continue;
    if (req.is_player)
      apply_eval(ev.eval_player(req.pfeat));
    else
      apply_eval(ev.eval_opp(req.ofeat));
  }
  finalize(&ev);
}

// Post-search forced-win proof: at the (lead) root, if we can prove a win,
// override the searched value to 1.0 and commit the winning first move. This
// runs after the simulation budget is spent and needs no evaluator.
void Search::apply_root_forced_win() {
  if (root_->terminal) return;
  int fm;
  if (az_is_forced_win(root_->st, fm)) {
    root_->forced_win_move = fm;
    root_->value = 1.0f;
  }
}

namespace {

// Opponent's possible per-rank holdings (public deduction), as a HandBits
// thermometer. INVARIANT down a forced line: each of our moves removes a card
// from our hand and adds it to the discard, leaving max_in_deck - hand - discard
// unchanged; and the opponent only ever passes. So build it once at the root.
HandBits opp_possible_bits(const std::array<int, 13> &hand,
                           const std::array<int, 13> &discard) {
  HandBits b{};
  for (int r = 0; r < 13; ++r) {
    int om = max_cards_in_deck_for_rank(r) - hand[r] - discard[r];
    if (om <= 0) continue;
    b.at1 |= static_cast<uint16_t>(1u << r);
    if (om >= 2) b.at2 |= static_cast<uint16_t>(1u << r);
    if (om >= 3) b.at3 |= static_cast<uint16_t>(1u << r);
    if (om >= 4) b.at4 |= static_cast<uint16_t>(1u << r);
  }
  return b;
}

constexpr int kForcedDepthCap = 4;

// Max value reachable from this (our-initiative) position by playing only moves
// the opponent provably cannot beat — the max over THIS node's NN value and the
// values of every forced-reachable descendant (1.0 on any hand-emptying line).
// `opp_bits` / `opp_size` are constant down the line (see opp_possible_bits).
float forced_subtree_value(const std::array<int, 13> &hand,
                           const std::array<int, 13> &discard,
                           const HandBits &opp_bits, int opp_size,
                           Evaluator &ev, int depth) {
  if (hand_size(hand) == 0) return 1.0f;
  PlayerFeatures f{hand, opp_max_counts(hand, discard), trick_counts(kPASS),
                   opp_size, hand_size(hand)};
  float best = ev.eval_player(f).value;  // option: stop here, trust this position
  if (best >= 1.0f || depth <= 0) return best;

  auto legal = compute_legal_moves(hand, Move(kPASS));
  for (int m : legal) {
    if (m == kPASS) continue;
    if (opponent_can_respond(m, opp_bits, opp_size)) continue;  // beatable, skip
    std::array<int, 13> nh = hand, nd = discard;
    const auto &cost = MOVE_TO_CARDS[m];
    for (int r = 0; r < 13; ++r) { nh[r] -= cost[r]; nd[r] += cost[r]; }
    if (hand_size(nh) == 0) return 1.0f;
    best = std::max(best,
                    forced_subtree_value(nh, nd, opp_bits, opp_size, ev, depth - 1));
    if (best >= 1.0f) return best;
  }
  return best;
}

}  // namespace

// NN-valued forced-move extension at the (lead) root: if playing an unbeatable
// move leads to a position (or forced descendant) the net values above the
// searched root value, raise the root value and commit that move. Per the
// max-over-all-nodes rule, a forced child is judged by the best stopping point
// in its forced subtree, so we never over-commit to a full forced sequence.
void Search::forced_expand_root(Evaluator &ev) {
  if (root_->terminal || root_->forced_win_move != -1) return;  // win proof wins
  if (root_->st.last_move != kPASS) return;  // only with the initiative

  const HandBits ob =
      opp_possible_bits(root_->st.our_hand, root_->st.discard);
  const int osz = root_->st.opp_size;

  float best = root_->value;  // floor: never downgrade the searched value
  int best_move = -1;
  for (int m : compute_legal_moves(root_->st.our_hand, Move(kPASS))) {
    if (m == kPASS) continue;
    if (opponent_can_respond(m, ob, osz)) continue;
    std::array<int, 13> nh = root_->st.our_hand, nd = root_->st.discard;
    const auto &cost = MOVE_TO_CARDS[m];
    for (int r = 0; r < 13; ++r) { nh[r] -= cost[r]; nd[r] += cost[r]; }
    float v = (hand_size(nh) == 0)
                  ? 1.0f
                  : forced_subtree_value(nh, nd, ob, osz, ev, kForcedDepthCap - 1);
    if (v > best) { best = v; best_move = m; }
    if (best >= 1.0f) break;
  }
  root_->value = best;
  if (best_move != -1) root_->forced_win_move = best_move;
}

void Search::finalize(Evaluator *ev) {
  apply_root_forced_win();          // eval-free forced-WIN proof (both paths)
  if (ev) forced_expand_root(*ev);  // NN-valued extension (play/eval path only)
}

void Search::advance_root(const SearchState &true_next) {
  pending_ = nullptr;
  path_.clear();
  const uint64_t k = state_key(true_next);
  Node *old_root = root_;
  auto it = memo_.find(k);
  Node *new_root;
  if (it != memo_.end()) {
    // Reuse the prior subtree; refresh the (key-invariant) discard / trick aux
    // from the real game state.
    it->second->st = true_next;
    new_root = it->second;
  } else {
    new_root = alloc_node(true_next);
    memo_[k] = new_root;
    finalize_terminal(new_root);
  }
  // Transfer the synthetic root ref to the new root, then drop it from the old
  // root. If the old root is now unreferenced, GC it and everything that becomes
  // unreachable. The DAG is acyclic, so the old root cannot lie under the new
  // root, and refcount cascade is leak-free / complete.
  ++new_root->in_edges;
  root_ = new_root;
  if (--old_root->in_edges == 0) release(old_root);
}

// Eager-cascade free of a subtree that just became unreachable. Worklist (not
// recursion) over the acyclic DAG: when an edge's child loses its last in-edge
// it is enqueued. Freed nodes leave the memo and return to the free list. Lazy
// reclamation is unsound here — a node reachable only through a freed parent
// would keep a phantom in-edge and could be reused for a stale key.
void Search::release(Node *start) {
  std::vector<Node *> stack{start};
  while (!stack.empty()) {
    Node *cur = stack.back();
    stack.pop_back();
    for (auto &e : cur->edges)
      if (e.child && --e.child->in_edges == 0 && e.child != root_)
        stack.push_back(e.child);
    memo_.erase(state_key(cur->st));
    free_list_.push_back(cur);
  }
}

int Search::best_move() const {
  if (root_->forced_win_move != -1) return root_->forced_win_move;
  int best = -1;
  long best_n = -1;
  float best_prior = -1.0f;
  for (const auto &e : root_->edges) {
    const long cn = e.child ? e.child->N : 0;
    if (cn > best_n || (cn == best_n && e.prior > best_prior)) {
      best_n = cn; best_prior = e.prior; best = e.move_id;
    }
  }
  return best;
}

std::vector<std::pair<int, long>> Search::root_visits() const {
  std::vector<std::pair<int, long>> out;
  if (root_->forced_win_move != -1) {
    out.emplace_back(root_->forced_win_move, 1);
    return out;
  }
  for (const auto &e : root_->edges) {
    const long cn = e.child ? e.child->N : 0;
    if (cn > 0) out.emplace_back(e.move_id, cn);
  }
  return out;
}

std::vector<std::pair<int, long>> Search::root_prior() const {
  std::vector<std::pair<int, long>> out;
  if (root_->forced_win_move != -1) {
    out.emplace_back(root_->forced_win_move, 1);
    return out;
  }
  constexpr long kScale = 100000;  // pseudo-counts; collate renormalizes by sum
  for (const auto &e : root_->edges) {
    const long c = std::lround((double)e.prior * kScale);
    if (c > 0) out.emplace_back(e.move_id, c);
  }
  if (out.empty() && !root_->edges.empty()) {  // all priors rounded to 0 -> argmax
    const Edge *best = &root_->edges.front();
    for (const auto &e : root_->edges)
      if (e.prior > best->prior) best = &e;
    out.emplace_back(best->move_id, 1);
  }
  return out;
}

}  // namespace az_search
