#include "az_pi/pi_search.h"

#include "az_search/considered_moves.h"
#include "az_search/features.h"  // trick_counts
#include "move.h"
#include "util.h"

#include <algorithm>
#include <cmath>

namespace az_pi {

using az_search::player_composed_logit;
using az_search::player_family_id;
using az_search::player_subgroup_key;

namespace {
float proven_value(const PiNode *n) {
  return (n->proof == Proof::WIN) ? 1.0f : 0.0f;  // terminal => proof LOSS => 0
}

// Greedy lexicographic score of the mover's hand AFTER playing mv (higher is
// better; mirrors greedy_hand_eval). Used to break ties among proven moves so
// the agent keeps shedding cards instead of passing out lost (or won) games.
std::array<int, 15> greedy_after(const Game &g, int mv) {
  auto h = g.player_hand(g.current_player());
  if (mv != kPASS)
    for (int r = 0; r < 13; ++r) h[r] -= MOVE_TO_CARDS[mv][r];
  int n_cards = 0, n_bombs = 0;
  for (int r = 0; r < 13; ++r) {
    n_cards += h[r];
    if ((r < 11 && h[r] == 4) || (r == 11 && h[r] == 3)) ++n_bombs;
  }
  std::array<int, 15> s;
  s[0] = (n_cards == 0) ? 1 : 0;
  s[1] = n_bombs;
  s[2] = -n_cards;
  for (int r = 0; r < 12; ++r) s[3 + r] = h[12 - r];  // 2s down to 4s
  return s;
}

// Best candidate by greedy score; pass only when it is the sole option.
int greedy_pick(const Game &g, const std::vector<int> &candidates) {
  int best = kPASS;
  std::array<int, 15> best_s{};
  bool have = false;
  for (int mv : candidates) {
    if (mv == kPASS) continue;
    auto s = greedy_after(g, mv);
    if (!have || best_s < s) {
      have = true;
      best_s = s;
      best = mv;
    }
  }
  return have ? best : kPASS;
}
}  // namespace

PiSearch::PiSearch(const Game &root, const PiSearchConfig &cfg,
                   PiNodePool *pool)
    : cfg_(cfg), rng_(cfg.seed), pool_(pool ? pool : &own_pool_) {
  path_.reserve(64);
  root_ = alloc_node(root);
}

PiSearch::~PiSearch() { release(root_); }

PiNode *PiSearch::alloc_node(const Game &g) {
  PiNode *n;
  if (!pool_->free_list.empty()) {
    n = pool_->free_list.back();
    pool_->free_list.pop_back();
    // Reset fields by hand: clear() keeps the vectors' capacity, so a warmed
    // pool re-expands nodes without touching malloc.
    n->expanded = false;
    n->proof = Proof::UNKNOWN;
    n->margin = 0;
    n->terminal = false;
    n->nn_value = 0.5f;
    n->N = 0;
    n->W = 0.0;
    n->edges.clear();
    n->fams.clear();
    n->subs.clear();
    n->sidx.clear();
  } else {
    pool_->storage.emplace_back();
    n = &pool_->storage.back();
  }
  n->state = g;
  ++live_nodes_;
  return n;
}

void PiSearch::release(PiNode *n) {
  if (!n) return;
  for (auto &e : n->edges)
    if (e.child) release(e.child);
  --live_nodes_;
  pool_->free_list.push_back(n);
}

// ----------------------------- expansion -----------------------------

void PiSearch::build_groups(PiNode *n) {
  n->fams.clear();
  n->subs.clear();
  n->sidx.clear();
  const int E = static_cast<int>(n->edges.size());
  tmp_fam_.resize(E);
  tmp_sub_.resize(E);
  for (int i = 0; i < E; ++i) {
    tmp_fam_[i] = player_family_id(n->edges[i].move_id);
    tmp_sub_[i] = player_subgroup_key(n->edges[i].move_id);
  }
  // Emit families and their subgroups as contiguous ranges. E is small
  // (legal moves of one position), so the quadratic scans are cheap.
  for (int i = 0; i < E; ++i) {
    bool seen = false;
    for (int j = 0; j < i; ++j)
      if (tmp_fam_[j] == tmp_fam_[i]) { seen = true; break; }
    if (seen) continue;
    PiFam F;
    F.key = tmp_fam_[i];
    F.sb = static_cast<int>(n->subs.size());
    for (int j = i; j < E; ++j) {
      if (tmp_fam_[j] != F.key) continue;
      F.prior_sum += n->edges[j].prior;
      bool sub_seen = false;
      for (int k = i; k < j; ++k)
        if (tmp_fam_[k] == F.key && tmp_sub_[k] == tmp_sub_[j]) {
          sub_seen = true;
          break;
        }
      if (sub_seen) continue;
      PiSub S;
      S.key = tmp_sub_[j];
      S.ib = static_cast<int>(n->sidx.size());
      for (int k = j; k < E; ++k)
        if (tmp_fam_[k] == F.key && tmp_sub_[k] == S.key) {
          S.prior_sum += n->edges[k].prior;
          n->sidx.push_back(k);
        }
      S.ie = static_cast<int>(n->sidx.size());
      n->subs.push_back(S);
    }
    F.se = static_cast<int>(n->subs.size());
    n->fams.push_back(F);
  }
}

void PiSearch::expand(PiNode *n, const float *logits) {
  const std::vector<int> legal = n->state.get_legal_moves();
  const int mover = n->state.current_player();
  const int our_size = n->state.get_player_hand_size(mover);

  // Composed logits -> softmax priors over legal moves.
  tmp_f_.resize(legal.size());
  std::vector<float> &comp = tmp_f_;
  float mx = -1e30f;
  for (std::size_t i = 0; i < legal.size(); ++i) {
    comp[i] = player_composed_logit(legal[i], logits);
    mx = std::max(mx, comp[i]);
  }
  float sum = 0.0f;
  for (std::size_t i = 0; i < legal.size(); ++i) {
    comp[i] = std::exp(comp[i] - mx);
    sum += comp[i];
  }
  const float inv = (sum > 0.0f) ? 1.0f / sum : 1.0f;

  n->edges.resize(legal.size());
  for (std::size_t i = 0; i < legal.size(); ++i) {
    n->edges[i].move_id = legal[i];
    n->edges[i].prior = comp[i] * inv;
    // Eager terminal proof: a move that empties our hand wins outright.
    if (all_moves()[legal[i]].numCards() == our_size) {
      n->edges[i].proof = Proof::WIN;
      n->edges[i].margin = static_cast<int8_t>(
          n->state.get_player_hand_size(1 - mover));
      n->proof = Proof::WIN;
      n->margin = n->edges[i].margin;
    }
  }
  build_groups(n);
  n->expanded = true;
}

// ------------------------- child resolution --------------------------

// Resolve the child state behind an edge. Proven children (terminal or
// solver-decided) get NO node: the edge alone carries proof + margin, the
// exact value backs up immediately, and selection never descends into a
// proven edge again — allocating a 92B Game + vectors for them was a large
// share of tree memory in endgames. Returns nullptr for proven children.
PiNode *PiSearch::resolve_child(PiNode *parent, PiEdge &e) {
  Game g = parent->state;
  g.apply_move(e.move_id);
  Proof child_proof;
  int m = 0;
  if (g.is_over()) {
    child_proof = Proof::LOSS;  // side to move at a terminal node is the loser
    m = g.get_player_hand_size(g.current_player());
  } else {
    child_proof = solve(g, cfg_.solver, &m);  // UNKNOWN if too large
  }
  if (child_proof != Proof::UNKNOWN) {
    e.proof = (child_proof == Proof::WIN) ? Proof::LOSS : Proof::WIN;
    e.margin = static_cast<int8_t>(m);
    propagate_proof_up();
    return nullptr;
  }
  PiNode *c = alloc_node(g);
  e.child = c;
  return c;
}

// Propagate a freshly proven child up the recorded selection path. path_.back()
// is the step from the parent into the just-resolved child.
void PiSearch::propagate_proof_up() {
  for (int k = static_cast<int>(path_.size()) - 1; k >= 0; --k) {
    PiNode *parent = path_[k].node;
    PiEdge &e = parent->edges[path_[k].ei];
    if (e.proof == Proof::UNKNOWN) break;  // edge not yet decided

    Proof parent_proof = Proof::UNKNOWN;
    int8_t parent_margin = 0;
    if (e.proof == Proof::WIN) {
      parent_proof = Proof::WIN;  // a winning reply exists
      parent_margin = e.margin;
      for (const auto &ed : parent->edges)  // best win seen so far
        if (ed.proof == Proof::WIN && ed.margin > parent_margin)
          parent_margin = ed.margin;
    } else {  // this edge loses; parent loses iff all do
      bool all_loss = true;
      int8_t min_m = 127;
      for (const auto &ed : parent->edges) {
        if (ed.proof != Proof::LOSS) { all_loss = false; break; }
        min_m = std::min(min_m, ed.margin);
      }
      if (all_loss) {
        parent_proof = Proof::LOSS;
        parent_margin = min_m;  // losing: shed as much as possible
      }
    }
    if (parent_proof == Proof::UNKNOWN) break;
    if (parent->proof == parent_proof) {
      // Verdict already known; a later-searched line may still improve the
      // margin (better win / softer loss). Record it, then stop.
      const bool better = (parent_proof == Proof::WIN)
                              ? parent_margin > parent->margin
                              : parent_margin < parent->margin;
      if (better) {
        parent->margin = parent_margin;
        if (k > 0)
          path_[k - 1].node->edges[path_[k - 1].ei].margin = parent_margin;
      }
      break;
    }
    parent->proof = parent_proof;
    parent->margin = parent_margin;

    // Push the parent's verdict one level higher.
    if (k == 0) break;
    PiEdge &up = path_[k - 1].node->edges[path_[k - 1].ei];
    up.proof = (parent_proof == Proof::WIN) ? Proof::LOSS : Proof::WIN;
    up.margin = parent_margin;
  }
}

// ----------------------------- selection -----------------------------

void PiSearch::run(PiEvaluator &ev) {
  for (int i = 0; i < cfg_.sims; ++i) {
    PiLeafRequest req = select_leaf();
    if (req.needs_eval) {
      std::vector<PiEvalFeatures> batch{req.feat};
      auto out = ev.eval_batch(batch);
      apply_eval(out[0]);
    }
  }
}

PiLeafRequest PiSearch::select_leaf() {
  path_.clear();
  PiNode *n = root_;
  while (true) {
    if (n->terminal || n->proof != Proof::UNKNOWN) {
      // n is the reached node; backup() resolves it from the last path step
      // (or root_ when the path is empty) and propagates the exact value.
      backup(proven_value(n));
      return PiLeafRequest{false, {}};
    }
    if (!n->expanded) {
      pending_ = n;
      PiLeafRequest req;
      req.needs_eval = true;
      const int mover = n->state.current_player();
      req.feat.hand = n->state.player_hand(mover);
      req.feat.opp_hand = n->state.player_hand(1 - mover);
      req.feat.trick = az_search::trick_counts(n->state.last_move_id());
      req.feat.our_size = n->state.get_player_hand_size(mover);
      req.feat.opp_size = n->state.get_player_hand_size(1 - mover);
      return req;
    }

    // Hierarchical PUCT descent.
    const double sqrtN = std::sqrt(static_cast<double>(std::max(1, n->N)));
    int gi = 0;
    double best = -1e30;
    for (int g = 0; g < static_cast<int>(n->fams.size()); ++g) {
      const PiFam &G = n->fams[g];
      const double Q = (G.N > 0) ? G.W / G.N : 0.5;
      const double u = cfg_.c_puct * G.prior_sum * sqrtN / (1.0 + G.N);
      if (Q + u > best) { best = Q + u; gi = g; }
    }
    const PiFam &G = n->fams[gi];
    const double sqrtNg = std::sqrt(static_cast<double>(std::max(1, G.N)));
    int si = G.sb;
    best = -1e30;
    for (int s = G.sb; s < G.se; ++s) {
      const PiSub &S = n->subs[s];
      const double Q = (S.N > 0) ? S.W / S.N : 0.5;
      const double u = cfg_.c_puct * S.prior_sum * sqrtNg / (1.0 + S.N);
      if (Q + u > best) { best = Q + u; si = s; }
    }
    const PiSub &S = n->subs[si];
    const double sqrtNs = std::sqrt(static_cast<double>(std::max(1, S.N)));
    int ei = n->sidx[S.ib];
    best = -1e30;
    int ei_any = n->sidx[S.ib];
    double best_any = -1e30;
    bool found_nonloss = false;
    for (int ii = S.ib; ii < S.ie; ++ii) {
      const int idx = n->sidx[ii];
      const PiEdge &e = n->edges[idx];
      const double childN = e.child ? static_cast<double>(e.child->N) : 0.0;
      const double Q =
          (e.child && e.child->N > 0) ? 1.0 - e.child->W / e.child->N : 0.5;
      const double u = cfg_.c_puct * e.prior * sqrtNs / (1.0 + childN);
      const double score = Q + u;
      if (score > best_any) { best_any = score; ei_any = idx; }
      if (e.proof == Proof::LOSS) continue;  // avoid provably losing replies
      if (score > best) { best = score; ei = idx; found_nonloss = true; }
    }
    if (!found_nonloss) ei = ei_any;

    path_.push_back(Step{n, gi, si, ei});
    PiEdge &e = n->edges[ei];
    if (!e.child && e.proof != Proof::UNKNOWN) {
      // Proven childless edge: exact value, child-mover perspective.
      backup(e.proof == Proof::WIN ? 0.0f : 1.0f);
      return PiLeafRequest{false, {}};
    }
    PiNode *c = e.child ? e.child : resolve_child(n, e);
    if (!c) {  // freshly proven by the solver — same exact backup
      backup(e.proof == Proof::WIN ? 0.0f : 1.0f);
      return PiLeafRequest{false, {}};
    }
    n = c;
  }
}

void PiSearch::apply_eval(const PiNetEval &e) {
  PiNode *n = pending_;
  pending_ = nullptr;
  n->nn_value = e.value;
  expand(n, e.policy.data());
  float v = n->nn_value;
  if (n->proof != Proof::UNKNOWN) {
    v = proven_value(n);
    // n just became proven by an eager-terminal edge; lift it up the path.
    if (!path_.empty()) {
      PiEdge &up = path_.back().node->edges[path_.back().ei];
      up.proof = (n->proof == Proof::WIN) ? Proof::LOSS : Proof::WIN;
      up.margin = n->margin;
      propagate_proof_up();
    }
  }
  backup(v);
}

// Back up `leaf_value` (mover-perspective P(win) at the reached node) along the
// recorded path, flipping perspective one ply at a time.
void PiSearch::backup(float leaf_value) {
  // The reached node is the child of path_.back() (or root_ when path_ empty).
  // Proven edges have no child node — their exact value still flows up the
  // path; only the per-leaf stats are skipped (selection short-circuits on
  // the edge proof, never on child Q).
  PiNode *leaf =
      path_.empty() ? root_ : path_.back().node->edges[path_.back().ei].child;
  if (leaf) {
    leaf->N++;
    leaf->W += leaf_value;
  }
  float cur = leaf_value;
  for (int k = static_cast<int>(path_.size()) - 1; k >= 0; --k) {
    cur = 1.0f - cur;
    Step &s = path_[k];
    s.node->N++;
    s.node->W += cur;
    PiFam &fam = s.node->fams[s.gi];
    fam.N++;
    fam.W += cur;
    PiSub &sg = s.node->subs[s.si];
    sg.N++;
    sg.W += cur;
  }
}

// ----------------------------- root noise ----------------------------

void PiSearch::apply_root_noise() {
  if (!root_->expanded || root_->edges.empty()) return;
  std::gamma_distribution<float> gamma(cfg_.dir_alpha, 1.0f);
  tmp_f_.resize(root_->edges.size());
  std::vector<float> &noise = tmp_f_;
  float sum = 0.0f;
  for (auto &v : noise) { v = gamma(rng_); sum += v; }
  const float inv = (sum > 0.0f) ? 1.0f / sum : 1.0f;
  for (std::size_t i = 0; i < root_->edges.size(); ++i) {
    root_->edges[i].prior = (1.0f - cfg_.dir_eps) * root_->edges[i].prior +
                            cfg_.dir_eps * noise[i] * inv;
  }
  build_groups(root_);  // refresh group prior sums
}

// ----------------------------- advance -------------------------------

void PiSearch::advance_root(int played_move) {
  Game next = root_->state;
  next.apply_move(played_move);

  PiNode *keep = nullptr;
  Proof played_proof = Proof::UNKNOWN;  // child-side proof, if edge was proven
  int8_t played_margin = 0;
  for (auto &e : root_->edges) {
    if (e.move_id == played_move) {
      keep = e.child;
      e.child = nullptr;
      if (e.proof != Proof::UNKNOWN) {
        played_proof =
            (e.proof == Proof::WIN) ? Proof::LOSS : Proof::WIN;
        played_margin = e.margin;
      }
    } else if (e.child) {
      release(e.child);
    }
  }
  root_->edges.clear();  // children already released/detached above
  --live_nodes_;
  pool_->free_list.push_back(root_);

  root_ = keep ? keep : alloc_node(next);
  if (!keep && played_proof != Proof::UNKNOWN) {
    // Advanced into a proven childless edge: carry the proof onto the fresh
    // root so early-stop and best_move keep their exact knowledge.
    root_->proof = played_proof;
    root_->margin = played_margin;
  }
  pending_ = nullptr;
  path_.clear();
  if (cfg_.root_noise) apply_root_noise();
}

// ----------------------------- results -------------------------------

float PiSearch::root_value() const {
  if (root_->proof == Proof::WIN) return 1.0f;
  if (root_->proof == Proof::LOSS) return 0.0f;
  return (root_->N > 0) ? static_cast<float>(root_->W / root_->N) : 0.5f;
}

int PiSearch::best_move() const {
  const Game &st = root_->state;

  if (root_->edges.empty()) {
    // Unexpanded root — possible after advance_root into a solver-proven child
    // (proven nodes are never expanded). Score the moves directly via the
    // solver (memo-warm, near-free): best margin first, then greedy.
    const auto legal = st.get_legal_moves();
    std::vector<int> cand;
    int best_m = -127;  // win: maximise loser(=opp) cards; loss: minimise ours
    for (int mv : legal) {
      Game child = st;
      child.apply_move(mv);
      int m;
      bool win = false;
      if (child.is_over()) {
        win = true;
        m = st.get_player_hand_size(1 - st.current_player());
      } else {
        int sm = 0;
        const Proof sub = solve(child, cfg_.solver, &sm);
        if (sub == Proof::UNKNOWN) continue;
        win = (sub == Proof::LOSS);
        m = sm;
      }
      if (root_->proof == Proof::WIN && !win) continue;  // only winning moves
      const int score = (root_->proof == Proof::WIN) ? m : -m;
      if (score > best_m) { best_m = score; cand.clear(); }
      if (score == best_m) cand.push_back(mv);
    }
    return greedy_pick(st, cand.empty() ? legal : cand);
  }

  // Proven LOSS: every move loses — pick the minimum-margin edge (shed the
  // most cards) instead of passing the game away; greedy breaks exact ties.
  if (root_->proof == Proof::LOSS) {
    int8_t min_m = 127;
    for (const auto &e : root_->edges) min_m = std::min(min_m, e.margin);
    std::vector<int> cand;
    for (const auto &e : root_->edges)
      if (e.margin == min_m) cand.push_back(e.move_id);
    return greedy_pick(st, cand);
  }
  // Proven WIN: among winning edges prefer the largest margin (opponent left
  // with the most cards); greedy breaks exact ties.
  if (root_->proof == Proof::WIN) {
    int8_t max_m = -127;
    for (const auto &e : root_->edges)
      if (e.proof == Proof::WIN) max_m = std::max(max_m, e.margin);
    std::vector<int> wins;
    for (const auto &e : root_->edges)
      if (e.proof == Proof::WIN && e.margin == max_m) wins.push_back(e.move_id);
    if (!wins.empty()) return greedy_pick(st, wins);
  }
  for (const auto &e : root_->edges)
    if (e.proof == Proof::WIN) return e.move_id;
  int best = root_->edges[0].move_id;
  long best_n = -1;
  for (const auto &e : root_->edges) {
    const long n = e.child ? e.child->N : 0;
    if (n > best_n) { best_n = n; best = e.move_id; }
  }
  return best;
}

std::vector<std::pair<int, long>> PiSearch::root_visits() const {
  std::vector<std::pair<int, long>> out;
  for (const auto &e : root_->edges) {
    const long n = e.child ? e.child->N : 0;
    if (n > 0) out.emplace_back(e.move_id, n);
  }
  return out;
}

int PiSearch::sample_move(float temperature, std::mt19937 &rng) const {
  auto visits = root_visits();
  // At temperature 0 (and whenever the root is proven) defer to best_move():
  // it honours proofs — a winning edge found late can have few visits.
  if (visits.empty() || temperature <= 1e-3f ||
      root_->proof != Proof::UNKNOWN)
    return best_move();
  std::vector<double> w(visits.size());
  double sum = 0.0;
  const double inv_t = 1.0 / temperature;
  for (std::size_t i = 0; i < visits.size(); ++i) {
    w[i] = std::pow(static_cast<double>(visits[i].second), inv_t);
    sum += w[i];
  }
  std::uniform_real_distribution<double> dist(0.0, sum);
  double r = dist(rng);
  for (std::size_t i = 0; i < visits.size(); ++i) {
    r -= w[i];
    if (r <= 0.0) return visits[i].first;
  }
  return visits.back().first;
}

}  // namespace az_pi
