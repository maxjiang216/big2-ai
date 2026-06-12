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
}  // namespace

PiSearch::PiSearch(const Game &root, const PiSearchConfig &cfg)
    : cfg_(cfg), rng_(cfg.seed) {
  path_.reserve(64);
  root_ = alloc_node(root);
}

PiNode *PiSearch::alloc_node(const Game &g) {
  PiNode *n;
  if (!free_list_.empty()) {
    n = free_list_.back();
    free_list_.pop_back();
    *n = PiNode();
  } else {
    arena_.emplace_back();
    n = &arena_.back();
  }
  n->state = g;
  ++live_nodes_;
  return n;
}

void PiSearch::release(PiNode *n) {
  if (!n) return;
  for (auto &e : n->edges)
    if (e.child) release(e.child);
  n->edges.clear();
  n->groups.clear();
  --live_nodes_;
  free_list_.push_back(n);
}

// ----------------------------- expansion -----------------------------

void PiSearch::build_groups(PiNode *n) {
  n->groups.clear();
  for (int i = 0; i < static_cast<int>(n->edges.size()); ++i) {
    const int mv = n->edges[i].move_id;
    const float pr = n->edges[i].prior;
    const int fam = player_family_id(mv);
    const int sub = player_subgroup_key(mv);

    PiGroup *fg = nullptr;
    for (auto &g : n->groups)
      if (g.key == fam) { fg = &g; break; }
    if (!fg) {
      n->groups.push_back(PiGroup{});
      fg = &n->groups.back();
      fg->key = fam;
    }
    fg->prior_sum += pr;

    PiGroup *sg = nullptr;
    for (auto &s : fg->sub)
      if (s.key == sub) { sg = &s; break; }
    if (!sg) {
      fg->sub.push_back(PiGroup{});
      sg = &fg->sub.back();
      sg->key = sub;
    }
    sg->prior_sum += pr;
    sg->idx.push_back(i);
  }
}

void PiSearch::expand(PiNode *n, const float *logits) {
  const std::vector<int> legal = n->state.get_legal_moves();
  const int mover = n->state.current_player();
  const int our_size = n->state.get_player_hand_size(mover);

  // Composed logits -> softmax priors over legal moves.
  std::vector<float> comp(legal.size());
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
      n->proof = Proof::WIN;
    }
  }
  build_groups(n);
  n->expanded = true;
}

// ------------------------- child resolution --------------------------

void PiSearch::classify_leaf(PiNode *child) {
  if (child->state.is_over()) {
    child->terminal = true;
    child->proof = Proof::LOSS;  // side to move at a terminal node is the loser
    return;
  }
  child->proof = solve(child->state, cfg_.solver);  // UNKNOWN if too large
}

PiNode *PiSearch::resolve_child(PiNode *parent, PiEdge &e) {
  Game g = parent->state;
  g.apply_move(e.move_id);
  PiNode *c = alloc_node(g);
  e.child = c;
  classify_leaf(c);
  if (c->proof == Proof::WIN) e.proof = Proof::LOSS;
  else if (c->proof == Proof::LOSS) e.proof = Proof::WIN;
  if (e.proof != Proof::UNKNOWN) propagate_proof_up();
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
    if (e.proof == Proof::WIN) {
      parent_proof = Proof::WIN;  // a winning reply exists
    } else {                      // this edge loses; parent loses iff all do
      bool all_loss = true;
      for (const auto &ed : parent->edges) {
        if (!ed.child || ed.proof != Proof::LOSS) { all_loss = false; break; }
      }
      if (all_loss) parent_proof = Proof::LOSS;
    }
    if (parent_proof == Proof::UNKNOWN) break;
    if (parent->proof == parent_proof) break;  // nothing new
    parent->proof = parent_proof;

    // Push the parent's verdict one level higher.
    if (k == 0) break;
    PiEdge &up = path_[k - 1].node->edges[path_[k - 1].ei];
    up.proof = (parent_proof == Proof::WIN) ? Proof::LOSS : Proof::WIN;
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
    const double sqrtN = std::sqrt(static_cast<double>(std::max(1L, n->N)));
    int gi = 0;
    double best = -1e30;
    for (int g = 0; g < static_cast<int>(n->groups.size()); ++g) {
      const PiGroup &G = n->groups[g];
      const double Q = (G.N > 0) ? G.W / G.N : 0.5;
      const double u = cfg_.c_puct * G.prior_sum * sqrtN / (1.0 + G.N);
      if (Q + u > best) { best = Q + u; gi = g; }
    }
    const PiGroup &G = n->groups[gi];
    const double sqrtNg = std::sqrt(static_cast<double>(std::max(1L, G.N)));
    int si = 0;
    best = -1e30;
    for (int s = 0; s < static_cast<int>(G.sub.size()); ++s) {
      const PiGroup &S = G.sub[s];
      const double Q = (S.N > 0) ? S.W / S.N : 0.5;
      const double u = cfg_.c_puct * S.prior_sum * sqrtNg / (1.0 + S.N);
      if (Q + u > best) { best = Q + u; si = s; }
    }
    const PiGroup &S = G.sub[si];
    const double sqrtNs = std::sqrt(static_cast<double>(std::max(1L, S.N)));
    int ei = S.idx[0];
    best = -1e30;
    int ei_any = S.idx[0];
    double best_any = -1e30;
    bool found_nonloss = false;
    for (int idx : S.idx) {
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
    PiNode *c = e.child ? e.child : resolve_child(n, e);
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
      propagate_proof_up();
    }
  }
  backup(v);
}

// Back up `leaf_value` (mover-perspective P(win) at the reached node) along the
// recorded path, flipping perspective one ply at a time.
void PiSearch::backup(float leaf_value) {
  // The reached node is the child of path_.back() (or root_ when path_ empty).
  PiNode *leaf =
      path_.empty() ? root_ : path_.back().node->edges[path_.back().ei].child;
  leaf->N++;
  leaf->W += leaf_value;
  float cur = leaf_value;
  for (int k = static_cast<int>(path_.size()) - 1; k >= 0; --k) {
    cur = 1.0f - cur;
    Step &s = path_[k];
    s.node->N++;
    s.node->W += cur;
    PiGroup &fam = s.node->groups[s.gi];
    fam.N++;
    fam.W += cur;
    PiGroup &sg = fam.sub[s.si];
    sg.N++;
    sg.W += cur;
  }
}

// ----------------------------- root noise ----------------------------

void PiSearch::apply_root_noise() {
  if (!root_->expanded || root_->edges.empty()) return;
  std::gamma_distribution<float> gamma(cfg_.dir_alpha, 1.0f);
  std::vector<float> noise(root_->edges.size());
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
  for (auto &e : root_->edges) {
    if (e.move_id == played_move) {
      keep = e.child;
      e.child = nullptr;
    } else if (e.child) {
      release(e.child);
    }
  }
  root_->edges.clear();
  root_->groups.clear();
  --live_nodes_;
  free_list_.push_back(root_);

  root_ = keep ? keep : alloc_node(next);
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
  // Prefer a proven win.
  for (const auto &e : root_->edges)
    if (e.proof == Proof::WIN) return e.move_id;
  int best = root_->edges.empty() ? 0 : root_->edges[0].move_id;
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
  if (visits.empty()) return best_move();
  if (temperature <= 1e-3f) {
    long best_n = -1;
    int best = visits[0].first;
    for (auto &v : visits)
      if (v.second > best_n) { best_n = v.second; best = v.first; }
    return best;
  }
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
