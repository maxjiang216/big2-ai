// Unit tests for the az_search move-set audit (considered_moves.h).
// Verifies the player/opponent head index mappings round-trip and that the two
// collapsed/dropped groups (TS5, DS8) satisfy the structural properties the
// audit relies on.

#include "az_search/az_search.h"
#include "az_search/considered_moves.h"
#include "az_search/evaluator.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <utility>

namespace {

using namespace az_search;

// Minimal cards-per-rank hand needed to make every move in `ids` legal at once
// is sum_r max over the moves of MOVE_TO_CARDS[id][r]. If that exceeds 16 for
// every distinct pair, no 16-card hand can hold two of them simultaneously.
int min_hand_for_pair(int a, int b) {
  int total = 0;
  for (int r = 0; r < 13; ++r)
    total += std::max(MOVE_TO_CARDS[a][r], MOVE_TO_CARDS[b][r]);
  return total;
}

void test_head_dims() {
  // Opponent head (flat): identity range + TS5 + DS8 slots.
  assert(AZ_OPP_HEAD_DIM == kTRIPLESTRAIGHT5_START + 2);
  assert(AZ_OPP_HEAD_DIM == 458);
  // Player head (factored / hierarchical).
  assert(AZ_PLAYER_HEAD_DIM == 138);
}

void test_opp_index_mapping() {
  int ts5_count = 0, ds8_count = 0;
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    const int oi = az_opp_head_index(id);
    assert(oi >= 0 && oi < AZ_OPP_HEAD_DIM);
    if (id < kTS5RangeBegin) {
      assert(oi == id);
      assert(az_opp_move_for_index(oi) == id);
    } else if (id < kTS5RangeEnd) {
      ++ts5_count;
      assert(oi == kTS5Slot);
    } else {
      ++ds8_count;
      assert(oi == kOppDS8Slot);
    }
  }
  assert(ts5_count == 7);
  assert(ds8_count == 5);
  assert(az_opp_move_for_index(kTS5Slot) == kTS5CanonicalMove);
  assert(az_opp_move_for_index(kOppDS8Slot) == kDS8CanonicalMove);
}

void test_player_factored_head() {
  using C = Move::Combination;
  // Every id resolves to a valid family + in-range path-logit indices.
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    const int fam = player_family_id(id);
    assert(fam >= kFamPass && fam <= kFamTplStraight);
    const PathLogits p = player_path_logits(id);
    assert(p.n == 1 || p.n == 2);
    for (int k = 0; k < p.n; ++k)
      assert(p.idx[k] >= 0 && p.idx[k] < AZ_PLAYER_HEAD_DIM);
  }
  // Full house: triple rank -> shared fh_rank slot; pair -> distinct fh_aux slot.
  for (int id = kFULL_HOUSE_START; id < kBOMB_START; ++id) {
    const Move m = all_moves()[id];
    assert(m.combination == C::kFullHouse);
    assert(player_family_id(id) == kFamFullHouse);
    const PathLogits p = player_path_logits(id);
    assert(p.n == 2);
    assert(p.idx[0] == kPHfhRank + (m.rank - 3));
    assert(p.idx[1] == kPHfhAux + (m.auxiliary - 3));
  }
  // Bomb: every bomb shares the bomb_entry; bare vs kicker distinct; rank uniform.
  for (int id = kBOMB_START; id < kSTRAIGHT5_START; ++id) {
    const Move m = all_moves()[id];
    assert(player_family_id(id) == kFamBomb);
    const PathLogits p = player_path_logits(id);
    assert(p.n == 2 && p.idx[0] == kPHbombEntry);
    if (m.auxiliary == 0) assert(p.idx[1] == kPHbombBare);
    else assert(p.idx[1] == kPHbombKicker + (m.auxiliary - 3));
  }
  // Straights: families + high/low indices in range.
  for (int id = kSTRAIGHT5_START; id < kDOUBLESTRAIGHT2_START; ++id) {
    assert(player_family_id(id) == kFamSglStraight);
    const PathLogits p = player_path_logits(id);
    assert(p.idx[0] >= kPHsglHigh && p.idx[0] < kPHsglHigh + 10);
    assert(p.idx[1] >= kPHsglLow && p.idx[1] < kPHsglLow + 10);
  }
  for (int id = kDOUBLESTRAIGHT2_START; id < kTRIPLESTRAIGHT2_START; ++id) {
    assert(player_family_id(id) == kFamDblStraight);
    const PathLogits p = player_path_logits(id);
    assert(p.idx[0] >= kPHdblHigh && p.idx[0] < kPHdblHigh + 11);
    assert(p.idx[1] >= kPHdblLow && p.idx[1] < kPHdblLow + 11);
  }
  for (int id = kTRIPLESTRAIGHT2_START; id < kDOUBLESTRAIGHT8_START; ++id) {
    assert(player_family_id(id) == kFamTplStraight);
    const PathLogits p = player_path_logits(id);
    assert(p.idx[0] >= kPHtplHigh && p.idx[0] < kPHtplHigh + 10);
    assert(p.idx[1] >= kPHtplLow && p.idx[1] < kPHtplLow + 10);
  }
  // 2-wrap endpoints: "23456" (lowest S5, eng hi=6, lo=2) and "JQKA2" (eng hi=15).
  {
    const PathLogits p = player_path_logits(kSTRAIGHT5_START);
    assert(all_moves()[kSTRAIGHT5_START].rank == 6);
    assert(p.idx[0] == kPHsglHigh + 0 && p.idx[1] == kPHsglLow + 0);
  }
  {
    const int id = kSTRAIGHT6_START - 1;  // highest S5
    assert(all_moves()[id].rank == 15);
    const PathLogits p = player_path_logits(id);
    assert(p.idx[0] == kPHsglHigh + 9 && p.idx[1] == kPHsglLow + 9);
  }
}

void test_collapsed_groups_structural() {
  // DS8 is always a full 16-card hand -> auto-win, never a policy choice.
  for (int id = kDS8RangeBegin; id < kDS8RangeEnd; ++id) {
    assert(all_moves()[id].combination == Move::Combination::kDoubleStraight8);
    assert(MOVE_TO_CARDS[id][13] == 16);
  }
  // TS5 uses 15 cards (a real continuation, leaves 1 card).
  for (int id = kTS5RangeBegin; id < kTS5RangeEnd; ++id) {
    assert(all_moves()[id].combination == Move::Combination::kTripleStraight5);
    assert(MOVE_TO_CARDS[id][13] == 15);
  }

  // At-most-one property: no 16-card hand can hold two distinct TS5 (or two
  // distinct DS8) simultaneously, so rank never offers a choice within a hand.
  for (int a = kTS5RangeBegin; a < kTS5RangeEnd; ++a)
    for (int b = a + 1; b < kTS5RangeEnd; ++b)
      assert(min_hand_for_pair(a, b) > 16);
  for (int a = kDS8RangeBegin; a < kDS8RangeEnd; ++a)
    for (int b = a + 1; b < kDS8RangeEnd; ++b)
      assert(min_hand_for_pair(a, b) > 16);
}

// ---------------------------------------------------------------------------
// Search-core tests (with a deterministic stub evaluator — no LibTorch).
// ---------------------------------------------------------------------------

using namespace az_search;

// Stub evaluator: value from caller-supplied lambdas, uniform (zero) logits by
// default. Counts calls so tests can confirm the NN is/ isn't queried, and can
// capture every EvalFeatures (incl. path tokens) for token-stream assertions.
struct StubEval : Evaluator {
  std::function<float(const EvalFeatures &)> pv = [](const EvalFeatures &) {
    return 0.5f;
  };
  std::function<float(const EvalFeatures &)> ov = [](const EvalFeatures &) {
    return 0.5f;
  };
  int player_calls = 0, opp_calls = 0;
  std::array<float, AZ_PLAYER_HEAD_DIM> plogits{};  // factored player head logits
  bool capture = false;
  std::vector<EvalFeatures> captured;

  NetEval eval(const EvalFeatures &f) override {
    if (capture) captured.push_back(f);
    NetEval e;
    if (f.owner_to_move) {
      ++player_calls;
      e.value = pv(f);
      e.policy = plogits;
    } else {
      ++opp_calls;
      e.qa.fill(ov(f));  // per-move q_a; uniform here unless a test varies it
      e.behavior.fill(0.0f);
    }
    return e;
  }
};

static std::array<int, 13> counts(std::initializer_list<std::pair<int, int>> kv) {
  std::array<int, 13> a{};
  for (auto [r, c] : kv) a[r] = c;
  return a;
}

void test_transition() {
  // Our move removes our cards, adds to discard, flips side.
  SearchState s{counts({{0, 1}, {1, 1}}), 5, {}, kPASS, kUs};
  SearchState ns = az_transition(s, kSINGLE_START);  // play single '3' (rank idx 0)
  assert(ns.our_hand[0] == 0 && ns.discard[0] == 1);
  assert(ns.side == kOpp && ns.last_move == kSINGLE_START);
  assert(ns.opp_size == 5);

  // Opp move decrements opp_size + adds to discard, leaves our hand.
  SearchState so{counts({{5, 1}}), 4, {}, kSINGLE_START + 4, kOpp};
  SearchState nso = az_transition(so, kSINGLE_START + 6);  // opp plays a single
  assert(nso.opp_size == 3 && nso.discard[6] == 1);
  assert(nso.our_hand[5] == 1 && nso.side == kUs);

  // Pass flips side and clears the trick.
  SearchState p = az_transition(so, kPASS);
  assert(p.side == kUs && p.last_move == kPASS && p.opp_size == 4);
}

void test_forced_win() {
  // Lead position, our hand is a single card -> playing it empties our hand.
  SearchState root{counts({{12, 1}}), 8, {}, kPASS, kUs};  // single '2'
  Search search(root, {}, {1.5f, 64});
  StubEval ev;  // value 0.5 — but a forced win should override to 1.0
  search.run(ev);
  assert(search.best_move() == kSINGLE_START + 12);  // the single '2'
  assert(std::abs(search.root_value() - 1.0f) < 1e-6);
}

void test_search_invariants_and_determinism() {
  // Response position: opp led a single '3'; we hold 4 and 5 (both beat it).
  SearchState root{counts({{1, 1}, {2, 1}}), 6, counts({{0, 1}}), kSINGLE_START, kUs};
  StubEval ev;
  Search s1(root, {}, {1.5f, 300});
  s1.run(ev);
  int bm = s1.best_move();

  // best_move is legal from the root.
  auto legal = compute_legal_moves(root.our_hand, Move(root.last_move));
  assert(std::find(legal.begin(), legal.end(), bm) != legal.end());

  // Root value within [0,1]; root visited; some nodes built.
  assert(s1.root_value() >= 0.0f && s1.root_value() <= 1.0f);
  assert(s1.root_n() > 0 && s1.num_nodes() > 1);

  // Determinism: identical config + stub -> identical outcome.
  StubEval ev2;
  Search s2(root, {}, {1.5f, 300});
  s2.run(ev2);
  assert(s2.best_move() == bm);
  assert(std::abs(s2.root_value() - s1.root_value()) < 1e-6);
}

void test_expectimax_backup() {
  // Player root backs up the MAX over expanded children; an opp child backs up
  // the unbiased CONTROL-VARIATE average: Sum_a prior(a)*(expanded? child.value
  // : q_a), with NO renormalization by expanded mass.
  SearchState root{counts({{1, 1}, {2, 1}, {7, 1}}), 6, counts({{0, 1}}),
                   kSINGLE_START, kUs};
  StubEval ev;
  // Opp per-move q distinct from the player leaf value (0.5) so the unbiased and
  // (old) renormalized estimators differ while any child is still unexpanded.
  ev.ov = [](const EvalFeatures &f) { return f.opp_size >= 6 ? 0.3f : 0.7f; };
  Search s(root, {}, {1.5f, 500});
  s.run(ev);

  const Node *r = s.root_node();
  assert(r->st.side == kUs && r->expanded && !r->terminal);

  // Player node: value == max over expanded children.
  float mx = -1.0f;
  bool any = false;
  const Node *opp_child = nullptr;
  for (const auto &e : r->edges) {
    if (e.child && e.child->expanded) {
      mx = std::max(mx, e.child->value);
      any = true;
      if (e.child->st.side == kOpp && !e.child->terminal && !opp_child)
        opp_child = e.child;
    }
  }
  assert(any);
  assert(std::abs(r->value - mx) < 1e-5);

  // Opp node leaf value == prior-weighted mean of q_a (control-variate baseline).
  // Opp node backed-up value == Sum prior*(expanded? child.value : q_a).
  if (opp_child) {
    double leaf = 0.0, v = 0.0;
    bool any_unexpanded = false;
    double ren_wsum = 0.0, ren_w = 0.0;  // old renormalized-by-expanded estimator
    for (const auto &e : opp_child->edges) {
      leaf += (double)e.prior * (double)e.move_value;
      const bool exp = e.child && e.child->expanded;
      v += (double)e.prior * (double)(exp ? e.child->value : e.move_value);
      if (!exp) any_unexpanded = true;
      else { ren_wsum += e.prior * e.child->value; ren_w += e.prior; }
    }
    assert(std::abs(opp_child->nn_value - (float)leaf) < 1e-5);
    assert(std::abs(opp_child->value - (float)v) < 1e-5);
    // Bias-fix regression guard: while a child is unexpanded and its q differs
    // from the expanded mean, the unbiased value must NOT equal the renormalized
    // one (which is what the old, biased backup computed).
    if (any_unexpanded && ren_w > 0) {
      float renorm = (float)(ren_wsum / ren_w);
      if (std::abs(renorm - 0.3f) > 1e-3)  // q here is 0.3; only meaningful if differs
        assert(std::abs(opp_child->value - renorm) > 1e-6);
    }
  }
}

void test_tree_node_bound() {
  // Plain tree (no transpositions): each simulation resolves at most one new
  // node, so the live node count stays <= sims + 1.
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}}), 4, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {}, {1.5f, 200});
  s.run(ev);
  assert((long)s.num_nodes() <= 201);
}

void test_tablebase_oracle() {
  // opp-has-1-card, all-singles -> play the highest single.
  SearchState s{counts({{0, 1}, {5, 1}}), 1, {}, kPASS, kUs};  // singles 3 and 8
  auto m = az_definitive_move(s, /*allow_forced_win=*/false);
  assert(m && *m == encodeMove(Move(Move::Combination::kSingle, 8)));

  // Hand-emptying move is always definitive.
  SearchState e{counts({{12, 1}}), 6, {}, kPASS, kUs};  // single '2'
  auto me = az_definitive_move(e, false);
  assert(me && *me == kSINGLE_START + 12);
}

void test_forced_win_extension() {
  // Lead, we hold single '2' + single '3'. The 2 is unbeatable only if the
  // opponent cannot bomb it: bury all three aces in the discard (no ace bomb)
  // and give the opponent just 2 cards (too few for a four-of-a-kind). Forced
  // line: play 2 (opp cannot respond), keep lead, play 3 -> empty.
  SearchState root{counts({{0, 1}, {12, 1}}), 2, counts({{11, 3}}), kPASS, kUs};
  int fm = -1;
  assert(az_is_forced_win(root, fm));

  StubEval ev;  // value 0.5 — the forced extension must override to 1.0
  Search s(root, {}, {1.5f, 64});
  s.run(ev);
  assert(std::abs(s.root_value() - 1.0f) < 1e-6);
  assert(s.best_move() == kSINGLE_START + 12);  // play the 2 first
}

void test_forced_move_expansion() {
  // Lead hand {3, 4, 2}, opp has 2 cards. The single '2' is unbeatable (highest
  // single; opp can't hold a 4-bomb with 2 cards), but playing it is NOT a proven
  // win (opp may beat our 3/4 afterward), so find_forced_win does not fire. With
  // sims=0 the search contributes nothing, so any value/move here comes purely
  // from the NN-valued forced-move expansion. The stub values the post-'2'
  // (size-2) position at 0.9, so the extension must raise the root to ~0.9 and
  // commit the single '2'.
  SearchState root{counts({{0, 1}, {1, 1}, {12, 1}}), 2, {}, kPASS, kUs};
  StubEval ev;
  ev.pv = [](const EvalFeatures &f) { return f.our_size == 2 ? 0.9f : 0.3f; };
  Search s(root, {}, {1.5f, /*sims=*/0});
  s.run(ev);
  assert(std::abs(s.root_value() - 0.9f) < 1e-4);
  assert(s.best_move() == kSINGLE_START + 12);  // play the unbeatable single '2'

  // Negative control: at a RESPONSE position (we lack the initiative) the
  // extension must not fire — nothing to force.
  SearchState resp{counts({{0, 1}, {1, 1}, {12, 1}}), 2, counts({{2, 1}}),
                   kSINGLE_START, kUs};
  StubEval ev2;
  ev2.pv = [](const EvalFeatures &) { return 0.9f; };
  Search s2(resp, {}, {1.5f, /*sims=*/0});
  s2.run(ev2);
  assert(s2.root_value() < 0.9f);  // unchanged from the unexpanded default
}

void test_player_composed_prior() {
  // The player edge priors are the masked softmax of COMPOSED logits (sum of the
  // factored-head path components per concrete move), summing to 1.
  SearchState root{counts({{0, 1}, {1, 1}, {2, 2}}), 6, {}, kPASS, kUs};
  StubEval ev;
  for (int i = 0; i < AZ_PLAYER_HEAD_DIM; ++i) ev.plogits[i] = 0.013f * i - 0.5f;
  Search s(root, {}, {1.5f, 1});  // one sim expands the root
  s.run(ev);
  const Node *r = s.root_node();
  assert(r->expanded && !r->edges.empty() && r->forced_win_move == -1);

  double maxl = -1e30;
  for (const auto &e : r->edges)
    maxl = std::max(maxl, (double)player_composed_logit(e.move_id, ev.plogits.data()));
  double Z = 0.0;
  for (const auto &e : r->edges)
    Z += std::exp((double)player_composed_logit(e.move_id, ev.plogits.data()) - maxl);
  double psum = 0.0;
  for (const auto &e : r->edges) {
    double expct =
        std::exp((double)player_composed_logit(e.move_id, ev.plogits.data()) - maxl) / Z;
    assert(std::abs((double)e.prior - expct) < 1e-5);
    psum += e.prior;
  }
  assert(std::abs(psum - 1.0) < 1e-4);
}

void test_hierarchical_groups_partition() {
  // Rich lead hand: singles, a pair, a triple -> several trick types -> groups.
  SearchState root{counts({{0, 1}, {1, 1}, {2, 2}, {5, 3}}), 7, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {}, {1.5f, 400});
  s.run(ev);
  const Node *r = s.root_node();
  assert(!r->terminal && r->expanded);

  // Player groups are a 3-level tree (family -> subgroup -> leaf edges) that
  // partitions the edges exactly once each, with prior_sum aggregating up.
  std::size_t covered = 0;
  std::vector<int> seen_fam;
  for (const auto &g : r->groups) {
    assert(g.idx.empty());  // player families hold subgroups, not direct edges
    assert(std::find(seen_fam.begin(), seen_fam.end(), g.key) == seen_fam.end());
    seen_fam.push_back(g.key);
    float fam_ps = 0.0f;
    std::vector<int> seen_sk;
    for (const auto &sg : g.sub) {
      assert(std::find(seen_sk.begin(), seen_sk.end(), sg.key) == seen_sk.end());
      seen_sk.push_back(sg.key);
      float sg_ps = 0.0f;
      for (int i : sg.idx) { ++covered; sg_ps += r->edges[i].prior; }
      assert(std::abs(sg_ps - sg.prior_sum) < 1e-4);
      fam_ps += sg.prior_sum;
    }
    assert(std::abs(fam_ps - g.prior_sum) < 1e-4);
  }
  assert(covered == r->edges.size());
  assert(r->groups.size() >= 2);  // singles + double + triple at least
}

void test_opp_node_grouping_wellformed() {
  // After we play from a lead, the resulting opp node should be expanded with a
  // valid (collapsed) behavior distribution summing to ~1.
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}, {7, 1}}), 6, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {}, {1.5f, 500});
  s.run(ev);
  const Node *opp = nullptr;
  for (const auto &e : s.root_node()->edges)
    if (e.child && e.child->expanded && e.child->st.side == kOpp &&
        !e.child->terminal) { opp = e.child; break; }
  if (opp) {
    assert(!opp->edges.empty());
    float psum = 0.0f;
    auto plausible = compute_possible_moves(opp->st.our_hand, opp->st.discard,
                                            opp->st.opp_size, Move(opp->st.last_move),
                                            /*exclude_bombs=*/false);
    for (const auto &e : opp->edges) {
      psum += e.prior;
      // Each collapsed-class representative is a genuine plausible opp move
      // (sampled from the class), never a fabricated/unsearched move.
      assert(std::find(plausible.begin(), plausible.end(), e.move_id) != plausible.end());
    }
    assert(std::abs(psum - 1.0f) < 1e-3);
    // Collapsing must not increase the edge count beyond the plausible set.
    assert(opp->edges.size() <= plausible.size());
  }
}

void test_leaf_path_tokens() {
  // Every captured leaf's path tokens, replayed from the root state through
  // az_transition, must land exactly on the leaf's feature state — the token
  // stream IS the move stream (forced fused passes appear explicitly).
  SearchState root{counts({{0, 1}, {1, 1}, {2, 2}, {5, 3}}), 7, {}, kPASS, kUs};
  StubEval ev;
  ev.capture = true;
  Search s(root, {}, {1.5f, 300});
  s.run(ev);
  assert(!ev.captured.empty());
  assert(ev.captured.front().path_tokens.empty());  // first eval is the root
  bool saw_player = false, saw_opp = false;
  for (const auto &f : ev.captured) {
    SearchState st = root;
    for (int tok : f.path_tokens) st = az_transition(st, tok);
    assert(st.our_hand == f.hand);
    assert(st.opp_size == f.opp_size);
    assert(hand_size(st.our_hand) == f.our_size);
    assert((st.side == kUs) == f.owner_to_move);
    assert(trick_counts(st.last_move) == f.trick);
    (f.owner_to_move ? saw_player : saw_opp) = true;
  }
  assert(saw_player && saw_opp);  // owner_to_move flips along real descents
}

void test_fused_pass_tokens() {
  // Lead with {3, 2}, opp holds 2 cards, all three aces discarded: after our
  // single '2' the opponent provably cannot respond (no higher single, no bomb
  // possible) -> the transition fuses the forced pass. The edge must be marked
  // fused and any deeper leaf's token stream must contain {single2, kPASS}.
  SearchState root{counts({{0, 1}, {12, 1}}), 2, counts({{11, 3}}), kPASS, kUs};
  StubEval ev;
  ev.capture = true;
  Search s(root, {}, {1.5f, 64});
  s.run(ev);
  const Node *r = s.root_node();
  const Edge *two = nullptr;
  for (const auto &e : r->edges)
    if (e.move_id == kSINGLE_START + 12 && e.child) two = &e;
  assert(two && two->fused_pass);
  bool saw_fused_seq = false;
  for (const auto &f : ev.captured) {
    const auto &t = f.path_tokens;
    for (std::size_t i = 0; i + 1 < t.size(); ++i)
      if (t[i] == kSINGLE_START + 12 && t[i + 1] == kPASS) saw_fused_seq = true;
  }
  assert(saw_fused_seq);
}

void test_play_mode_is_seed_independent() {
  // In play/eval mode (training=false) the opponent representative is the
  // max-probability member — no sampling — so the result must not depend on the
  // seed. (Training mode draws weighted-random and may differ across seeds.)
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}, {7, 1}}), 6, {}, kPASS, kUs};
  StubEval ev1, ev2;
  Search a(root, {}, {1.5f, 400, /*seed=*/1u, /*training=*/false});
  Search b(root, {}, {1.5f, 400, /*seed=*/9999u, /*training=*/false});
  a.run(ev1);
  b.run(ev2);
  assert(a.best_move() == b.best_move());
  assert(std::abs(a.root_value() - b.root_value()) < 1e-6);
}

// Build the real-history suffix for following `e` from the root: the move plus
// the fused forced pass when the transition fused one.
static std::vector<int> edge_suffix(const Edge &e) {
  std::vector<int> h{e.move_id};
  if (e.fused_pass) h.push_back(kPASS);
  return h;
}

void test_advance_root_reuse() {
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}}), 4, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {}, {1.5f, 300});
  s.run(ev);

  // Follow a genuinely-visited edge via its move suffix: the child's visit
  // count must carry over (subtree reuse) and the siblings must be freed.
  const Node *r = s.root_node();
  const Edge *kept = nullptr;
  for (const auto &e : r->edges)
    if (e.child && e.child->N > 0) { kept = &e; break; }
  assert(kept);
  const SearchState visited = kept->child->st;
  const long visited_n = kept->child->N;

  const std::size_t before = s.num_nodes();
  s.advance_root(visited, edge_suffix(*kept));
  assert(s.num_nodes() <= before);            // kept subtree reused; rest freed
  assert(s.root_n() == visited_n);            // prior visits preserved
  assert(s.history() == edge_suffix(*kept));

  // A history that diverges from every resolved edge discards the whole tree:
  // only a fresh, unvisited root remains.
  SearchState fresh = visited;
  fresh.opp_size = 1;
  std::vector<int> bogus = s.history();
  bogus.push_back(kSINGLE_START + 9);  // a move no tree edge matches
  s.advance_root(fresh, bogus);
  assert(s.root_n() == 0);
  assert(s.num_nodes() == 1);
}

void test_advance_root_gc() {
  // Two distinct singles at lead -> two top-level children whose subtrees are
  // mutually unreachable (a played single can't return to the hand). Re-rooting
  // onto one must free the other (and its exclusive nodes), preserve the kept
  // subtree's visits, recycle freed slots, and leave the search deterministic.
  SearchState root{counts({{0, 1}, {5, 1}}), 8, {}, kPASS, kUs};

  auto run_and_advance = [&](unsigned seed) {
    StubEval ev;
    Search s(root, {}, {1.5f, 300, seed, /*training=*/true});
    s.run(ev);
    const std::size_t before = s.num_nodes();
    const Node *r = s.root_node();
    const Edge *kept = nullptr;
    for (const auto &e : r->edges)
      if (e.child && e.child->N > 0) { kept = &e; break; }
    assert(kept);
    const Node *keep = kept->child;
    const long target_n = keep->N;

    s.advance_root(keep->st, edge_suffix(*kept));
    assert(s.root_node() == keep);             // re-rooted onto the existing node
    assert(s.root_n() == target_n);            // kept subtree's visits preserved
    assert(s.num_nodes() < before);            // sibling subtree freed
    assert(s.free_count() > 0);                // freed slots available for reuse
    assert(s.root_value() >= 0.0f && s.root_value() <= 1.0f);

    s.run(ev);                                 // follow-up search still runs
    return s.best_move();
  };

  assert(run_and_advance(123u) == run_and_advance(123u));  // GC preserves determinism
}

}  // namespace

void run_az_search_tests() {
  test_head_dims();
  test_opp_index_mapping();
  test_player_factored_head();
  test_collapsed_groups_structural();
  test_transition();
  test_forced_win();
  test_search_invariants_and_determinism();
  test_expectimax_backup();
  test_tree_node_bound();
  test_tablebase_oracle();
  test_forced_win_extension();
  test_forced_move_expansion();
  test_player_composed_prior();
  test_hierarchical_groups_partition();
  test_opp_node_grouping_wellformed();
  test_leaf_path_tokens();
  test_fused_pass_tokens();
  test_play_mode_is_seed_independent();
  test_advance_root_reuse();
  test_advance_root_gc();
}
