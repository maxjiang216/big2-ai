// Unit tests for the az_search move-set audit (considered_moves.h).
// Verifies the player/opponent head index mappings round-trip and that the two
// collapsed/dropped groups (TS5, DS8) satisfy the structural properties the
// audit relies on.

#include "az_search/az_search.h"
#include "az_search/considered_moves.h"
#include "az_search/eval_cache.h"
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
  // Identity range is [0, kTRIPLESTRAIGHT5_START); player adds the TS5 slot,
  // opponent adds TS5 + DS8 slots.
  assert(AZ_PLAYER_HEAD_DIM == kTRIPLESTRAIGHT5_START + 1);
  assert(AZ_OPP_HEAD_DIM == kTRIPLESTRAIGHT5_START + 2);
  assert(AZ_OPP_HEAD_DIM == AZ_PLAYER_HEAD_DIM + 1);
  // Sanity against the documented numbers.
  assert(AZ_PLAYER_HEAD_DIM == 457);
  assert(AZ_OPP_HEAD_DIM == 458);
}

void test_index_mapping_roundtrips() {
  int ts5_count = 0, ds8_count = 0;
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    const int pi = az_player_head_index(id);
    const int oi = az_opp_head_index(id);

    // Opponent index always valid; player index either valid or -1 (dropped).
    assert(oi >= 0 && oi < AZ_OPP_HEAD_DIM);
    assert(pi == -1 || (pi >= 0 && pi < AZ_PLAYER_HEAD_DIM));

    if (id < kTS5RangeBegin) {
      // Kept verbatim in both heads.
      assert(pi == id);
      assert(oi == id);
      assert(az_player_move_for_index(pi) == id);
      assert(az_opp_move_for_index(oi) == id);
    } else if (id < kTS5RangeEnd) {
      // TS5 collapses to one shared slot in both heads.
      ++ts5_count;
      assert(pi == kTS5Slot);
      assert(oi == kTS5Slot);
    } else {
      // DS8: dropped from the player head, single slot in the opponent head.
      ++ds8_count;
      assert(pi == -1);
      assert(oi == kOppDS8Slot);
    }
  }
  assert(ts5_count == 7);  // 7 TS5 ids collapse to 1
  assert(ds8_count == 5);  // 5 DS8 ids collapse to 1 (opp) / drop (player)

  // Reverse maps for the collapsed slots land on canonical engine ids.
  assert(az_player_move_for_index(kTS5Slot) == kTS5CanonicalMove);
  assert(az_opp_move_for_index(kTS5Slot) == kTS5CanonicalMove);
  assert(az_opp_move_for_index(kOppDS8Slot) == kDS8CanonicalMove);
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
// default. Counts calls so tests can confirm the NN is/ isn't queried.
struct StubEval : Evaluator {
  std::function<float(const PlayerFeatures &)> pv = [](const PlayerFeatures &) {
    return 0.5f;
  };
  std::function<float(const OppFeatures &)> ov = [](const OppFeatures &) {
    return 0.5f;
  };
  int player_calls = 0, opp_calls = 0;

  PlayerEval eval_player(const PlayerFeatures &f) override {
    ++player_calls;
    PlayerEval e;
    e.value = pv(f);
    e.logits.fill(0.0f);
    return e;
  }
  OppEval eval_opp(const OppFeatures &f) override {
    ++opp_calls;
    OppEval e;
    e.move_value.fill(ov(f));  // per-move q_a; uniform here unless a test varies it
    e.logits.fill(0.0f);
    return e;
  }
};

static std::array<int, 13> counts(std::initializer_list<std::pair<int, int>> kv) {
  std::array<int, 13> a{};
  for (auto [r, c] : kv) a[r] = c;
  return a;
}

void test_transition_and_key() {
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

  // Key excludes discard but distinguishes hand / opp_size / trick / side.
  SearchState a{counts({{0, 1}}), 5, {}, kPASS, kUs};
  SearchState b = a; b.discard = counts({{2, 2}});  // different discard only
  assert(state_key(a) == state_key(b));
  SearchState c = a; c.side = kOpp;
  assert(state_key(a) != state_key(c));
  SearchState d = a; d.opp_size = 6;
  assert(state_key(a) != state_key(d));
}

void test_forced_win() {
  // Lead position, our hand is a single card -> playing it empties our hand.
  SearchState root{counts({{12, 1}}), 8, {}, kPASS, kUs};  // single '2'
  Search search(root, {1.5f, 64});
  StubEval ev;  // value 0.5 — but a forced win should override to 1.0
  search.run(ev);
  assert(search.best_move() == kSINGLE_START + 12);  // the single '2'
  assert(std::abs(search.root_value() - 1.0f) < 1e-6);
}

void test_search_invariants_and_determinism() {
  // Response position: opp led a single '3'; we hold 4 and 5 (both beat it).
  SearchState root{counts({{1, 1}, {2, 1}}), 6, counts({{0, 1}}), kSINGLE_START, kUs};
  StubEval ev;
  Search s1(root, {1.5f, 300});
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
  Search s2(root, {1.5f, 300});
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
  ev.ov = [](const OppFeatures &f) { return f.opp_size >= 6 ? 0.3f : 0.7f; };
  Search s(root, {1.5f, 500});
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

void test_transposition_merge() {
  // Two different discards reaching the same (hand,opp_size,trick,side) share a
  // node: searching from such a position must not double-count. Here we just
  // assert the memo stays a DAG — node count <= a loose tree bound — and that a
  // crafted equal-key pair collapses (covered in test_transition_and_key).
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}}), 4, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {1.5f, 200});
  s.run(ev);
  // Each simulation creates at most one new node, so memo size <= sims + 1.
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
  Search s(root, {1.5f, 64});
  s.run(ev);
  assert(std::abs(s.root_value() - 1.0f) < 1e-6);
  assert(s.best_move() == kSINGLE_START + 12);  // play the 2 first
}

void test_hierarchical_groups_partition() {
  // Rich lead hand: singles, a pair, a triple -> several trick types -> groups.
  SearchState root{counts({{0, 1}, {1, 1}, {2, 2}, {5, 3}}), 7, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {1.5f, 400});
  s.run(ev);
  const Node *r = s.root_node();
  assert(!r->terminal && r->expanded);

  // Groups partition the edges exactly once each.
  std::size_t covered = 0;
  std::vector<int> seen_keys;
  for (const auto &g : r->groups) {
    covered += g.idx.size();
    float ps = 0.0f;
    for (int i : g.idx) ps += r->edges[i].prior;
    assert(std::abs(ps - g.prior_sum) < 1e-4);
    assert(std::find(seen_keys.begin(), seen_keys.end(), g.key) == seen_keys.end());
    seen_keys.push_back(g.key);
  }
  assert(covered == r->edges.size());
  assert(r->groups.size() >= 2);  // singles + double + triple at least
}

void test_opp_node_grouping_wellformed() {
  // After we play from a lead, the resulting opp node should be expanded with a
  // valid (collapsed) behavior distribution summing to ~1.
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}, {7, 1}}), 6, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {1.5f, 500});
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

void test_caching_evaluator() {
  StubEval base;
  base.pv = [](const PlayerFeatures &f) { return f.opp_size / 16.0f; };
  CachingEvaluator cache(base);

  PlayerFeatures f{counts({{0, 1}}), counts({{1, 2}}), {}, 8, 1};
  PlayerEval a = cache.eval_player(f);
  PlayerEval b = cache.eval_player(f);  // identical input -> cache hit
  assert(cache.player_misses() == 1 && base.player_calls == 1);
  assert(a.value == b.value);

  PlayerFeatures g = f; g.opp_size = 7;  // different input -> miss
  cache.eval_player(g);
  assert(cache.player_misses() == 2 && base.player_calls == 2);

  // Opponent evals are now keyed on the full input INCLUDING our hand (the opp
  // net takes it). Identical inputs hit; inputs differing only in our hand miss.
  OppFeatures o{counts({{0, 1}}), counts({{2, 2}}), counts({{3, 1}}), 5, 4};
  cache.eval_opp(o);
  cache.eval_opp(o);  // identical -> hit
  assert(cache.opp_misses() == 1 && base.opp_calls == 1);
  OppFeatures o2 = o; o2.hand = counts({{1, 1}});  // differs only in our hand -> miss
  cache.eval_opp(o2);
  assert(cache.opp_misses() == 2 && base.opp_calls == 2);
}

void test_play_mode_is_seed_independent() {
  // In play/eval mode (training=false) the opponent representative is the
  // max-probability member — no sampling — so the result must not depend on the
  // seed. (Training mode draws weighted-random and may differ across seeds.)
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}, {7, 1}}), 6, {}, kPASS, kUs};
  StubEval ev1, ev2;
  Search a(root, {1.5f, 400, /*seed=*/1u, /*training=*/false});
  Search b(root, {1.5f, 400, /*seed=*/9999u, /*training=*/false});
  a.run(ev1);
  b.run(ev2);
  assert(a.best_move() == b.best_move());
  assert(std::abs(a.root_value() - b.root_value()) < 1e-6);
}

void test_advance_root_reuse() {
  SearchState root{counts({{0, 1}, {1, 1}, {2, 1}}), 4, {}, kPASS, kUs};
  StubEval ev;
  Search s(root, {1.5f, 300});
  s.run(ev);

  // Grab a genuinely-visited child state and re-root onto it: its visit count
  // and the node count must carry over (subtree reuse, no new node).
  const Node *r = s.root_node();
  SearchState visited{};
  long visited_n = -1;
  for (const auto &e : r->edges)
    if (e.child && e.child->N > 0) { visited = e.child->st; visited_n = e.child->N; break; }
  assert(visited_n > 0);

  const std::size_t before = s.num_nodes();
  s.advance_root(visited);
  assert(s.num_nodes() <= before);            // kept subtree reused; rest GC'd
  assert(s.root_n() == visited_n);            // prior visits preserved

  // Re-rooting onto an unseen state allocates a fresh (unvisited) root; the old
  // subtree, now fully unreachable, is reclaimed (only the fresh root remains).
  SearchState fresh = visited;
  fresh.opp_size = 1;  // a key not in the memo
  s.advance_root(fresh);
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
    Search s(root, {1.5f, 300, seed, /*training=*/true});
    s.run(ev);
    const std::size_t before = s.num_nodes();
    const Node *r = s.root_node();
    SearchState target{};
    long target_n = -1;
    const Node *keep = nullptr;
    for (const auto &e : r->edges)
      if (e.child && e.child->N > 0) { target = e.child->st; target_n = e.child->N; keep = e.child; break; }
    assert(target_n > 0);

    s.advance_root(target);
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
  test_index_mapping_roundtrips();
  test_collapsed_groups_structural();
  test_transition_and_key();
  test_forced_win();
  test_search_invariants_and_determinism();
  test_expectimax_backup();
  test_transposition_merge();
  test_tablebase_oracle();
  test_forced_win_extension();
  test_hierarchical_groups_partition();
  test_opp_node_grouping_wellformed();
  test_caching_evaluator();
  test_play_mode_is_seed_independent();
  test_advance_root_reuse();
  test_advance_root_gc();
}
