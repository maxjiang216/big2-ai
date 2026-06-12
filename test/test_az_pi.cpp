// Unit tests for the perfect-information AlphaZero module (az_pi):
// the exact endgame solver (pi_solver) and the PUCT search core (pi_search).

#include "az_pi/pi_eval.h"
#include "az_pi/pi_prune.h"
#include "az_pi/pi_search.h"
#include "az_pi/pi_solver.h"
#include "az_search/considered_moves.h"
#include "game.h"
#include "move.h"
#include "util.h"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

using namespace az_pi;

// Build a 13-rank count array from (rank_index, count) pairs.
std::array<int, 13> hand(std::initializer_list<std::pair<int, int>> rc) {
  std::array<int, 13> h{};
  for (auto &p : rc) h[p.first] = p.second;
  return h;
}

const Move kLead{Move::Combination::kPass};

// ----------------------------- solver -----------------------------

void test_solver_immediate_win() {
  // Player 0 leads with a single card -> plays it, empties hand, wins.
  Game g(hand({{0, 1}}), hand({{1, 1}}), {}, kLead, 0);
  assert(solve(g, SolverLimits{}) == Proof::WIN);
}

void test_solver_forced_loss() {
  // Player 0 holds two unmatched singles (3,4); opponent holds a single 5.
  // Either lead is beaten by the 5, which empties the opponent's hand -> loss.
  Game g(hand({{0, 1}, {1, 1}}), hand({{2, 1}}), {}, kLead, 0);
  assert(solve(g, SolverLimits{}) == Proof::LOSS);
}

void test_solver_win_by_squeeze() {
  // Player 0 holds 3,4; opponent holds a lone 3. Lead the 4 (opp can't beat,
  // must pass), regain the lead, then play the 3 to empty out and win.
  Game g(hand({{0, 1}, {1, 1}}), hand({{0, 1}}), {}, kLead, 0);
  assert(solve(g, SolverLimits{}) == Proof::WIN);
}

void test_solver_card_threshold() {
  // Total cards (3) above the max_total_cards limit -> UNKNOWN, no search.
  Game g(hand({{0, 1}, {1, 1}}), hand({{2, 1}}), {}, kLead, 0);
  SolverLimits lim;
  lim.max_total_cards = 2;
  assert(solve(g, lim) == Proof::UNKNOWN);
}

void test_solver_budget_exhaustion() {
  // A position requiring multiple node expansions, but only one node of budget.
  solver_clear_memo();  // the memo persists across calls; isolate this test
  Game g(hand({{0, 1}, {1, 1}}), hand({{2, 1}}), {}, kLead, 0);
  SolverLimits lim;
  lim.node_budget = 1;
  assert(solve(g, lim) == Proof::UNKNOWN);
  // Persistence: a full-budget solve proves it, after which even a zero-budget
  // call answers from the memo.
  assert(solve(g, SolverLimits{}) == Proof::LOSS);
  lim.node_budget = 0;
  assert(solve(g, lim) == Proof::LOSS);
}

void test_solver_exact_margins() {
  // Margin minimax. P0 {3,3} vs opp {4}: P0 must lead a 3 (or the pair);
  // pair 33 empties the hand at once -> win margin = 1 (opp keeps its card).
  // Single 3 is beaten by the 4 (opp empties -> we lose) — so the pair is the
  // ONLY win and the margin is exactly 1.
  solver_clear_memo();
  {
    Game g(hand({{0, 2}}), hand({{1, 1}}), {}, kLead, 0);
    int m = -1;
    assert(solve(g, SolverLimits{}, &m) == Proof::WIN);
    assert(m == 1);
  }
  // P0 {3,4,5} vs opp {6}: every P0 lead is beaten by the 6 unless... the 6
  // beats any single; opp then empties -> P0 LOSES. Loser margin = P0's final
  // cards: P0 sheds exactly one card before opp finishes -> margin 2.
  {
    Game g(hand({{0, 1}, {1, 1}, {2, 1}}), hand({{3, 1}}), {}, kLead, 0);
    int m = -1;
    assert(solve(g, SolverLimits{}, &m) == Proof::LOSS);
    assert(m == 2);
  }
  // Win margin maximisation: P0 {7,8} vs opp {3,4}. P0 leads the 7 or 8, opp
  // cannot beat either (must pass), P0 plays the other and wins with opp still
  // holding BOTH cards -> margin exactly 2 (the ceiling).
  {
    Game g(hand({{4, 1}, {5, 1}}), hand({{0, 1}, {1, 1}}), {}, kLead, 0);
    int m = -1;
    assert(solve(g, SolverLimits{}, &m) == Proof::WIN);
    assert(m == 2);
  }
}

// ----------------------------- search -----------------------------

// Stub evaluator: uniform priors, fixed scalar value. Lets the search core be
// tested without a NN. `value` is P(mover wins) at every leaf.
struct StubEvaluator : PiEvaluator {
  float value;
  explicit StubEvaluator(float v) : value(v) {}
  std::vector<PiNetEval> eval_batch(
      const std::vector<PiEvalFeatures> &feats) override {
    std::vector<PiNetEval> out(feats.size());
    for (auto &e : out) {
      e.value = value;
      e.policy.fill(0.0f);  // uniform after softmax
    }
    return out;
  }
};

void test_search_visit_count() {
  // root.N equals the number of simulations after a run (no solver shortcut).
  Game g(hand({{0, 2}, {1, 2}, {2, 2}, {3, 2}}),
         hand({{4, 2}, {5, 2}, {6, 2}, {7, 2}}), {}, kLead, 0);
  PiSearchConfig cfg;
  cfg.sims = 64;
  cfg.solver.max_total_cards = 0;  // disable solver
  PiSearch s(g, cfg);
  StubEvaluator ev(0.5f);
  s.run(ev);
  assert(s.root_n() == cfg.sims);
}

void test_search_terminal_win_dominates() {
  // Player 0 can empty its hand in one move; search must prefer that move.
  Game g(hand({{0, 1}}), hand({{1, 1}, {2, 1}}), {}, kLead, 0);
  PiSearchConfig cfg;
  cfg.sims = 32;
  cfg.solver.max_total_cards = 0;  // force search (not solver) to find the win
  PiSearch s(g, cfg);
  StubEvaluator ev(0.5f);
  s.run(ev);
  const int best = s.best_move();
  Game check = g;
  check.apply_move(best);
  assert(check.is_over() && check.get_winner() == 0);
}

void test_search_mean_backup() {
  // Averaging (not max) backup: every leaf returns 0.5, perspective-flips to
  // 0.5, so a shallow search with no terminal reachable must average to exactly
  // 0.5. Scattered ranks (no consecutive run, no whole-hand move) prevent any
  // early hand-emptying move, and sims are kept below the earliest possible
  // terminal ply, so no exact 0/1 ever enters the average.
  Game g(hand({{0, 2}, {2, 2}, {4, 2}, {6, 2}}),
         hand({{1, 2}, {3, 2}, {5, 2}, {7, 2}}), {}, kLead, 0);
  PiSearchConfig cfg;
  cfg.sims = 6;
  cfg.solver.max_total_cards = 0;  // disable solver
  PiSearch s(g, cfg);
  StubEvaluator ev(0.5f);
  s.run(ev);
  const float v = s.root_value();
  assert(std::fabs(v - 0.5f) < 1e-3f);
}

void test_search_solver_finds_win() {
  // With the solver enabled, a winnable endgame root should resolve to a proven
  // win and pick a move that begins a winning line.
  Game g(hand({{0, 1}, {1, 1}}), hand({{0, 1}}), {}, kLead, 0);
  PiSearchConfig cfg;
  cfg.sims = 16;  // solver should prove it well within budget
  PiSearch s(g, cfg);
  StubEvaluator ev(0.5f);
  s.run(ev);
  assert(s.root_value() > 0.99f);
}

void test_search_advance_root_reuse() {
  // After advancing along a played move, the surviving subtree keeps visits.
  // Scattered ranks (no whole-hand move) so the best move is a searched edge
  // with a real visited child, not an eager-proven win with no child node.
  Game g(hand({{0, 2}, {2, 2}, {4, 2}, {6, 2}}),
         hand({{1, 2}, {3, 2}, {5, 2}, {7, 2}}), {}, kLead, 0);
  PiSearchConfig cfg;
  cfg.sims = 128;
  cfg.solver.max_total_cards = 0;
  PiSearch s(g, cfg);
  StubEvaluator ev(0.5f);
  s.run(ev);
  const int mv = s.best_move();
  s.advance_root(mv);
  // The new root inherits the played child's visit count (>0, < full budget).
  assert(s.root_n() > 0);
  assert(s.root_n() < cfg.sims);
}

void test_search_proven_root_best_move() {
  // Advancing into a solver-proven child leaves an UNEXPANDED proven root
  // (select_leaf never expands proven nodes). best_move() must still return a
  // legal — and when proven WIN, winning — move, not a bogus default.
  // P0 {3,4} vs opp {3}: P0 leads the 4 (proven win), opp must pass, then the
  // proven-WIN root must play the 3 to win, never pass.
  solver_clear_memo();
  Game g(hand({{0, 1}, {1, 1}}), hand({{0, 1}}), {}, kLead, 0);
  PiSearchConfig cfg;
  cfg.sims = 64;
  PiSearch s(g, cfg);
  StubEvaluator ev(0.5f);
  s.run(ev);
  const int lead = s.best_move();
  assert(lead != kPASS);  // both leads win (a tied single can't beat)
  s.advance_root(lead);
  // Opp node: proven LOSS, unexpanded; only legal reply is pass.
  const int reply = s.best_move();
  assert(reply == kPASS);
  s.advance_root(reply);
  // Back to P0: proven WIN, unexpanded. Must play the 3 (wins), not pass.
  const int finish = s.best_move();
  assert(finish != kPASS);
  Game end = g;
  end.apply_move(lead);
  end.apply_move(reply);
  end.apply_move(finish);
  assert(end.is_over() && end.get_winner() == 0);
}

void test_search_series_values() {
  // With a series table, proven roots back up series_value_after_win instead
  // of 1/0. Flat stub table: V = 0.7 everywhere.
  SeriesTable t;
  for (auto &row : t.v) row.fill(0.7f);
  t.loaded = true;

  solver_clear_memo();
  Game g(hand({{0, 1}, {1, 1}}), hand({{0, 1}}), {}, kLead, 0);  // proven win
  {
    PiSearchConfig cfg;
    cfg.sims = 16;
    cfg.series = &t;  // pts (0,0): the win never ends the series -> V = 0.7
    PiSearch s(g, cfg);
    StubEvaluator ev(0.5f);
    s.run(ev);
    assert(std::fabs(s.root_value() - 0.7f) < 1e-4f);
  }
  {
    PiSearchConfig cfg;
    cfg.sims = 16;
    cfg.series = &t;
    cfg.pts[0] = 49;  // any win ends the series -> exact 1.0
    PiSearch s(g, cfg);
    StubEvaluator ev(0.5f);
    s.run(ev);
    assert(s.root_value() > 0.999f);
  }
  {
    PiSearchConfig cfg;  // no table: classic 1/0 backup unchanged
    cfg.sims = 16;
    PiSearch s(g, cfg);
    StubEvaluator ev(0.5f);
    s.run(ev);
    assert(s.root_value() > 0.99f);
  }
}

void test_prune_dominated_attachments() {
  // Hand: bomb 5555 + loose singles 8 and K (no straight possible), plus a
  // loose pair of 9s and triple of 3s (full house bases).
  // Bomb kicker: keep 8, drop K. FH 333+99 has no competing loose pair here.
  auto count_att = [](const std::vector<int> &legal, Move::Combination comb,
                      int att_rank) {
    int n = 0;
    for (int mid : legal) {
      if (Move(mid).combination != comb) continue;
      if (MOVE_TO_CARDS[mid][att_rank] == 0) continue;
      ++n;
    }
    return n;
  };

  {
    Game g(hand({{2, 4}, {5, 1}, {10, 1}, {0, 3}, {6, 2}}), hand({{1, 1}}),
           {}, kLead, 0);
    auto legal = g.get_legal_moves();
    auto pruned = legal;
    prune_dominated_attachments(g.player_hand(0), pruned);
    // 8 (rank 5) is the lower loose single: bomb+8 kept, bomb+K (rank 10) gone.
    assert(count_att(pruned, Move::Combination::kBomb, 5) == 1);
    assert(count_att(pruned, Move::Combination::kBomb, 10) == 0);
    assert(count_att(legal, Move::Combination::kBomb, 10) == 1);
    // Non-attachment moves untouched: both FH bases (333+99, 555+99) keep
    // their only loose pair.
    assert(count_att(pruned, Move::Combination::kFullHouse, 6) == 2);
    assert(pruned.size() == legal.size() - 1);
  }
  {
    // Kicker inside a fully-held straight window is NOT loose: hand holds
    // 3,4,5,6,7 singles + bomb 9999 + lone K. The 3..7 cards can form a
    // straight, so only the K is loose — no competing loose single, nothing
    // to prune among bomb kickers of differing loose ranks.
    Game g(hand({{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {6, 4}, {10, 1}}),
           hand({{1, 1}}), {}, kLead, 0);
    auto legal = g.get_legal_moves();
    auto pruned = legal;
    prune_dominated_attachments(g.player_hand(0), pruned);
    // All straight-member kickers kept; K kept (sole loose single).
    assert(pruned.size() == legal.size());
  }
}

void test_search_full_game_legal() {
  // A full stub self-play game terminates and only ever plays legal moves.
  Game g;
  std::mt19937 rng(123);
  g.shuffle_deal(rng);
  StubEvaluator ev(0.5f);
  int guard = 0;
  while (!g.is_over()) {
    PiSearchConfig cfg;
    cfg.sims = 16;
    cfg.seed = guard;
    PiSearch s(g, cfg);
    s.run(ev);
    const int mv = s.best_move();
    const auto legal = g.get_legal_moves();
    bool ok = false;
    for (int l : legal)
      if (l == mv) ok = true;
    assert(ok);
    g.apply_move(mv);
    assert(++guard < 200);
  }
}

}  // namespace

void run_az_pi_tests() {
  test_solver_immediate_win();
  test_solver_forced_loss();
  test_solver_win_by_squeeze();
  test_solver_card_threshold();
  test_solver_budget_exhaustion();
  test_solver_exact_margins();
  test_search_visit_count();
  test_search_terminal_win_dominates();
  test_search_mean_backup();
  test_search_solver_finds_win();
  test_search_advance_root_reuse();
  test_search_proven_root_best_move();
  test_search_series_values();
  test_prune_dominated_attachments();
  test_search_full_game_legal();
}
