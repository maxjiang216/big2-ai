#include "../src/core/move.h"
#include "../src/core/series.h"
#include "../src/core/util.h"

#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include <sstream>
#include <vector>

namespace {

void test_points_for_cards() {
  for (int c = 1; c <= 12; ++c)
    assert(points_for_cards(c) == c);
  assert(points_for_cards(13) == 20);
  assert(points_for_cards(14) == 30);
  assert(points_for_cards(15) == 40);
  assert(points_for_cards(16) == 50);
  assert(points_for_cards(0) == 0);
}

SeriesTable make_table() {
  SeriesTable t;
  for (int a = 0; a < kSeriesTarget; ++a)
    for (int b = 0; b < kSeriesTarget; ++b)
      t.v[a][b] = 0.25f * a - 0.1f * b + 0.5f; // arbitrary, just for lookup checks
  t.loaded = true;
  return t;
}

void test_series_value_after_win() {
  SeriesTable t = make_table();
  // Non-ending win: winner had 10, loser had 5, loser left 4 cards (p=4).
  // winner total 14 < 50 -> v[14][5].
  float exp = t.v[14][5];
  assert(std::fabs(series_value_after_win(t, 10, 5, 4) - exp) < 1e-6);
  // Big win replacement: loser left 16 cards -> p=50 -> winner 40+50 >=50 -> 1.0.
  assert(series_value_after_win(t, 40, 5, 16) == 1.0f);
  // Exactly reaching 50 ends the series.
  assert(series_value_after_win(t, 45, 0, 5) == 1.0f);
  // Just short stays in the table: 44 + 5 = 49 -> v[49][0].
  assert(std::fabs(series_value_after_win(t, 44, 0, 5) - t.v[49][0]) < 1e-6);
}

void test_csv_roundtrip() {
  std::string csv =
      "# gen=test\n"
      "a,b,v,natural_freq\n"
      "0,0,0.5132,1.0\n"
      "14,5,0.3300,0.0231\n"
      "49,0,0.9900,0.0\n";
  std::stringstream ss(csv);
  SeriesTable t;
  assert(load_series_table(ss, t));
  assert(t.loaded);
  assert(std::fabs(t.v[0][0] - 0.5132f) < 1e-5);
  assert(std::fabs(t.v[14][5] - 0.33f) < 1e-5);
  assert(std::fabs(t.natural[0][0] - 1.0f) < 1e-5);
  assert(std::fabs(t.natural[14][5] - 0.0231f) < 1e-5);
  assert(std::fabs(t.v[49][0] - 0.99f) < 1e-5);
}

void test_first_player_deterministic() {
  std::mt19937 rng(123);
  std::array<int, 13> h0{}, h1{};
  // All four 3s in hand0 -> holds 3 of spades for certain.
  h0[0] = 4;
  for (int i = 0; i < 50; ++i)
    assert(sample_first_player_3s(h0, h1, rng) == 0);
  // All four 3s in hand1.
  std::array<int, 13> g0{}, g1{};
  g1[0] = 4;
  for (int i = 0; i < 50; ++i)
    assert(sample_first_player_3s(g0, g1, rng) == 1);
}

void test_first_player_statistical() {
  std::mt19937 rng(7);
  std::array<int, 13> h0{}, h1{};
  h0[0] = 2;
  h1[0] = 2; // rank 3 split 2/2, no unused copies -> spade pass resolves at r=0
  int count0 = 0;
  const int N = 20000;
  for (int i = 0; i < N; ++i)
    if (sample_first_player_3s(h0, h1, rng) == 0)
      ++count0;
  double frac = static_cast<double>(count0) / N;
  assert(std::fabs(frac - 0.5) < 0.03);
}

// Decrement `hand` by the cards used in encoded move `id`.
void apply_to_hand(std::array<int, 13> &hand, int id) {
  Move m(id);
  using C = Move::Combination;
  auto rm = [&](int rank, int n) { hand[rank - 3] -= n; };
  switch (m.combination) {
  case C::kSingle: rm(m.rank, 1); break;
  case C::kDouble: rm(m.rank, 2); break;
  case C::kTriple: rm(m.rank, 3); break;
  case C::kBomb:
    rm(m.rank, m.rank == 14 ? 3 : 4);
    if (m.auxiliary != 0) rm(m.auxiliary, 1);
    break;
  default: assert(false && "unexpected combo from opp1_series_move");
  }
}

int total_cards(const std::array<int, 13> &h) {
  int s = 0;
  for (int c : h) s += c;
  return s;
}

void test_opp1_sheds_everything() {
  // Hand: a four-of-a-kind (5s), a triple (7s), a pair (9s), two loose singles
  // (4 and a King). No straights possible. Oracle must empty it.
  std::array<int, 13> hand{};
  hand[1] = 1;  // 4 (loose single, smallest -> bomb aux)
  hand[2] = 4;  // 5s bomb
  hand[4] = 3;  // 7s triple
  hand[6] = 2;  // 9s pair
  hand[10] = 1; // K (loose single)
  assert(!hand_has_straight_lead(hand));

  // First move: bomb with the smallest loose single (the 4) as auxiliary.
  int first = opp1_series_move(hand);
  Move fm(first);
  assert(fm.combination == Move::Combination::kBomb);
  assert(fm.rank == 5);
  assert(fm.auxiliary == 4);

  std::vector<int> seq;
  int guard = 0;
  while (total_cards(hand) > 0) {
    int id = opp1_series_move(hand);
    seq.push_back(id);
    apply_to_hand(hand, id);
    assert(++guard < 50);
  }
  assert(total_cards(hand) == 0);

  // Last emitted moves are singles, highest rank first. The only remaining
  // loose single after the bomb consumed the 4 is the King -> single K.
  Move last(seq.back());
  assert(last.combination == Move::Combination::kSingle);
  assert(last.rank == 13); // King
}

void test_opp1_singles_top_down() {
  // All singles (no combos, no straights): expect highest-first ordering.
  std::array<int, 13> hand{};
  hand[0] = 1;  // 3
  hand[5] = 1;  // 8
  hand[12] = 1; // 2 (highest rank)
  assert(!hand_has_straight_lead(hand));
  int id = opp1_series_move(hand);
  Move m(id);
  assert(m.combination == Move::Combination::kSingle);
  assert(m.rank == 15); // the 2 is the highest single
}

void test_has_straight_lead() {
  std::array<int, 13> straight{};
  for (int r = 0; r < 5; ++r) straight[r] = 1; // 3-4-5-6-7
  assert(hand_has_straight_lead(straight));
  std::array<int, 13> nostraight{};
  nostraight[0] = 2;
  nostraight[2] = 2;
  assert(!hand_has_straight_lead(nostraight));
}

} // namespace

void run_series_tests() {
  test_points_for_cards();
  test_series_value_after_win();
  test_csv_roundtrip();
  test_first_player_deterministic();
  test_first_player_statistical();
  test_opp1_sheds_everything();
  test_opp1_singles_top_down();
  test_has_straight_lead();
}
