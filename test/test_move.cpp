#include "move.h"
#include "util.h"

#include <cassert>
#include <stdexcept>

// Encode -> decode roundtrip must be lossless for every valid move id.
static void test_encode_decode_roundtrip() {
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    Move m(id);
    int re = encodeMove(m);
    assert(re == id);
  }
}

// numCards() must equal the sum of per-rank counts in MOVE_TO_CARDS.
static void test_num_cards_matches_move_to_cards() {
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    Move m(id);
    int table_total = MOVE_TO_CARDS[id][13];
    int counted_total = m.numCards();
    assert(counted_total == table_total);
  }
}

// Card counts in MOVE_TO_CARDS must not exceed per-rank deck limits.
static void test_move_to_cards_limits() {
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    for (int rank = 0; rank < 13; ++rank) {
      int count = MOVE_TO_CARDS[id][rank];
      int limit = max_cards_in_deck_for_rank(rank);
      assert(count >= 0);
      assert(count <= limit);
    }
    // [13] must equal sum of [0..12]
    int sum = 0;
    for (int rank = 0; rank < 13; ++rank)
      sum += MOVE_TO_CARDS[id][rank];
    assert(sum == MOVE_TO_CARDS[id][13]);
  }
}

// A pass encodes to 0 and decodes symmetrically.
static void test_pass() {
  Move p(Move::Combination::kPass);
  assert(p.numCards() == 0);
  assert(encodeMove(p) == kPASS);
  Move back(kPASS);
  assert(back.combination == Move::Combination::kPass);
}

// Singles: rank 3 (index 0) .. rank A (index 10, actual rank 14).
// In the encoding rank 3 maps to kSINGLE_START = 1.
static void test_singles() {
  for (int rank = 3; rank <= 15; ++rank) {
    if (rank == 2) continue; // 2 = rank index 12, encoded as single 13
    Move s(Move::Combination::kSingle, rank);
    int id = encodeMove(s);
    Move back(id);
    assert(back.combination == Move::Combination::kSingle);
    assert(back.rank == rank);
    assert(back.numCards() == 1);
  }
}

// Full houses: triple rank != pair rank, check encode->decode for a sample.
static void test_full_house_sample() {
  // Triple of 5s (rank 5), pair of 7s (rank 7)
  Move fh(Move::Combination::kFullHouse, 5, 7);
  int id = encodeMove(fh);
  assert(id >= kFULL_HOUSE_START && id < kBOMB_START);
  Move back(id);
  assert(back.combination == Move::Combination::kFullHouse);
  assert(back.rank == 5);
  assert(back.auxiliary == 7);
  assert(back.numCards() == 5);

  // Also triple of 7s, pair of 5s (reversed)
  Move fh2(Move::Combination::kFullHouse, 7, 5);
  int id2 = encodeMove(fh2);
  assert(id2 != id);
  Move back2(id2);
  assert(back2.combination == Move::Combination::kFullHouse);
  assert(back2.rank == 7);
  assert(back2.auxiliary == 5);
}

// All move types produce ids within [0, LEGAL_MOVES_SIZE).
static void test_all_ids_in_range() {
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    Move m(id);
    int re = encodeMove(m);
    assert(re >= 0 && re < LEGAL_MOVES_SIZE);
  }
}

void run_move_tests() {
  test_pass();
  test_singles();
  test_full_house_sample();
  test_encode_decode_roundtrip();
  test_num_cards_matches_move_to_cards();
  test_move_to_cards_limits();
  test_all_ids_in_range();
}
