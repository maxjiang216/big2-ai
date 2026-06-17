// Emits the per-move card counts as JSON: for each of the LEGAL_MOVES_SIZE
// engine move ids, the 13 rank counts of the cards the move plays (pass = all
// zeros). This is the SINGLE SOURCE OF TRUTH (C++ MOVE_TO_CARDS) for the
// history-transformer token features; nn/model_az_seq.py loads the emitted
// file and applies the exact-count 48-dim encoding, so C++ move ids and Python
// token features cannot drift.
//
//   make az_tokens_gen && ./bin/az_tokens_gen > nn/az_token_cards.json

#include "util.h"

#include <cstdio>

int main() {
  std::printf("{\n  \"num_moves\": %d,\n  \"cards\": [\n", LEGAL_MOVES_SIZE);
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    std::printf("    [");
    for (int r = 0; r < 13; ++r)
      std::printf("%s%d", r ? ", " : "", MOVE_TO_CARDS[id][r]);
    std::printf("]%s\n", id + 1 < LEGAL_MOVES_SIZE ? "," : "");
  }
  std::printf("  ]\n}\n");
  return 0;
}
