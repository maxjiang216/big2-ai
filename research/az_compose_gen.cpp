// Emits the az_search player-policy composition map as JSON: for each of the
// LEGAL_MOVES_SIZE engine move ids, the list of factored-head logit indices its
// composed logit sums (its path through family -> rank/high -> aux/low). This is
// the SINGLE SOURCE OF TRUTH (C++ considered_moves.h); nn/model_az.py loads the
// emitted file to build the 468 x AZ_PLAYER_HEAD_DIM binary composition matrix C,
// so the C++ search (player_composed_logit) and the Python training loss (head @
// C^T) cannot drift.
//
//   make az_compose_gen && ./bin/az_compose_gen > nn/az_compose.json

#include "az_search/considered_moves.h"
#include "util.h"

#include <cstdio>

int main() {
  std::printf("{\n  \"player_head_dim\": %d,\n  \"paths\": [\n",
              az_search::AZ_PLAYER_HEAD_DIM);
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    const az_search::PathLogits p = az_search::player_path_logits(id);
    std::printf("    [");
    for (int k = 0; k < p.n; ++k)
      std::printf("%s%d", k ? ", " : "", p.idx[k]);
    std::printf("]%s\n", id + 1 < LEGAL_MOVES_SIZE ? "," : "");
  }
  std::printf("  ]\n}\n");
  return 0;
}
