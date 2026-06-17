// Emits the FULL per-move static table for the web JS engine as JSON. This is
// the single source of truth (C++ move.cpp / util.cpp / considered_moves.h) so
// the browser MCTS cannot drift from the C++ az_search:
//
//   - combo / rank / aux       (Move decode)
//   - cards[13] + total        (MOVE_TO_CARDS)
//   - beating[]                (get_beating_moves: who beats this move)
//   - oppHead                  (az_opp_head_index: TS5/DS8 collapse)
//   - family / subKey          (player node grouping)
//   - path                     (138-factored composition; mirrors az_compose.json)
//
//   make az_moves_gen && ./bin/az_moves_gen > web/az_moves.json
//
// The JS loads this once; legality/possible-move/grouping/composition all reduce
// to table lookups (see web/engine.js).

#include "az_search/considered_moves.h"
#include "move.h"
#include "util.h"

#include <cstdio>

int main() {
  const auto &moves = all_moves();
  const auto &beating = get_beating_moves();

  std::printf("{\n");
  std::printf("  \"num_moves\": %d,\n", LEGAL_MOVES_SIZE);
  std::printf("  \"kPASS\": %d,\n", kPASS);
  std::printf("  \"opp_head_dim\": %d,\n", az_search::AZ_OPP_HEAD_DIM);
  std::printf("  \"player_head_dim\": %d,\n", az_search::AZ_PLAYER_HEAD_DIM);

  std::printf("  \"max_deck\": [");
  for (int r = 0; r < 13; ++r)
    std::printf("%s%d", r ? ", " : "", max_cards_in_deck_for_rank(r));
  std::printf("],\n");

  std::printf("  \"moves\": [\n");
  for (int id = 0; id < LEGAL_MOVES_SIZE; ++id) {
    const Move &m = moves[id];
    const int total = MOVE_TO_CARDS[id][13];
    const int fam = az_search::player_family_id(id);
    const int sub = az_search::player_subgroup_key(id);
    const int opp = az_search::az_opp_head_index(id);
    const az_search::PathLogits p = az_search::player_path_logits(id);

    std::printf("    {\"id\": %d, \"combo\": %d, \"rank\": %d, \"aux\": %d, "
                "\"total\": %d, \"oppHead\": %d, \"family\": %d, \"subKey\": %d, ",
                id, static_cast<int>(m.combination), m.rank, m.auxiliary,
                total, opp, fam, sub);

    std::printf("\"cards\": [");
    for (int r = 0; r < 13; ++r)
      std::printf("%s%d", r ? "," : "", MOVE_TO_CARDS[id][r]);
    std::printf("], ");

    std::printf("\"path\": [");
    for (int k = 0; k < p.n; ++k) std::printf("%s%d", k ? "," : "", p.idx[k]);
    std::printf("], ");

    std::printf("\"beating\": [");
    for (std::size_t k = 0; k < beating[id].size(); ++k)
      std::printf("%s%d", k ? "," : "", beating[id][k]);
    std::printf("]}%s\n", id + 1 < LEGAL_MOVES_SIZE ? "," : "");
  }
  std::printf("  ]\n}\n");
  return 0;
}
