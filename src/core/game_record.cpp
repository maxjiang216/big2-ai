#include "game_record.h"
#include "game.h"

TablebaseFirstHitStats tablebase_first_hit_stats(const GameRecord &record) {
  TablebaseFirstHitStats s{};
  int last_tb[2] = {-1, -1};
  for (const auto &turn : record.turns()) {
    int p = turn.current_player;
    if (turn.tb_case == 1) {
      if (last_tb[p] != 1) {
        ++s.case1;
        s.case1_seq_sum += turn.tb_forced_seq_len;
        if (turn.tb_forced_seq_len > s.case1_seq_max)
          s.case1_seq_max = turn.tb_forced_seq_len;
      }
      last_tb[p] = 1;
    } else if (turn.tb_case == 2) {
      if (last_tb[p] != 2) {
        ++s.case2;
        if (turn.tb_opp1_table_straight)
          ++s.case2_table_straight;
      }
      last_tb[p] = 2;
    } else {
      last_tb[p] = -1;
    }
  }
  return s;
}

GameRecord::GameRecord() {}

void GameRecord::set_initial_state(const Game &game) {
  _game = game;
  _views[0] = PartialGame(game, 0);
  _views[1] = PartialGame(game, 1);
}

void GameRecord::add_move(const Move &move, int tb_case, int tb_forced_seq_len,
                          bool tb_opp1_table_straight) {
  TurnRecord new_record{
      _game.current_player(),
      _game,
      _views,
      _views[_game.current_player()].get_legal_moves(),
      _views[1 - _game.current_player()].get_possible_moves(),
      move,
      tb_case,
      tb_forced_seq_len,
      tb_opp1_table_straight};
  _turns.push_back(new_record);
  _game.apply_move(move);
  _views[0].apply_move(move);
  _views[1].apply_move(move);
}
