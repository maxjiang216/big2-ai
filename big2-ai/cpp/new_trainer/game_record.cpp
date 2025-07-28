// game_record.cpp
#include "game_record.h"
#include "game.h"
#include <arrow/api.h>
#include <sstream>

GameRecord::GameRecord() {}

void GameRecord::set_initial_state(const Game &game) {
  _game = game;
  _views[0] = PartialGame(game, 0);
  _views[1] = PartialGame(game, 1);
}

void GameRecord::add_move(const Move &move) {
  TurnRecord new_record{
      /* current_player  */ _game.current_player(),
      /* game           */ _game,
      /* views          */ _views,
      /* legal_moves    */ _views[_game.current_player()].get_legal_moves(),
      /* possible_moves */
      _views[1 - _game.current_player()]
          .get_possible_moves(),
      /* move           */ move};
  _turns.push_back(new_record);
}
