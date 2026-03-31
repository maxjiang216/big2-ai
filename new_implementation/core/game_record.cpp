#include "game_record.h"
#include "game.h"

GameRecord::GameRecord() {}

void GameRecord::set_initial_state(const Game &game) {
  _game = game;
  _views[0] = PartialGame(game, 0);
  _views[1] = PartialGame(game, 1);
}

void GameRecord::add_move(const Move &move) {
  TurnRecord new_record{
      _game.current_player(),
      _game,
      _views,
      _views[_game.current_player()].get_legal_moves(),
      _views[1 - _game.current_player()].get_possible_moves(),
      move};
  _turns.push_back(new_record);
  _game.apply_move(move);
  _views[0].apply_move(move);
  _views[1].apply_move(move);
}
