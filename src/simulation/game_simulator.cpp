#include "game_simulator.h"

#include "game.h"
#include "player.h"

#include <iostream>
#include <sstream>

GameSimulator::GameSimulator(std::unique_ptr<Player> player0,
                             std::unique_ptr<Player> player1, std::mt19937 &rng,
                             const std::string &log_file)
    : _seed(static_cast<int>(rng() % 1000000000)),
      _player0(std::move(player0)), _player1(std::move(player1)), _rng(rng),
      _game(), log_file_(log_file), log_enabled_(false) {
  if (!log_file.empty()) {
    std::ostringstream oss;
    oss << log_file << "_" << _seed << ".txt";
    std::string log_filename = oss.str();

    log_stream_.open(log_filename, std::ios::out);
    log_enabled_ = log_stream_.is_open();
    if (!log_enabled_) {
      std::cerr << "Warning: Could not open log file " << log_filename << "\n";
    }
  }
}

GameRecord GameSimulator::run() {
  initialize_game();
  _record.set_initial_state(_game);

  if (log_enabled_) {
    log_stream_ << "Initial state:\n" << _game << std::endl;
  }

  play_loop();

  return _record;
}

void GameSimulator::initialize_game() {
  _game.shuffle_deal(_rng);

  _player0->accept_deal(_game, 0);
  _player1->accept_deal(_game, 1);
}

void GameSimulator::play_loop() {
  while (!_game.is_over()) {
    play_turn();
  }
}

void GameSimulator::play_turn() {
  int current = _game.current_player();
  Player *curr_player = (current == 0 ? _player0.get() : _player1.get());
  Player *other_player = (current == 0 ? _player1.get() : _player0.get());

  Move move = curr_player->select_move();

  _record.add_move(move, curr_player->last_tb_case(),
                   curr_player->last_tb_forced_seq_len(),
                   curr_player->last_tb_opp1_table_straight());
  _game.apply_move(move);
  other_player->accept_opponent_move(move);

  if (log_enabled_) {
    log_stream_ << "Game state:\n" << _game << std::endl;
  }
}
