// game_simulator.cpp
#include "game_simulator.h"
#include "game.h"
#include "player.h"
#include <iostream>
using namespace std;

GameSimulator::GameSimulator(
    std::unique_ptr<Player> player0,
    std::unique_ptr<Player> player1,
    std::mt19937 &rng,
    const std::string& log_file)
    : _seed(rng() % 1000000000),
     _player0(std::move(player0)),
      _player1(std::move(player1)),
      _rng(rng),
      _game(),
      log_file_(log_file),
      log_enabled_(false)
{
    if (!log_file.empty()) {
        cout << "Started game " << _seed << endl;
        // Construct filename: log_file + "_" + seed + ".txt"
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
  // Initialize game state and inform players
  initialize_game();

  // Record initial deal
  _record.set_initial_state(_game);

  if (log_enabled_) {
      log_stream_ << "Initial state:\n" << _game << std::endl;
  }

  // Main play loop
  play_loop();

  return _record;
}

void GameSimulator::initialize_game() {
  // Shuffle and deal
  _game.shuffle_deal(_rng);

  // Inform players of new game and their hands
  auto hand0 = _game.player_hand(0);
  auto hand1 = _game.player_hand(1);
  _player0->accept_deal(hand0, 0);
  _player1->accept_deal(hand1, 1);
}

void GameSimulator::play_loop() {
  // Continue until someone wins
  while (!_game.is_over()) {
    play_turn();
  }
}

void GameSimulator::play_turn() {
    int current = _game.current_player();
    Player *curr_player = (current == 0 ? _player0.get() : _player1.get());
    Player *other_player = (current == 0 ? _player1.get() : _player0.get());

    auto move = curr_player->select_move();

    _game.apply_move(move);
    _record.add_move(move);
    other_player->accept_opponent_move(move);

    if (log_enabled_) {
        log_stream_ << "Game state:\n" << _game << std::endl;
    }
}