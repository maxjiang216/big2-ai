#ifndef GAME_SIMULATOR_H
#define GAME_SIMULATOR_H

#include "game_record.h"
#include "player.h"

#include <fstream>
#include <memory>
#include <random>
#include <string>

class GameSimulator {
public:
  GameSimulator(std::unique_ptr<Player> player0,
                std::unique_ptr<Player> player1, std::mt19937 &rng,
                const std::string &log_path = "");

  GameRecord run();

private:
  int _seed;
  std::unique_ptr<Player> _player0;
  std::unique_ptr<Player> _player1;
  std::mt19937 &_rng;
  Game _game;
  GameRecord _record;

  std::string log_file_;
  std::ofstream log_stream_;
  bool log_enabled_ = false;

  void initialize_game();
  void play_loop();
  void play_turn();
};

#endif
