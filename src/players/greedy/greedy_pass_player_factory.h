#ifndef GREEDY_PASS_PLAYER_FACTORY_H
#define GREEDY_PASS_PLAYER_FACTORY_H

#include "greedy_pass_player.h"
#include "player_factory.h"

#include <memory>
#include <string>

class GreedyPassPlayerFactory : public PlayerFactory {
public:
  explicit GreedyPassPlayerFactory(const std::string &model_path = "data/pass_ridge_w.txt")
      : path_(model_path) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyPassPlayer>(path_);
  }

private:
  std::string path_;
};

#endif
