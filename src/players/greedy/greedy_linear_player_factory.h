#ifndef GREEDY_LINEAR_PLAYER_FACTORY_H
#define GREEDY_LINEAR_PLAYER_FACTORY_H

#include "greedy_linear_player.h"
#include "player_factory.h"

#include <memory>
#include <string>

class GreedyLinearPlayerFactory : public PlayerFactory {
public:
  explicit GreedyLinearPlayerFactory(const std::string &model_path = "data/linear_rollout_w.txt")
      : model_path_(model_path) {}

  std::unique_ptr<Player> create_player() override {
    return std::make_unique<GreedyLinearPlayer>(model_path_);
  }

private:
  std::string model_path_;
};

#endif
