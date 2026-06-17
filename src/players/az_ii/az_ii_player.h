#ifndef AZ_II_AZ_II_PLAYER_H
#define AZ_II_AZ_II_PLAYER_H

// Policy-greedy player for the imperfect-info distillation net (Big2NetII).
// Runs the belief transformer over the full public move history each turn and
// picks the legal move with the highest composed player-policy logit — no
// search. The net is distilled from the PI series champion, so greedy policy
// already plays strongly; an MCTS wrapper is a later phase. Torch-dependent.
//
// The base Player feeds every applied move (ours + opponent's, incl. tablebase
// root-skips) through on_*_move, so history_ is the seq net's token prefix.

#include "player.h"

#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

namespace az_ii {

class AzIiPlayer : public ::Player {
public:
  AzIiPlayer(std::shared_ptr<torch::jit::Module> model, torch::Device device)
      : model_(std::move(model)), device_(device) {}

protected:
  void on_deal(const std::array<int, 13> & /*hand*/, int /*turn*/) override {
    history_.clear();
  }
  void on_opponent_move(const Move &move) override;
  void on_self_move(const Move &move) override;
  Move select_move_impl() override;

private:
  std::shared_ptr<torch::jit::Module> model_;
  torch::Device device_;
  std::vector<int> history_;  // every applied move since the deal (token prefix)
};

}  // namespace az_ii

#endif  // AZ_II_AZ_II_PLAYER_H
