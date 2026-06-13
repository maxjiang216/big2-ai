#ifndef AZ_II_AZ_II_PLAYER_FACTORY_H
#define AZ_II_AZ_II_PLAYER_FACTORY_H

// Factory for AzIiPlayer. Loads the scripted Big2NetII once and shares it across
// all players it creates. Torch-dependent (-DBIG2_WITH_TORCH).

#include "player_factory.h"

#include "az_ii_player.h"

#include <memory>
#include <string>

namespace az_ii {

class AzIiPlayerFactory : public ::PlayerFactory {
public:
  AzIiPlayerFactory(const std::string &model_path = "models/az_ii.ts.pt",
                    torch::Device device = torch::Device(torch::kCPU))
      : model_(std::make_shared<torch::jit::Module>(
            torch::jit::load(model_path, device))),
        device_(device) {
    model_->eval();
  }

  std::unique_ptr<::Player> create_player() override {
    return std::make_unique<AzIiPlayer>(model_, device_);
  }

private:
  std::shared_ptr<torch::jit::Module> model_;
  torch::Device device_;
};

}  // namespace az_ii

#endif  // AZ_II_AZ_II_PLAYER_FACTORY_H
