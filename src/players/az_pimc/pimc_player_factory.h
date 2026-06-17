#ifndef AZ_PIMC_PIMC_PLAYER_FACTORY_H
#define AZ_PIMC_PIMC_PLAYER_FACTORY_H

// Factory for AzPimcPlayer. Loads the scripted belief net (Big2NetII with the
// sample_opp surface) and the PI series champion (via PiNNEvaluator) ONCE and
// shares both across all players it creates. Torch-dependent.

#include "az_pi/pi_nn_eval.h"
#include "player_factory.h"

#include "pimc_player.h"

#include <torch/cuda.h>

#include <cstdlib>
#include <memory>
#include <string>

namespace az_pimc {

// Env overrides (probe convenience): PIMC_PI_MODEL=<path> swaps the PI evaluator
// net; PIMC_PI_PTS=0 selects the game-value (size-input) net, =1 the series net.
inline std::string pimc_pi_model(const std::string &def) {
  const char *e = std::getenv("PIMC_PI_MODEL");
  return e ? std::string(e) : def;
}
inline bool pimc_pi_pts() {
  const char *e = std::getenv("PIMC_PI_PTS");
  return e ? std::string(e) == "1" : false;  // default game-value evaluator
}

class AzPimcPlayerFactory : public ::PlayerFactory {
public:
  AzPimcPlayerFactory(
      int n_samples = 16, float gamma = 0.1f, unsigned seed = 0,
      const std::string &belief_path = "models/az_ii_belief.ts.pt",
      const std::string &pi_path = "models/az_pi.pt",
      torch::Device device = torch::cuda::is_available()
                                 ? torch::Device(torch::kCUDA)
                                 : torch::Device(torch::kCPU))
      : belief_(std::make_shared<torch::jit::Module>(
            torch::jit::load(belief_path, device))),
        pi_(std::make_shared<az_pi::PiNNEvaluator>(
            pimc_pi_model(pi_path), device, /*pts_inputs=*/pimc_pi_pts())),
        device_(device), n_(n_samples), gamma_(gamma), seed_(seed) {
    belief_->eval();
  }

  std::unique_ptr<::Player> create_player() override {
    return std::make_unique<AzPimcPlayer>(belief_, pi_, device_, n_, gamma_,
                                          seed_++);
  }

private:
  std::shared_ptr<torch::jit::Module> belief_;
  std::shared_ptr<az_pi::PiNNEvaluator> pi_;
  torch::Device device_;
  int n_;
  float gamma_;
  unsigned seed_;
};

}  // namespace az_pimc

#endif  // AZ_PIMC_PIMC_PLAYER_FACTORY_H
