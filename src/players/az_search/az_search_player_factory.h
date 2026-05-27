#ifndef AZ_SEARCH_AZ_SEARCH_PLAYER_FACTORY_H
#define AZ_SEARCH_AZ_SEARCH_PLAYER_FACTORY_H

// Factory for AzSearchPlayer. Loads both TorchScript nets once and shares the
// evaluator across all players it creates (shared_ptr<NNEvaluator>); each player
// gets a distinct search seed. Defaults to play/eval mode (deterministic max
// opponent representative) — self-play data generation flips `training` on.
// Torch-dependent (-DBIG2_WITH_TORCH).

#include "player_factory.h"

#include "az_search.h"
#include "az_search_player.h"
#include "nn_eval.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace az_search {

class AzSearchPlayerFactory : public ::PlayerFactory {
public:
  AzSearchPlayerFactory(int sims, std::uint64_t base_seed,
                        const std::string &player_path = "models/az_player.pt",
                        const std::string &opp_path = "models/az_opp.pt",
                        bool training = false,
                        torch::Device device = torch::Device(torch::kCPU))
      : base_seed_(base_seed),
        nn_(std::make_shared<NNEvaluator>(player_path, opp_path, device)) {
    cfg_.sims = sims;
    cfg_.training = training;
  }

  std::unique_ptr<::Player> create_player() override {
    SearchConfig c = cfg_;
    c.seed = static_cast<unsigned>(base_seed_ ^
                                   (0x9E3779B97F4A7C15ull * (++counter_)));
    return std::make_unique<AzSearchPlayer>(nn_, c);
  }

private:
  std::uint64_t base_seed_;
  std::shared_ptr<NNEvaluator> nn_;
  SearchConfig cfg_;
  std::atomic<std::uint64_t> counter_{0};
};

}  // namespace az_search

#endif  // AZ_SEARCH_AZ_SEARCH_PLAYER_FACTORY_H
