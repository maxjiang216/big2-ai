#ifndef AZ_PIMC_PIMC_PLAYER_H
#define AZ_PIMC_PIMC_PLAYER_H

// Determinization / Perfect-Information Monte Carlo player (root-only, no tree).
//
// At each decision: sample N opponent hands from the AR belief net (Big2NetII
// scripted sample_opp, mixed with a uniform γ-floor), then for every legal move
// score it by the Max-of-Average rule:
//
//     Q(a) = 1 - (1/N) Σ_i V_PI( opp=hand_i, us=our_hand-after-a, trick=a )
//
// where V_PI is the perfect-information series champion's value (P(side-to-move
// wins). We commit ONE move across all worlds and average — this respects the
// information-set constraint at the root (no strategy fusion at the decision)
// while leaning on the strong PI evaluator for a sharp, discriminating value.
// Pure root probe: if this clears parity vs the champion, it justifies folding
// determinized value into the MCTS leaf eval. Torch-dependent.

#include "az_pi/pi_eval.h"
#include "player.h"

#include <torch/script.h>

#include <memory>
#include <random>
#include <vector>

namespace az_pimc {

class AzPimcPlayer : public ::Player {
public:
  AzPimcPlayer(std::shared_ptr<torch::jit::Module> belief,
               std::shared_ptr<az_pi::PiEvaluator> pi, torch::Device device,
               int n_samples, float gamma, unsigned seed)
      : belief_(std::move(belief)), pi_(std::move(pi)), device_(device),
        n_(n_samples), gamma_(gamma), rng_(seed) {}

protected:
  void on_deal(const std::array<int, 13> & /*hand*/, int /*turn*/) override {
    history_.clear();
  }
  void on_opponent_move(const Move &move) override;
  void on_self_move(const Move &move) override;
  Move select_move_impl() override;

private:
  // Ancestral-sample N opponent hands (13-count) from the belief net, mixing in
  // a uniform γ-floor drawn from the unseen pool (thermo).
  std::vector<std::array<int, 13>> sample_hands(
      const std::array<int, 13> &thermo, int opp_size);
  std::array<int, 13> uniform_hand(const std::array<int, 13> &thermo,
                                   int opp_size);

  std::shared_ptr<torch::jit::Module> belief_;
  std::shared_ptr<az_pi::PiEvaluator> pi_;
  torch::Device device_;
  int n_;
  float gamma_;
  std::mt19937 rng_;
  std::vector<int> history_;  // every applied move since the deal (token prefix)
};

}  // namespace az_pimc

#endif  // AZ_PIMC_PIMC_PLAYER_H
