#ifndef PLAYER_FACTORY_REGISTRY_H
#define PLAYER_FACTORY_REGISTRY_H

#include "player_factory.h"
#include "greedy/greedy_player_factory.h"
#include "greedy/greedy_random_player_factory.h"
#include "greedy/greedy_random_pass_player_factory.h"
#include "greedy/greedy_no_bomb_player_factory.h"
#include "greedy/greedy_linear_player_factory.h"
#include "greedy/greedy_pass_player_factory.h"
#include "greedy/tree_greedy_player_factory.h"
#include "pimc/pimc_adaptive_player_factory.h"
#include "pimc/pimc_player_factory.h"
#include "pimc/pimc_tree_adaptive_player_factory.h"
#include "pimc/pimc_tree_player_factory.h"
#include "pimc/pimc_linear_player_factory.h"
#include "pimc/pimc_linear_redet_player_factory.h"
#include "pimc/pimc_pass_rollout_player_factory.h"
#include "pimc/pimc_redet_player_factory.h"
#include "pimc/pimc_tree_redet_player_factory.h"
#include "random/random_player_factory.h"
#include "typed_search/typed_search_player_factory.h"

// az_search pulls in LibTorch; only available in torch-enabled builds so that
// non-torch binaries (test_core, eval_match, ...) that include this header stay
// torch-free.
#ifdef BIG2_WITH_TORCH
#include "az_search/az_search_player_factory.h"
#endif

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

// Creates a PlayerFactory by name. `param` is used by parameterized variants;
// ignored for "random" and "greedy". `seed` is the RNG seed for stochastic players.
//
// Supported names:
//   "random"             — uniform random legal move
//   "greedy"             — greedy (no randomness)
//   "greedy_random"      — greedy with prob param of playing a random move
//   "greedy_random_pass" — greedy with prob param of passing instead
//   "greedy_no_bomb"     — greedy; plays a bomb only with probability param
//   "tree_greedy"        — greedy with decision-tree evaluator; param = model depth
//                          (loads data/tree_model_d{int(param)}.txt)
//   "pimc"               — PIMC; greedy rollout after TB; param = samples N (default 10)
//   "pimc_tree"          — PIMC; tree rollout after TB; param = N;
//                          loads data/tree_model_d10.txt (depth 10)
//   "pimc_adaptive"      — PIMC + Hoeffding early stop; param = n_max (default 10)
//   "pimc_tree_adaptive" — same with tree rollout; depth 10 model file
//   "pimc_linear"        — PIMC; Ridge linear rollout; param = N;
//                          loads data/linear_rollout_w.txt
//   "pimc_redet"         — PIMC + greedy rollout; opponent hand re-sampled each
//                          opponent turn inside rollouts (param = N)
//   "pimc_linear_redet"  — same as pimc_linear with per-opponent-turn resampling
//   "pimc_tree_redet"    — same as pimc_tree with per-opponent-turn resampling
//   "greedy_linear"      — greedy with Ridge linear evaluator (same weights file)
//   "greedy_pass"        — greedy + Ridge pass-vs-greedy Δ (data/pass_ridge_w.txt)
//   "pimc_pass_rollout"  — PIMC + pass-aware rollout (param = N)
//   "pimc_pass_rollout_adaptive" — adaptive PIMC + pass-aware rollout
inline std::shared_ptr<PlayerFactory> make_player_factory(const std::string &name,
                                                           double param,
                                                           unsigned int seed) {
  if (name == "random")
    return std::make_shared<RandomPlayerFactory>(seed);
  if (name == "greedy")
    return std::make_shared<GreedyPlayerFactory>();
  if (name == "greedy_random")
    return std::make_shared<GreedyRandomPlayerFactory>(static_cast<float>(param), seed);
  if (name == "greedy_random_pass")
    return std::make_shared<GreedyRandomPassPlayerFactory>(static_cast<float>(param), seed);
  if (name == "greedy_no_bomb")
    return std::make_shared<GreedyNoBombPlayerFactory>(static_cast<float>(param), seed);
  if (name == "tree_greedy")
    return std::make_shared<TreeGreedyPlayerFactory>(param);
  if (name == "pimc")
    return std::make_shared<PimcPlayerFactory>(param == 0.0 ? 10.0 : param, seed);
  if (name == "pimc_tree")
    return std::make_shared<PimcTreePlayerFactory>(param == 0.0 ? 10.0 : param, 10,
                                                     seed);
  if (name == "pimc_adaptive")
    return std::make_shared<PimcAdaptivePlayerFactory>(param == 0.0 ? 10.0 : param,
                                                        seed);
  if (name == "pimc_tree_adaptive")
    return std::make_shared<PimcTreeAdaptivePlayerFactory>(
        param == 0.0 ? 10.0 : param, 10, seed);
  if (name == "pimc_linear")
    return std::make_shared<PimcLinearPlayerFactory>(param == 0.0 ? 10.0 : param, seed);
  if (name == "pimc_redet")
    return std::make_shared<PimcRedetPlayerFactory>(param == 0.0 ? 10.0 : param, seed);
  if (name == "pimc_linear_redet")
    return std::make_shared<PimcLinearRedetPlayerFactory>(param == 0.0 ? 10.0 : param,
                                                             seed);
  if (name == "pimc_tree_redet")
    return std::make_shared<PimcTreeRedetPlayerFactory>(param == 0.0 ? 10.0 : param, 10,
                                                         seed);
  if (name == "greedy_linear")
    return std::make_shared<GreedyLinearPlayerFactory>();
  if (name == "greedy_pass")
    return std::make_shared<GreedyPassPlayerFactory>();
  if (name == "pimc_pass_rollout")
    return std::make_shared<PimcPassRolloutPlayerFactory>(param == 0.0 ? 10.0 : param,
                                                            seed);
  if (name == "pimc_pass_rollout_adaptive")
    return std::make_shared<PimcPassRolloutAdaptivePlayerFactory>(
        param == 0.0 ? 10.0 : param, seed);
  if (name == "typed_search") {
    int v = static_cast<int>(std::round(param));
    std::string dir = (v == 0) ? std::string("data/typed_search")
                                 : std::string("data/typed_search_v") +
                                     std::to_string(v);
    return std::make_shared<typed_search::TypedSearchPlayerFactory>(seed, dir);
  }
#ifdef BIG2_WITH_TORCH
  if (name == "az_search") {
    // param = simulations per turn (default 200). Loads models/az_seq.pt;
    // play mode (deterministic opponent representative).
    int sims = (param <= 0.0) ? 200 : static_cast<int>(std::round(param));
    return std::make_shared<az_search::AzSearchPlayerFactory>(sims, seed);
  }
#endif

  std::cerr << "Error: unknown player '" << name << "'\n"
            << "Available: random, greedy, greedy_random, greedy_random_pass, "
               "greedy_no_bomb, greedy_linear, greedy_pass, tree_greedy, pimc, "
               "pimc_tree, pimc_linear, pimc_redet, pimc_linear_redet, pimc_tree_redet, "
               "pimc_adaptive, pimc_tree_adaptive, pimc_pass_rollout, "
               "pimc_pass_rollout_adaptive\n";
  return nullptr;
}

#endif
