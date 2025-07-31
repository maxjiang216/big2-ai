#include "bomb_feature.h"
#include "features/turn_level/rank_le_feature.h"
#include "game_coordinator.h"
#include "greedy_player_factory.h"
#include "highest_double_feature.h"
#include "highest_double_not_bomb_feature.h"
#include "highest_rank_feature.h"
#include "highest_triple_feature.h"
#include "highest_triple_not_bomb_feature.h"
#include "last_move_card_count_feature.h"
#include "last_trick_combination_feature.h"
#include "last_trick_straight_feature.h"
#include "length_feature.h"
#include "new_trick_feature.h"
#include "next_player_feature.h"
#include "opponent_hand_size_feature.h"
#include "outcome_feature.h"
#include "player_hand_size_feature.h"
#include "possible_moves_feature.h"
#include "possible_moves_not_bomb_feature.h"
#include "random_player_factory.h"
#include "rank_count_feature.h"
#include "rank_ge_feature.h"
#include "trick_rank_feature.h"
#include "turn_outcome_feature.h"
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

// Helper to instantiate and register features
template <typename... FeatureTypes>
std::vector<std::shared_ptr<FeatureExtractor>> make_feature_list() {
  std::vector<std::shared_ptr<FeatureExtractor>> features;
  (void)std::initializer_list<int>{(
      features.push_back(std::make_shared<FeatureTypes>()),
      std::cout << "Added feature: " << std::make_shared<FeatureTypes>()->name()
                << std::endl,
      0)...};
  return features;
}

int main() {
  // Config
  // Crashes at 400k
  int num_games = 150000;
  std::string output_path = "game_records.jsonl";
  int num_threads = std::max(1u, std::thread::hardware_concurrency());
  unsigned int seed = std::random_device{}();

  std::cout << "Running " << num_games << " self-play games with "
            << num_threads << " threads, output to '" << output_path
            << "'...\n";

  // Players
  auto factory0 = std::make_shared<GreedyPlayerFactory>();
  auto factory1 = std::make_shared<GreedyPlayerFactory>();

  // ----------- DEFINE FEATURES HERE -------------
  auto game_features =
      make_feature_list<OutcomeFeature, GameLengthExtractor
                        // Add more game-level feature classes here
                        >();

  // ---- Instantiate all 13 rank features, 3 (0) ... 2 (12) ----
  std::vector<std::shared_ptr<FeatureExtractor>> rank_features = {
      std::make_shared<RankCountFeature<0>>("n_threes"),
      std::make_shared<RankCountFeature<1>>("n_fours"),
      std::make_shared<RankCountFeature<2>>("n_fives"),
      std::make_shared<RankCountFeature<3>>("n_sixes"),
      std::make_shared<RankCountFeature<4>>("n_sevens"),
      std::make_shared<RankCountFeature<5>>("n_eights"),
      std::make_shared<RankCountFeature<6>>("n_nines"),
      std::make_shared<RankCountFeature<7>>("n_tens"),
      std::make_shared<RankCountFeature<8>>("n_jacks"),
      std::make_shared<RankCountFeature<9>>("n_queens"),
      std::make_shared<RankCountFeature<10>>("n_kings"),
      std::make_shared<RankCountFeature<11>>("n_aces"),
      std::make_shared<RankCountFeature<12>>("n_twos")};

  // ---- Instantiate all 13 "rank >= X" features, 3 (0) ... 2 (12) ----
  std::vector<std::shared_ptr<FeatureExtractor>> rank_ge_features = {
      std::make_shared<RankGreaterEqualFeature<1>>("n_ge_four"),  // 4 and up
      std::make_shared<RankGreaterEqualFeature<2>>("n_ge_five"),  // 5 and up
      std::make_shared<RankGreaterEqualFeature<3>>("n_ge_six"),   // 6 and up
      std::make_shared<RankGreaterEqualFeature<4>>("n_ge_seven"), // 7 and up
      std::make_shared<RankGreaterEqualFeature<5>>("n_ge_eight"), // 8 and up
      std::make_shared<RankGreaterEqualFeature<6>>("n_ge_nine"),  // 9 and up
      std::make_shared<RankGreaterEqualFeature<7>>("n_ge_ten"),   // 10 and up
      std::make_shared<RankGreaterEqualFeature<8>>("n_ge_jack"),  // J and up
      std::make_shared<RankGreaterEqualFeature<9>>("n_ge_queen"), // Q and up
      std::make_shared<RankGreaterEqualFeature<10>>("n_ge_king"), // K and up
      std::make_shared<RankGreaterEqualFeature<11>>("n_ge_ace"),  // A and up
  };

  // ...
  std::vector<std::shared_ptr<FeatureExtractor>> rank_le_features = {
      std::make_shared<RankLessEqualFeature<1>>(), // "n_le_four"
      std::make_shared<RankLessEqualFeature<2>>(), // etc...
      std::make_shared<RankLessEqualFeature<3>>(),
      std::make_shared<RankLessEqualFeature<4>>(),
      std::make_shared<RankLessEqualFeature<5>>(),
      std::make_shared<RankLessEqualFeature<6>>(),
      std::make_shared<RankLessEqualFeature<7>>(),
      std::make_shared<RankLessEqualFeature<8>>(),
      std::make_shared<RankLessEqualFeature<9>>(),
      std::make_shared<RankLessEqualFeature<10>>(),
      std::make_shared<RankLessEqualFeature<11>>(),
  };

  // --- LastTrickCombinationFeature for each combination ---

  std::vector<std::shared_ptr<FeatureExtractor>> trick_combo_features = {
      std::make_shared<LastTrickCombinationFeature<Move::Combination::kSingle>>(
          "last_trick_is_single"),
      std::make_shared<LastTrickCombinationFeature<Move::Combination::kDouble>>(
          "last_trick_is_double"),
      std::make_shared<LastTrickCombinationFeature<Move::Combination::kTriple>>(
          "last_trick_is_triple"),
      std::make_shared<
          LastTrickCombinationFeature<Move::Combination::kFullHouse>>(
          "last_trick_is_fullhouse"),
      std::make_shared<LastTrickCombinationFeature<Move::Combination::kBomb>>(
          "last_trick_is_bomb"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight5>>(
      //       "last_trick_is_straight5"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight6>>(
      //       "last_trick_is_straight6"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight7>>(
      //       "last_trick_is_straight7"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight8>>(
      //       "last_trick_is_straight8"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight9>>(
      //       "last_trick_is_straight9"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight10>>(
      //       "last_trick_is_straight10"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight11>>(
      //       "last_trick_is_straight11"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight12>>(
      //       "last_trick_is_straight12"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kStraight13>>(
      //       "last_trick_is_straight13"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kDoubleStraight2>>(
      //       "last_trick_is_doublestraight2"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kDoubleStraight3>>(
      //       "last_trick_is_doublestraight3"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kDoubleStraight4>>(
      //       "last_trick_is_doublestraight4"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kDoubleStraight5>>(
      //       "last_trick_is_doublestraight5"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kDoubleStraight6>>(
      //       "last_trick_is_doublestraight6"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kTripleStraight2>>(
      //       "last_trick_is_triplestraight2"),
      //   std::make_shared<
      //       LastTrickCombinationFeature<Move::Combination::kTripleStraight3>>(
      //       "last_trick_is_triplestraight3"),
  };

  std::vector<std::shared_ptr<FeatureExtractor>> straight_type_features = {
      std::make_shared<LastTrickIsStraightFeature<1>>(),
      std::make_shared<LastTrickIsStraightFeature<2>>(),
      std::make_shared<LastTrickIsStraightFeature<3>>(),
  };

  auto turn_features =
      make_feature_list<TurnOutcomeFeature, NextPlayerFeature,
                        PlayerHandSizeFeature, OpponentHandSizeFeature,
                        // Trick rank covers new trick
                        TrickRankFeature, BombFeature, HighestRankFeature,
                        HighestDoubleFeature, HighestDoubleNotBombFeature,
                        HighestTripleFeature, HighestTripleNotBombFeature,
                        LastMoveCardCountFeature, PossibleMovesFeature,
                        PossibleMovesNotBombFeature
                        // Add more turn-level feature classes here
                        >();

  // Append the 13 rank features to your list:
  turn_features.insert(turn_features.end(), rank_features.begin(),
                       rank_features.end());

  turn_features.insert(turn_features.end(), rank_ge_features.begin(),
                       rank_ge_features.end());

  turn_features.insert(turn_features.end(), rank_le_features.begin(),
                       rank_le_features.end());

  turn_features.insert(turn_features.end(), trick_combo_features.begin(),
                       trick_combo_features.end());

  turn_features.insert(turn_features.end(), straight_type_features.begin(),
                       straight_type_features.end());

  // ------------------------------------------------

  std::cout << "Game features: " << game_features.size() << std::endl;
  std::cout << "Turn features: " << turn_features.size() << std::endl;

  GameCoordinator coordinator(factory0, factory1, num_games, output_path,
                              num_threads, seed, "", game_features,
                              turn_features);

  std::cout << "Starting simulation..." << std::endl;
  coordinator.run_all("game_features.parquet", "turn_features.parquet");
  std::cout << "Simulation completed." << std::endl;

  std::cout << "Done.\n";
  return 0;
}