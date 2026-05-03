#ifndef FEATURE_REGISTRY_H
#define FEATURE_REGISTRY_H

// Game-level features
#include "game_level/outcome_feature.h"
#include "game_level/length_feature.h"
#include "game_level/tablebase_hits_feature.h"
#include "game_level/tablebase_segment_features.h"
#include "game_level/start_legal_moves_feature.h"

// Turn-level features
#include "turn_level/turn_outcome_feature.h"
#include "turn_level/next_player_feature.h"
#include "turn_level/player_hand_size_feature.h"
#include "turn_level/opponent_hand_size_feature.h"
#include "turn_level/rank_feature.h"
#include "turn_level/rank_ge_feature.h"
#include "turn_level/rank_le_feature.h"
#include "turn_level/highest_rank_with_count_feature.h"
#include "turn_level/highest_rank_with_count_not_bomb_feature.h"
#include "turn_level/last_move_combination_feature.h"
#include "turn_level/last_move_is_straight_feature.h"
#include "turn_level/last_move_card_count_feature.h"
#include "turn_level/bomb_feature.h"
#include "turn_level/possible_moves_feature.h"
#include "turn_level/possible_moves_not_bomb_feature.h"
#include "turn_level/trick_rank_feature.h"
#include "turn_level/tb_case_feature.h"
#include "turn_level/only_single_feature.h"
#include "turn_level/hand_structure_features.h"

#include "feature_extractor.h"
#include "move.h"

#include <iostream>
#include <memory>
#include <string>

// Returns a FeatureExtractor for the given name, or nullptr on unknown name.
// The name strings match the old implementation's generate_data CLI conventions.
inline std::shared_ptr<FeatureExtractor> create_feature(const std::string &name) {
  // --- game-level ---
  if (name == "outcome")  return std::make_shared<OutcomeFeature>();
  if (name == "length")   return std::make_shared<GameLengthExtractor>();
  if (name == "tb_hits")  return std::make_shared<TablebaseHitsFeature>();
  if (name == "tb_case1") return std::make_shared<TbCase1GameFeature>();
  if (name == "tb_case2") return std::make_shared<TbCase2GameFeature>();
  if (name == "tb_forced_seq_len")
    return std::make_shared<TbForcedSeqLenGameFeature>();
  if (name == "tb_forced_seq_len_sum")
    return std::make_shared<TbForcedSeqLenSumGameFeature>();
  if (name == "tb_opp1_table_straight")
    return std::make_shared<TbOpp1TableStraightGameFeature>();
  if (name == "start_legal_moves")
    return std::make_shared<StartLegalMovesFeature>();

  // --- turn-level: basic ---
  if (name == "turn_outcome")      return std::make_shared<TurnOutcomeFeature>();
  if (name == "next_player")       return std::make_shared<NextPlayerFeature>();
  if (name == "player_hand_size")  return std::make_shared<PlayerHandSizeFeature>();
  if (name == "opponent_hand_size") return std::make_shared<OpponentHandSizeFeature>();

  // --- rank counts (n_3 .. n_2) ---
  if (name == "n_3")  return std::make_shared<RankFeature>(0,  "3");
  if (name == "n_4")  return std::make_shared<RankFeature>(1,  "4");
  if (name == "n_5")  return std::make_shared<RankFeature>(2,  "5");
  if (name == "n_6")  return std::make_shared<RankFeature>(3,  "6");
  if (name == "n_7")  return std::make_shared<RankFeature>(4,  "7");
  if (name == "n_8")  return std::make_shared<RankFeature>(5,  "8");
  if (name == "n_9")  return std::make_shared<RankFeature>(6,  "9");
  if (name == "n_10") return std::make_shared<RankFeature>(7,  "10");
  if (name == "n_j")  return std::make_shared<RankFeature>(8,  "j");
  if (name == "n_q")  return std::make_shared<RankFeature>(9,  "q");
  if (name == "n_k")  return std::make_shared<RankFeature>(10, "k");
  if (name == "n_a")  return std::make_shared<RankFeature>(11, "a");
  if (name == "n_2")  return std::make_shared<RankFeature>(12, "2");

  // --- rank >= threshold ---
  if (name == "n_ge_4")  return std::make_shared<RankGeFeature>(1,  "4");
  if (name == "n_ge_5")  return std::make_shared<RankGeFeature>(2,  "5");
  if (name == "n_ge_6")  return std::make_shared<RankGeFeature>(3,  "6");
  if (name == "n_ge_7")  return std::make_shared<RankGeFeature>(4,  "7");
  if (name == "n_ge_8")  return std::make_shared<RankGeFeature>(5,  "8");
  if (name == "n_ge_9")  return std::make_shared<RankGeFeature>(6,  "9");
  if (name == "n_ge_10") return std::make_shared<RankGeFeature>(7,  "10");
  if (name == "n_ge_j")  return std::make_shared<RankGeFeature>(8,  "j");
  if (name == "n_ge_q")  return std::make_shared<RankGeFeature>(9,  "q");
  if (name == "n_ge_k")  return std::make_shared<RankGeFeature>(10, "k");
  if (name == "n_ge_a")  return std::make_shared<RankGeFeature>(11, "a");
  if (name == "n_ge_2")  return std::make_shared<RankGeFeature>(12, "2");

  // --- rank <= threshold ---
  if (name == "n_le_3")  return std::make_shared<RankLeFeature>(0,  "3");
  if (name == "n_le_4")  return std::make_shared<RankLeFeature>(1,  "4");
  if (name == "n_le_5")  return std::make_shared<RankLeFeature>(2,  "5");
  if (name == "n_le_6")  return std::make_shared<RankLeFeature>(3,  "6");
  if (name == "n_le_7")  return std::make_shared<RankLeFeature>(4,  "7");
  if (name == "n_le_8")  return std::make_shared<RankLeFeature>(5,  "8");
  if (name == "n_le_9")  return std::make_shared<RankLeFeature>(6,  "9");
  if (name == "n_le_10") return std::make_shared<RankLeFeature>(7,  "10");
  if (name == "n_le_j")  return std::make_shared<RankLeFeature>(8,  "j");
  if (name == "n_le_q")  return std::make_shared<RankLeFeature>(9,  "q");
  if (name == "n_le_k")  return std::make_shared<RankLeFeature>(10, "k");
  if (name == "n_le_a")  return std::make_shared<RankLeFeature>(11, "a");

  // --- highest rank with N copies ---
  if (name == "highest_single") return std::make_shared<HighestRankWithCountFeature>(1, "single");
  if (name == "highest_double") return std::make_shared<HighestRankWithCountFeature>(2, "double");
  if (name == "highest_triple") return std::make_shared<HighestRankWithCountFeature>(3, "triple");
  if (name == "highest_bomb")   return std::make_shared<HighestRankWithCountFeature>(4, "bomb");

  // --- highest rank with N copies, excluding bombs ---
  if (name == "highest_single_not_bomb")
    return std::make_shared<HighestRankWithCountNotBombFeature>(1, "single_not_bomb");
  if (name == "highest_double_not_bomb")
    return std::make_shared<HighestRankWithCountNotBombFeature>(2, "double_not_bomb");
  if (name == "highest_triple_not_bomb")
    return std::make_shared<HighestRankWithCountNotBombFeature>(3, "triple_not_bomb");

  // --- last move combination ---
  if (name == "last_move_is_pass")
    return std::make_shared<LastMoveCombinationFeature>(Move::Combination::kPass, "pass");
  if (name == "last_move_is_single")
    return std::make_shared<LastMoveCombinationFeature>(Move::Combination::kSingle, "single");
  if (name == "last_move_is_double")
    return std::make_shared<LastMoveCombinationFeature>(Move::Combination::kDouble, "double");
  if (name == "last_move_is_triple")
    return std::make_shared<LastMoveCombinationFeature>(Move::Combination::kTriple, "triple");
  if (name == "last_move_is_full_house")
    return std::make_shared<LastMoveCombinationFeature>(Move::Combination::kFullHouse, "full_house");
  if (name == "last_move_is_bomb")
    return std::make_shared<LastMoveCombinationFeature>(Move::Combination::kBomb, "bomb");

  // --- last move straight type ---
  if (name == "last_move_is_single_straight")
    return std::make_shared<LastMoveIsStraightFeature>(1, "single_straight");
  if (name == "last_move_is_double_straight")
    return std::make_shared<LastMoveIsStraightFeature>(2, "double_straight");
  if (name == "last_move_is_triple_straight")
    return std::make_shared<LastMoveIsStraightFeature>(3, "triple_straight");

  // --- miscellaneous ---
  if (name == "last_move_card_count") return std::make_shared<LastMoveCardCountFeature>();
  if (name == "n_bombs")              return std::make_shared<BombFeature>();
  if (name == "possible_moves")       return std::make_shared<PossibleMovesFeature>();
  if (name == "possible_moves_not_bomb") return std::make_shared<PossibleMovesNotBombFeature>();
  if (name == "trick_rank")           return std::make_shared<TrickRankFeature>();
  if (name == "tb_case")              return std::make_shared<TbCaseFeature>();
  if (name == "only_single")          return std::make_shared<OnlySingleFeature>();

  // --- intrinsic hand combos on fresh trick (compute_legal_moves with Pass lead) ---
  if (name == "pass_sr5_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR5, "pass_sr5_best_rank");
  if (name == "pass_sr6_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR6, "pass_sr6_best_rank");
  if (name == "pass_sr7_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR7, "pass_sr7_best_rank");
  if (name == "pass_sr8_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR8, "pass_sr8_best_rank");
  if (name == "pass_sr9_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR9, "pass_sr9_best_rank");
  if (name == "pass_sr10_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR10, "pass_sr10_best_rank");
  if (name == "pass_sr11_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR11, "pass_sr11_best_rank");
  if (name == "pass_sr12_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR12, "pass_sr12_best_rank");
  if (name == "pass_sr13_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SR13, "pass_sr13_best_rank");
  if (name == "pass_best_score_single_straight")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SCORE_SINGLE_STR, "pass_best_score_single_straight");
  if (name == "pass_ds2_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS2, "pass_ds2_best_rank");
  if (name == "pass_ds3_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS3, "pass_ds3_best_rank");
  if (name == "pass_ds4_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS4, "pass_ds4_best_rank");
  if (name == "pass_ds5_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS5, "pass_ds5_best_rank");
  if (name == "pass_ds6_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS6, "pass_ds6_best_rank");
  if (name == "pass_ds7_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS7, "pass_ds7_best_rank");
  if (name == "pass_ds8_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_DS8, "pass_ds8_best_rank");
  if (name == "pass_best_score_double_straight")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SCORE_DOUBLE_STR, "pass_best_score_double_straight");
  if (name == "pass_ts2_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_TS2, "pass_ts2_best_rank");
  if (name == "pass_ts3_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_TS3, "pass_ts3_best_rank");
  if (name == "pass_ts4_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_TS4, "pass_ts4_best_rank");
  if (name == "pass_ts5_best_rank")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_TS5, "pass_ts5_best_rank");
  if (name == "pass_best_score_triple_straight")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_SCORE_TRIPLE_STR, "pass_best_score_triple_straight");
  if (name == "pass_has_full_house")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_HAS_FULL_HOUSE, "pass_has_full_house");
  if (name == "pass_has_straight5p")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_HAS_STRAIGHT5P, "pass_has_straight5p");
  if (name == "pass_has_double_straight")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_HAS_DOUBLE_STR, "pass_has_double_straight");
  if (name == "pass_has_triple_straight")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_HAS_TRIPLE_STR, "pass_has_triple_straight");
  if (name == "pass_has_bomb")
    return std::make_shared<PassLegalColumnFeature>(
        PL_COL_HAS_BOMB_PASS, "pass_has_bomb");

  std::cerr << "Warning: unknown feature '" << name << "', skipping\n";
  return nullptr;
}

#endif
