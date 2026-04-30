#ifndef HAND_STRUCTURE_FEATURES_H
#define HAND_STRUCTURE_FEATURES_H

#include "feature_extractor.h"
#include "game_record.h"
#include "move.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

struct PassLegalAnalysis {
  std::array<int, 9> sr_best_rank{}; // straight5 .. straight13 -> -1 if none
  int best_single_straight_score{-1};

  std::array<int, 7> ds_best_rank{}; // double straight "length" 2..8 pairs
  int best_double_straight_score{-1};

  std::array<int, 4> ts_best_rank{}; // triple straight triplets 2..5
  int best_triple_straight_score{-1};

  int has_full_house{0};
  int has_straight5p{0};
  int has_any_double_straight{0};
  int has_any_triple_straight{0};
  int has_bomb_pass{0};
};

// Column IDs for registry / PassLegalColumnFeature (must match feature_registry entries).
constexpr int PL_COL_SR5 = 0;
constexpr int PL_COL_SR6 = 1;
constexpr int PL_COL_SR7 = 2;
constexpr int PL_COL_SR8 = 3;
constexpr int PL_COL_SR9 = 4;
constexpr int PL_COL_SR10 = 5;
constexpr int PL_COL_SR11 = 6;
constexpr int PL_COL_SR12 = 7;
constexpr int PL_COL_SR13 = 8;
constexpr int PL_COL_SCORE_SINGLE_STR = 9;
constexpr int PL_COL_DS2 = 10;
constexpr int PL_COL_DS3 = 11;
constexpr int PL_COL_DS4 = 12;
constexpr int PL_COL_DS5 = 13;
constexpr int PL_COL_DS6 = 14;
constexpr int PL_COL_DS7 = 15;
constexpr int PL_COL_DS8 = 16;
constexpr int PL_COL_SCORE_DOUBLE_STR = 17;
constexpr int PL_COL_TS2 = 18;
constexpr int PL_COL_TS3 = 19;
constexpr int PL_COL_TS4 = 20;
constexpr int PL_COL_TS5 = 21;
constexpr int PL_COL_SCORE_TRIPLE_STR = 22;
constexpr int PL_COL_HAS_FULL_HOUSE = 23;
constexpr int PL_COL_HAS_STRAIGHT5P = 24;
constexpr int PL_COL_HAS_DOUBLE_STR = 25;
constexpr int PL_COL_HAS_TRIPLE_STR = 26;
constexpr int PL_COL_HAS_BOMB_PASS = 27;

inline PassLegalAnalysis analyze_pass_legal(const std::array<int, 13> &hand) {
  PassLegalAnalysis a{};
  a.sr_best_rank.fill(-1);
  a.ds_best_rank.fill(-1);
  a.ts_best_rank.fill(-1);

  const std::vector<int> legal = compute_legal_moves(hand, kPASS);

  for (int mid : legal) {
    if (mid == kPASS)
      continue;
    Move m(mid);
    const int scr = m.numCards() + m.rank;

    switch (m.combination) {
    case Move::Combination::kFullHouse:
      a.has_full_house = 1;
      break;
    case Move::Combination::kBomb:
      a.has_bomb_pass = 1;
      break;
    case Move::Combination::kStraight5:
    case Move::Combination::kStraight6:
    case Move::Combination::kStraight7:
    case Move::Combination::kStraight8:
    case Move::Combination::kStraight9:
    case Move::Combination::kStraight10:
    case Move::Combination::kStraight11:
    case Move::Combination::kStraight12:
    case Move::Combination::kStraight13:
      a.has_straight5p = 1;
      if (scr > a.best_single_straight_score)
        a.best_single_straight_score = scr;
      {
        int idx =
            static_cast<int>(m.combination) -
            static_cast<int>(Move::Combination::kStraight5);
        if (idx >= 0 && idx < 9)
          a.sr_best_rank[idx] = std::max(a.sr_best_rank[idx], m.rank);
      }
      break;
    case Move::Combination::kDoubleStraight2:
    case Move::Combination::kDoubleStraight3:
    case Move::Combination::kDoubleStraight4:
    case Move::Combination::kDoubleStraight5:
    case Move::Combination::kDoubleStraight6:
    case Move::Combination::kDoubleStraight7:
    case Move::Combination::kDoubleStraight8:
      a.has_any_double_straight = 1;
      if (scr > a.best_double_straight_score)
        a.best_double_straight_score = scr;
      {
        int idx =
            static_cast<int>(m.combination) -
            static_cast<int>(Move::Combination::kDoubleStraight2);
        if (idx >= 0 && idx < 7)
          a.ds_best_rank[idx] = std::max(a.ds_best_rank[idx], m.rank);
      }
      break;
    case Move::Combination::kTripleStraight2:
    case Move::Combination::kTripleStraight3:
    case Move::Combination::kTripleStraight4:
    case Move::Combination::kTripleStraight5:
      a.has_any_triple_straight = 1;
      if (scr > a.best_triple_straight_score)
        a.best_triple_straight_score = scr;
      {
        int idx =
            static_cast<int>(m.combination) -
            static_cast<int>(Move::Combination::kTripleStraight2);
        if (idx >= 0 && idx < 4)
          a.ts_best_rank[idx] = std::max(a.ts_best_rank[idx], m.rank);
      }
      break;
    default:
      break;
    }
  }
  return a;
}

inline int pass_legal_pick(const PassLegalAnalysis &a, int col_id) {
  switch (col_id) {
  case PL_COL_SR5:
    return a.sr_best_rank[0];
  case PL_COL_SR6:
    return a.sr_best_rank[1];
  case PL_COL_SR7:
    return a.sr_best_rank[2];
  case PL_COL_SR8:
    return a.sr_best_rank[3];
  case PL_COL_SR9:
    return a.sr_best_rank[4];
  case PL_COL_SR10:
    return a.sr_best_rank[5];
  case PL_COL_SR11:
    return a.sr_best_rank[6];
  case PL_COL_SR12:
    return a.sr_best_rank[7];
  case PL_COL_SR13:
    return a.sr_best_rank[8];
  case PL_COL_SCORE_SINGLE_STR:
    return a.best_single_straight_score;
  case PL_COL_DS2:
    return a.ds_best_rank[0];
  case PL_COL_DS3:
    return a.ds_best_rank[1];
  case PL_COL_DS4:
    return a.ds_best_rank[2];
  case PL_COL_DS5:
    return a.ds_best_rank[3];
  case PL_COL_DS6:
    return a.ds_best_rank[4];
  case PL_COL_DS7:
    return a.ds_best_rank[5];
  case PL_COL_DS8:
    return a.ds_best_rank[6];
  case PL_COL_SCORE_DOUBLE_STR:
    return a.best_double_straight_score;
  case PL_COL_TS2:
    return a.ts_best_rank[0];
  case PL_COL_TS3:
    return a.ts_best_rank[1];
  case PL_COL_TS4:
    return a.ts_best_rank[2];
  case PL_COL_TS5:
    return a.ts_best_rank[3];
  case PL_COL_SCORE_TRIPLE_STR:
    return a.best_triple_straight_score;
  case PL_COL_HAS_FULL_HOUSE:
    return a.has_full_house;
  case PL_COL_HAS_STRAIGHT5P:
    return a.has_straight5p;
  case PL_COL_HAS_DOUBLE_STR:
    return a.has_any_double_straight;
  case PL_COL_HAS_TRIPLE_STR:
    return a.has_any_triple_straight;
  case PL_COL_HAS_BOMB_PASS:
    return a.has_bomb_pass;
  default:
    return 0;
  }
}

class PassLegalColumnFeature : public FeatureExtractor {
public:
  PassLegalColumnFeature(int col_id, std::string name)
      : col_id_(col_id), name_(std::move(name)) {}

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> out;
    out.reserve(record.turns().size() * 2);
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        PassLegalAnalysis a =
            analyze_pass_legal(turn.views[p].player_hand());
        out.push_back(pass_legal_pick(a, col_id_));
      }
    }
    return out;
  }

private:
  int col_id_;
  std::string name_;
};

#endif
