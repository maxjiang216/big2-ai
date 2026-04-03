#ifndef TABLEBASE_SEGMENT_FEATURES_H
#define TABLEBASE_SEGMENT_FEATURES_H

#include "feature_extractor.h"

// Game-level tablebase stats: each value is derived from first-hit segments only
// (see tablebase_first_hit_stats in game_record.h).

class TbCase1GameFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "tb_case1"; }

  int gameExtract(const GameRecord &record) const override {
    return tablebase_first_hit_stats(record).case1;
  }
};

class TbCase2GameFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "tb_case2"; }

  int gameExtract(const GameRecord &record) const override {
    return tablebase_first_hit_stats(record).case2;
  }
};

// Sum of forced-win sequence lengths at each case1 first hit in this game.
class TbForcedSeqLenSumGameFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "tb_forced_seq_len_sum"; }

  int gameExtract(const GameRecord &record) const override {
    auto s = tablebase_first_hit_stats(record);
    return static_cast<int>(s.case1_seq_sum);
  }
};

// Max forced-win sequence length among case1 first hits in this game.
class TbForcedSeqLenGameFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "tb_forced_seq_len"; }

  int gameExtract(const GameRecord &record) const override {
    return tablebase_first_hit_stats(record).case1_seq_max;
  }
};

// Count of case2 first hits that used the opp-1 precomputed straight table.
class TbOpp1TableStraightGameFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::GameLevel; }
  std::string name() const override { return "tb_opp1_table_straight"; }

  int gameExtract(const GameRecord &record) const override {
    return tablebase_first_hit_stats(record).case2_table_straight;
  }
};

#endif
