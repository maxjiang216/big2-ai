#ifndef MOVE_CARD_COUNT_FEATURE_H
#define MOVE_CARD_COUNT_FEATURE_H

#include "feature_extractor.h"
#include "move.h"
#include "util.h"

#include <stdexcept>

// Turn-level feature: count of a specific rank used in the move played this
// turn.  Rank index 0=3, 1=4, ..., 10=K, 11=A, 12=2.
// Same value is emitted for both perspectives since the played move is public.
class MoveCardCountFeature : public FeatureExtractor {
public:
  explicit MoveCardCountFeature(int rank, const std::string &rank_name)
      : rank_(rank), rank_name_(rank_name) {
    if (rank < 0 || rank > 12)
      throw std::invalid_argument("Rank must be in [0, 12]");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "move_" + rank_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      int move_id = encodeMove(turn.move);
      int count = MOVE_TO_CARDS[move_id][rank_];
      features.push_back(count);
      features.push_back(count);
    }
    return features;
  }

private:
  int rank_;
  std::string rank_name_;
};

#endif
