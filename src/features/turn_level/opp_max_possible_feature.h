#ifndef OPP_MAX_POSSIBLE_FEATURE_H
#define OPP_MAX_POSSIBLE_FEATURE_H

#include "feature_extractor.h"
#include "util.h"

#include <algorithm>
#include <stdexcept>

// Turn-level feature: maximum number of cards of a specific rank that the
// opponent could possibly hold, given what this perspective player can see
// (their own hand + the discard pile).
//
// opp_max[r] = max(0, max_deck[r] - player_hand[r] - discard[r])
//
// This is computed independently per perspective, so each perspective reflects
// what that player can deduce.  Rank index 0=3, 1=4, ..., 10=K, 11=A, 12=2.
class OppMaxPossibleFeature : public FeatureExtractor {
public:
  explicit OppMaxPossibleFeature(int rank, const std::string &rank_name)
      : rank_(rank), rank_name_(rank_name) {
    if (rank < 0 || rank > 12)
      throw std::invalid_argument("Rank must be in [0, 12]");
  }

  Type type() const override { return Type::TurnLevel; }
  std::string name() const override { return "opp_max_" + rank_name_; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    for (const auto &turn : record.turns()) {
      for (int p = 0; p < 2; ++p) {
        const auto &hand = turn.views[p].player_hand();
        const auto &disc = turn.views[p].discard_pile();
        int max_p = std::max(
            0, max_cards_in_deck_for_rank(rank_) - hand[rank_] - disc[rank_]);
        features.push_back(max_p);
      }
    }
    return features;
  }

private:
  int rank_;
  std::string rank_name_;
};

#endif
