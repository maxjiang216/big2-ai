#ifndef LAST_TRICK_COMBINATION_FEATURE_H
#define LAST_TRICK_COMBINATION_FEATURE_H

#include "../../game_record.h"
#include "../../move.h"
#include "../feature_extractor.h"
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Outputs 1 if the last trick type matches the given combination,
 *        else 0. For both player perspectives at each turn.
 *
 * Usage: instantiate with the desired Move::Combination enum value.
 * Example: LastTrickCombinationFeature<Move::Combination::kBomb>("is_bomb")
 */
template <Move::Combination COMBO>
class LastTrickCombinationFeature : public FeatureExtractor {
public:
  Type type() const override { return Type::TurnLevel; }

  // You can give a custom name or auto-generate:
  LastTrickCombinationFeature(const std::string &name = "")
      : _name(name.empty() ? make_default_name() : name) {}

  std::string name() const override { return _name; }

  std::vector<int> turnExtract(const GameRecord &record) const override {
    std::vector<int> features;
    const auto &turns = record.turns();
    for (const auto &turn : turns) {
      for (int perspective = 0; perspective < 2; ++perspective) {
        const PartialGame &view = turn.views[perspective];
        const Move &last_move = view.last_move();
        int is_combo = (last_move.combination == COMBO ? 1 : 0);
        features.push_back(is_combo);
      }
    }
    return features;
  }

private:
  std::string _name;

  // Auto-generate feature name based on combination
  static std::string make_default_name() {
    std::ostringstream oss;
    oss << "last_trick_is_" << combination_to_string(COMBO);
    return oss.str();
  }
};

#endif // LAST_TRICK_COMBINATION_FEATURE_H
