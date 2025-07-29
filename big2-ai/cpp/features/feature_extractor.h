#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <memory>
#include <string>
#include <vector>

#include "../game_record.h"

class GameRecord;
class TurnRecord;

class FeatureExtractor {
public:
  enum class Type { GameLevel, TurnLevel };

  virtual ~FeatureExtractor() = default;

  // Identify feature type
  virtual Type type() const = 0;

  // Canonical column name for output
  virtual std::string name() const = 0;

  // --------- Game-level: One value per game ----------
  virtual int gameExtract(const GameRecord &game) const {
    throw std::logic_error("Game-level extract() not implemented.");
  }

  // --------- Turn-level: Two values per turn (one for each perspective)
  // ----------
  virtual std::vector<int> turnExtract(const GameRecord &game) const {
    throw std::logic_error("Turn-level extract() not implemented.");
  }
};

#endif // FEATURE_EXTRACTOR_H
