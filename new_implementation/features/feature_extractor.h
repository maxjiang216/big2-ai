#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class GameRecord;

class FeatureExtractor {
public:
  enum class Type { GameLevel, TurnLevel };

  virtual ~FeatureExtractor() = default;

  virtual Type type() const = 0;

  virtual std::string name() const = 0;

  virtual int gameExtract(const GameRecord & /*game*/) const {
    throw std::logic_error("Game-level extract() not implemented.");
  }

  virtual std::vector<int> turnExtract(const GameRecord & /*game*/) const {
    throw std::logic_error("Turn-level extract() not implemented.");
  }
};

#endif
