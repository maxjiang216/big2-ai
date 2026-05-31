#ifndef AZ_SEARCH_EVALUATOR_H
#define AZ_SEARCH_EVALUATOR_H

// Abstract leaf evaluator for the search core. Torch-free: the real NN-backed
// evaluator (nn_eval.h) and a deterministic test stub both implement it, so the
// search core compiles and unit-tests without LibTorch. Returns RAW logits +
// value (P(root/searcher wins)); the search applies the legal mask + softmax.

#include "features.h"

namespace az_search {

class Evaluator {
public:
  virtual ~Evaluator() = default;
  virtual PlayerEval eval_player(const PlayerFeatures &f) = 0;
  virtual OppEval eval_opp(const OppFeatures &f) = 0;
};

}  // namespace az_search

#endif  // AZ_SEARCH_EVALUATOR_H
