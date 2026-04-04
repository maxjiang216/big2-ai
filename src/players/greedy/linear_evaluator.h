#ifndef LINEAR_EVALUATOR_H
#define LINEAR_EVALUATOR_H

#include "partial_game.h"
#include "tree_evaluator.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Ridge linear model on a subset of ``extract_tree_features`` dimensions.
//
// Weight file format (text):
//   line 1: k  (number of active features)
//   line 2: intercept  (double)
//   line 3: k indices into [0, TREE_N_FEATURES)  (tree / FEATURE_COLS order)
//   line 4: k means
//   line 5: k scales (std); use >= 1e-12 to avoid division by zero
//   line 6: k coefficients
//
// Predict: intercept + sum_j coef[j] * (f[idx[j]] - mean[j]) / scale[j]
//
// Written by ``analysis/train_linear_rollout.py``.
// ---------------------------------------------------------------------------

class LinearEvaluator {
public:
  LinearEvaluator() = default;

  explicit LinearEvaluator(const std::string &path) { load(path); }

  void load(const std::string &path) {
    loaded_ = false;
    k_ = 0;
    intercept_ = 0.0;
    feat_idx_.clear();
    mean_.clear();
    scale_.clear();
    coef_.clear();

    FILE *fp = std::fopen(path.c_str(), "r");
    if (!fp)
      throw std::runtime_error("LinearEvaluator: cannot open '" + path + "'");

    if (std::fscanf(fp, "%d", &k_) != 1) {
      std::fclose(fp);
      throw std::runtime_error("LinearEvaluator: bad k in '" + path + "'");
    }
    if (k_ <= 0 || k_ > TREE_N_FEATURES) {
      std::fclose(fp);
      throw std::runtime_error("LinearEvaluator: invalid k=" + std::to_string(k_) +
                               " in '" + path + "'");
    }
    if (std::fscanf(fp, "%lf", &intercept_) != 1) {
      std::fclose(fp);
      throw std::runtime_error("LinearEvaluator: bad intercept in '" + path + "'");
    }

    feat_idx_.resize(k_);
    mean_.resize(k_);
    scale_.resize(k_);
    coef_.resize(k_);

    for (int j = 0; j < k_; ++j) {
      if (std::fscanf(fp, "%d", &feat_idx_[j]) != 1) {
        std::fclose(fp);
        throw std::runtime_error("LinearEvaluator: bad feature index " + std::to_string(j) +
                                 " in '" + path + "'");
      }
      if (feat_idx_[j] < 0 || feat_idx_[j] >= TREE_N_FEATURES) {
        std::fclose(fp);
        throw std::runtime_error("LinearEvaluator: bad feature index value " +
                                 std::to_string(feat_idx_[j]) + " in '" + path + "'");
      }
    }
    for (int j = 0; j < k_; ++j) {
      if (std::fscanf(fp, "%lf", &mean_[j]) != 1) {
        std::fclose(fp);
        throw std::runtime_error("LinearEvaluator: bad mean " + std::to_string(j) +
                                 " in '" + path + "'");
      }
    }
    for (int j = 0; j < k_; ++j) {
      if (std::fscanf(fp, "%lf", &scale_[j]) != 1) {
        std::fclose(fp);
        throw std::runtime_error("LinearEvaluator: bad scale " + std::to_string(j) +
                                 " in '" + path + "'");
      }
    }
    for (int j = 0; j < k_; ++j) {
      if (std::fscanf(fp, "%lf", &coef_[j]) != 1) {
        std::fclose(fp);
        throw std::runtime_error("LinearEvaluator: bad coef " + std::to_string(j) +
                                 " in '" + path + "'");
      }
    }
    std::fclose(fp);
    loaded_ = true;
  }

  bool loaded() const { return loaded_; }

  double predict(const PartialGame &sim) const {
    auto f = extract_tree_features(sim);
    double s = intercept_;
    for (int j = 0; j < k_; ++j) {
      int idx = feat_idx_[j];
      double z = static_cast<double>(f[idx]) - mean_[j];
      double sc = scale_[j];
      if (sc < 1e-12)
        sc = 1.0;
      z /= sc;
      s += coef_[j] * z;
    }
    return s;
  }

private:
  bool loaded_{false};
  int k_{0};
  double intercept_{0.0};
  std::vector<int> feat_idx_;
  std::vector<double> mean_;
  std::vector<double> scale_;
  std::vector<double> coef_;
};

#endif
