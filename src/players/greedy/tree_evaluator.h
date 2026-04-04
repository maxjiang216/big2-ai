#ifndef TREE_EVALUATOR_H
#define TREE_EVALUATOR_H

#include "move.h"
#include "partial_game.h"
#include "util.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Feature extraction — must match FEATURE_COLS in analysis/train_tree_greedy.py
//
// All numeric turn_level features from feature_registry.h except:
//   turn_outcome (label), next_player, tb_case (used only for training filters).
// ---------------------------------------------------------------------------

static constexpr int TREE_N_FEATURES = 60;

inline int highest_rank_with_count(const std::array<int, 13> &hand, int count) {
  for (int r = 12; r >= 0; --r) {
    if (hand[r] >= count)
      return r;
  }
  return -1;
}

// Mirrors HighestRankWithCountNotBombFeature (count in [1,3]).
inline int highest_rank_with_count_not_bomb(const std::array<int, 13> &hand,
                                            int count) {
  for (int i = 12; i >= 0; --i) {
    bool is_bomb = (i == 11 && hand[i] == 3) || (i != 11 && hand[i] == 4);
    if (hand[i] >= count && !is_bomb)
      return i;
  }
  return -1;
}

inline int count_ge(const std::array<int, 13> &h, int start_idx) {
  int s = 0;
  for (int i = start_idx; i <= 12; ++i)
    s += h[i];
  return s;
}

inline int count_le(const std::array<int, 13> &h, int end_idx) {
  int s = 0;
  for (int i = 0; i <= end_idx; ++i)
    s += h[i];
  return s;
}

inline bool is_single_straight(Move::Combination c) {
  return c >= Move::Combination::kStraight5 &&
         c <= Move::Combination::kStraight13;
}
inline bool is_double_straight(Move::Combination c) {
  return c >= Move::Combination::kDoubleStraight2 &&
         c <= Move::Combination::kDoubleStraight8;
}
inline bool is_triple_straight(Move::Combination c) {
  return c >= Move::Combination::kTripleStraight2 &&
         c <= Move::Combination::kTripleStraight5;
}

inline std::array<float, TREE_N_FEATURES>
extract_tree_features(const PartialGame &sim) {
  const auto &h = sim.player_hand();
  const Move &lm = sim.last_move();
  Move::Combination c = lm.combination;

  int n_cards = 0, n_bombs = 0;
  for (int i = 0; i < 13; ++i) {
    n_cards += h[i];
    if ((i < 11 && h[i] == 4) || (i == 11 && h[i] == 3))
      ++n_bombs;
  }

  int pm    = static_cast<int>(sim.get_possible_moves().size());
  int pm_nb = static_cast<int>(sim.get_possible_moves_not_bomb().size());

  int mid = encodeMove(lm);
  int last_card_count = MOVE_TO_CARDS[mid][13];

  std::array<float, TREE_N_FEATURES> f{};
  int k = 0;

  f[k++] = static_cast<float>(n_cards);
  f[k++] = static_cast<float>(sim.opponent_hand_size());
  f[k++] = hand_is_only_singles(h) ? 1.f : 0.f;

  for (int i = 0; i < 13; ++i)
    f[k++] = static_cast<float>(h[i]);

  // n_ge_4 .. n_ge_a  (rank index 1..11).  Omit n_ge_2: for top rank "2", count_ge
  // equals n_2 (redundant with per-rank count).
  for (int start = 1; start <= 11; ++start)
    f[k++] = static_cast<float>(count_ge(h, start));

  // n_le_3 .. n_le_a  (RankLeFeature end index 0..11)
  for (int end = 0; end <= 11; ++end)
    f[k++] = static_cast<float>(count_le(h, end));

  f[k++] = static_cast<float>(highest_rank_with_count(h, 1));
  f[k++] = static_cast<float>(highest_rank_with_count(h, 2));
  f[k++] = static_cast<float>(highest_rank_with_count(h, 3));
  f[k++] = static_cast<float>(highest_rank_with_count(h, 4));

  f[k++] = static_cast<float>(highest_rank_with_count_not_bomb(h, 1));
  f[k++] = static_cast<float>(highest_rank_with_count_not_bomb(h, 2));
  f[k++] = static_cast<float>(highest_rank_with_count_not_bomb(h, 3));

  f[k++] = (c == Move::Combination::kPass) ? 1.f : 0.f;
  f[k++] = (c == Move::Combination::kSingle) ? 1.f : 0.f;
  f[k++] = (c == Move::Combination::kDouble) ? 1.f : 0.f;
  f[k++] = (c == Move::Combination::kTriple) ? 1.f : 0.f;
  f[k++] = (c == Move::Combination::kFullHouse) ? 1.f : 0.f;
  f[k++] = (c == Move::Combination::kBomb) ? 1.f : 0.f;

  f[k++] = is_single_straight(c) ? 1.f : 0.f;
  f[k++] = is_double_straight(c) ? 1.f : 0.f;
  f[k++] = is_triple_straight(c) ? 1.f : 0.f;

  f[k++] = static_cast<float>(last_card_count);
  f[k++] = static_cast<float>(n_bombs);
  f[k++] = static_cast<float>(pm);
  f[k++] = static_cast<float>(pm_nb);
  f[k++] = static_cast<float>(sim.get_trick_rank());

  return f;
}

// ---------------------------------------------------------------------------
// TreeEvaluator — loads a text-format decision tree and predicts win prob.
// ---------------------------------------------------------------------------

class TreeEvaluator {
public:
  TreeEvaluator() = default;

  explicit TreeEvaluator(const std::string &path) { load(path); }

  void load(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "r");
    if (!fp)
      throw std::runtime_error("TreeEvaluator: cannot open '" + path + "'");

    int n_nodes = 0, n_features = 0;
    if (std::fscanf(fp, "%d %d", &n_nodes, &n_features) != 2) {
      std::fclose(fp);
      throw std::runtime_error("TreeEvaluator: bad header in '" + path + "'");
    }
    if (n_features != TREE_N_FEATURES) {
      std::fclose(fp);
      throw std::runtime_error(
          "TreeEvaluator: feature count mismatch (file=" +
          std::to_string(n_features) + " expected=" +
          std::to_string(TREE_N_FEATURES) + ")");
    }

    feature_.resize(n_nodes);
    threshold_.resize(n_nodes);
    left_.resize(n_nodes);
    right_.resize(n_nodes);
    value_.resize(n_nodes);

    for (int i = 0; i < n_nodes; ++i) {
      if (std::fscanf(fp, "%d %f %d %d %f",
                      &feature_[i], &threshold_[i],
                      &left_[i], &right_[i], &value_[i]) != 5) {
        std::fclose(fp);
        throw std::runtime_error("TreeEvaluator: error reading node " +
                                 std::to_string(i) + " in '" + path + "'");
      }
    }
    std::fclose(fp);
  }

  bool loaded() const { return !feature_.empty(); }

  double predict(const PartialGame &sim) const {
    auto features = extract_tree_features(sim);
    int node = 0;
    while (feature_[node] != -2) {
      float x = features[feature_[node]];
      node = (x <= threshold_[node]) ? left_[node] : right_[node];
    }
    return static_cast<double>(value_[node]);
  }

private:
  std::vector<int>   feature_;
  std::vector<float> threshold_;
  std::vector<int>   left_;
  std::vector<int>   right_;
  std::vector<float> value_;
};

#endif
