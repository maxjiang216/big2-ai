#ifndef GAME_COORDINATOR_H
#define GAME_COORDINATOR_H

#include "game_record.h"
#include "feature_extractor.h"

#include "player_factory.h"

#include <climits>
#include <atomic>
#include <memory>
#include <ostream>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

class GameCoordinator {
public:
  GameCoordinator(
      std::shared_ptr<PlayerFactory> player_factory_p0,
      std::shared_ptr<PlayerFactory> player_factory_p1, int num_games,
      const std::string &output_path, int num_threads,
      unsigned int random_seed = std::random_device{}(),
      const std::string &log_path = "",
      std::vector<std::shared_ptr<FeatureExtractor>> game_level_features = {},
      std::vector<std::shared_ptr<FeatureExtractor>> turn_level_features = {});

  // If both paths are empty, runs games and accumulates stats only (no Parquet).
  void run_all(const std::string &game_feature_out,
               const std::string &turn_feature_out);

  int games_completed() const { return _games_completed.load(); }

  // After run_all: full indexed list for sample_games.md (deduplicated anomalies).
  const std::vector<std::pair<int, GameRecord>> &anomaly_indexed_games() const {
    return _anomaly_indexed;
  }

  void print_run_summary(std::ostream &os) const;

private:
  void export_features(const std::vector<std::pair<int, GameRecord>> &indexed,
                       const std::string &game_feature_out,
                       const std::string &turn_feature_out) const;

  void merge_batch_stats(const std::vector<std::pair<int, GameRecord>> &sorted);
  void update_anomaly_candidates(
      const std::vector<std::pair<int, GameRecord>> &sorted);
  void rebuild_anomaly_indexed_list();

  std::shared_ptr<PlayerFactory> _player_factory_p0;
  std::shared_ptr<PlayerFactory> _player_factory_p1;
  int _num_games;
  std::string _output_path;
  int _num_threads;
  std::string _log_path;
  unsigned int _rng_seed;

  std::vector<std::shared_ptr<FeatureExtractor>> _game_level_features;
  std::vector<std::shared_ptr<FeatureExtractor>> _turn_level_features;

  std::vector<GameRecord> _records;
  std::atomic<int> _games_completed{0};

  // Aggregates (filled across batches).
  int _p0_wins = 0;
  long _total_turns = 0;
  long _tb_total = 0;
  long _tb_case1 = 0;
  long _tb_case2 = 0;
  long long _tb_case1_seq_sum = 0;
  int _tb_case1_seq_max = 0;
  long _tb_case2_table_straight = 0;
  int _games_with_tb = 0;
  int _games_with_case1 = 0;
  int _games_with_case2 = 0;

  struct FeatureAgg {
    double sum = 0;
    int mn = INT_MAX, mx = INT_MIN;
    long count = 0;
    void add(int v) {
      sum += v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      ++count;
    }
    double mean() const { return count > 0 ? sum / count : 0.0; }
  };
  std::vector<FeatureAgg> _game_feat_agg;
  std::vector<FeatureAgg> _turn_feat_agg;

  std::optional<std::pair<int, GameRecord>> _best_long;
  std::optional<std::pair<int, GameRecord>> _best_short;
  std::optional<std::pair<int, GameRecord>> _best_start;
  std::optional<std::pair<int, GameRecord>> _worst_start;
  std::vector<std::pair<int, GameRecord>> _anomaly_indexed;

  long long _run_elapsed_ms = 0;

  GameRecord simulate_single_game(std::mt19937 &rng);
};

#endif
