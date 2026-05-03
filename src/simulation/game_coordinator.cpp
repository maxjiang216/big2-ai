#include "game_coordinator.h"

#include "game_simulator.h"
#include "parquet_export.hpp"
#include "game_record.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

GameCoordinator::GameCoordinator(
    std::shared_ptr<PlayerFactory> player_factory_p0,
    std::shared_ptr<PlayerFactory> player_factory_p1, int num_games,
    const std::string &output_path, int num_threads, unsigned int random_seed,
    const std::string &log_path,
    std::vector<std::shared_ptr<FeatureExtractor>> game_level_features,
    std::vector<std::shared_ptr<FeatureExtractor>> turn_level_features)
    : _player_factory_p0(std::move(player_factory_p0)),
      _player_factory_p1(std::move(player_factory_p1)), _num_games(num_games),
      _output_path(output_path), _num_threads(std::max(1, num_threads)),
      _log_path(log_path), _rng_seed(random_seed),
      _game_level_features(std::move(game_level_features)),
      _turn_level_features(std::move(turn_level_features)) {
  _game_feat_agg.resize(_game_level_features.size());
  _turn_feat_agg.resize(_turn_level_features.size());
}

static int start_legal_moves_count(const GameRecord &rec) {
  const auto &turns = rec.turns();
  if (turns.empty())
    return -1;
  return static_cast<int>(turns[0].legal_moves.size());
}

void GameCoordinator::merge_batch_stats(
    const std::vector<std::pair<int, GameRecord>> &sorted) {
  for (const auto &pr : sorted) {
    const auto &rec = pr.second;
    if (rec.game().get_winner() == 0)
      ++_p0_wins;
    _total_turns += static_cast<long>(rec.turns().size());
    auto s = tablebase_first_hit_stats(rec);
    int seg = s.case1 + s.case2;
    _tb_total += seg;
    if (seg > 0)
      ++_games_with_tb;
    _tb_case1 += s.case1;
    _tb_case2 += s.case2;
    if (s.case1 > 0)
      ++_games_with_case1;
    if (s.case2 > 0)
      ++_games_with_case2;
    _tb_case1_seq_sum += s.case1_seq_sum;
    if (s.case1_seq_max > _tb_case1_seq_max)
      _tb_case1_seq_max = s.case1_seq_max;
    _tb_case2_table_straight += s.case2_table_straight;
  }

  for (size_t fi = 0; fi < _game_level_features.size(); ++fi) {
    if (!_game_level_features[fi])
      continue;
    for (const auto &pr : sorted)
      _game_feat_agg[fi].add(_game_level_features[fi]->gameExtract(pr.second));
  }

  for (size_t fi = 0; fi < _turn_level_features.size(); ++fi) {
    if (!_turn_level_features[fi])
      continue;
    for (const auto &pr : sorted) {
      for (int v : _turn_level_features[fi]->turnExtract(pr.second))
        _turn_feat_agg[fi].add(v);
    }
  }
}

void GameCoordinator::update_anomaly_candidates(
    const std::vector<std::pair<int, GameRecord>> &sorted) {
  for (const auto &pr : sorted) {
    const GameRecord &rec = pr.second;
    int L = static_cast<int>(rec.turns().size());
    int sl = start_legal_moves_count(rec);

    if (!_best_long || L > static_cast<int>(_best_long->second.turns().size()))
      _best_long = pr;

    if (!_best_short || L < static_cast<int>(_best_short->second.turns().size()))
      _best_short = pr;

    if (sl >= 0) {
      if (!_best_start ||
          sl > start_legal_moves_count(_best_start->second))
        _best_start = pr;
      if (!_worst_start ||
          sl < start_legal_moves_count(_worst_start->second))
        _worst_start = pr;
    }
  }
}

void GameCoordinator::rebuild_anomaly_indexed_list() {
  _anomaly_indexed.clear();
  auto push_unique = [&](const std::optional<std::pair<int, GameRecord>> &opt) {
    if (!opt)
      return;
    int gi = opt->first;
    for (const auto &p : _anomaly_indexed) {
      if (p.first == gi)
        return;
    }
    _anomaly_indexed.push_back(*opt);
  };
  push_unique(_best_long);
  push_unique(_best_short);
  push_unique(_best_start);
  push_unique(_worst_start);
}

void GameCoordinator::print_run_summary(std::ostream &os) const {
  int num_games = _num_games;
  if (num_games <= 0)
    return;
  int p1_wins = num_games - _p0_wins;
  double avg_turns =
      _total_turns > 0 ? static_cast<double>(_total_turns) / num_games : 0.0;

  os << "\n"
     << "P0 wins: " << _p0_wins << " (" << 100.0 * _p0_wins / num_games << "%)"
     << "   P1 wins: " << p1_wins << " (" << 100.0 * p1_wins / num_games
     << "%)\n"
     << "Avg game length: " << avg_turns << " turns\n";

  if (!_game_level_features.empty()) {
    os << "\n--- Game features ---\n";
    for (size_t fi = 0; fi < _game_level_features.size(); ++fi) {
      if (!_game_level_features[fi])
        continue;
      const auto &agg = _game_feat_agg[fi];
      os << "  " << _game_level_features[fi]->name() << ": mean=" << agg.mean()
         << "  min=" << agg.mn << "  max=" << agg.mx << "\n";
    }
  }

  if (!_turn_level_features.empty()) {
    os << "\n--- Turn features (mean over all turns x perspectives) ---\n";
    for (size_t fi = 0; fi < _turn_level_features.size(); ++fi) {
      if (!_turn_level_features[fi])
        continue;
      const auto &agg = _turn_feat_agg[fi];
      os << "  " << _turn_level_features[fi]->name() << ": mean=" << agg.mean()
         << "  min=" << agg.mn << "  max=" << agg.mx << "\n";
    }
  }

  os << "\n--- Tablebase usage (game-level first-hit segments) ---\n";
  os << "  Games with ≥1 TB segment: " << _games_with_tb << " / " << num_games
     << " (" << 100.0 * _games_with_tb / num_games << "%)\n"
     << "  Total TB segments:        " << _tb_total << "  (avg "
     << static_cast<double>(_tb_total) / num_games << " per game)\n"
     << "    case1 (forced win seq): " << _tb_case1 << " segments in "
     << _games_with_case1 << " games";
  if (_tb_case1 > 0) {
    double mean_len = static_cast<double>(_tb_case1_seq_sum) /
                      static_cast<double>(_tb_case1);
    os << "  mean seq len=" << mean_len << "  max=" << _tb_case1_seq_max;
  }
  os << "\n"
     << "    case2 (opp has 1 card): " << _tb_case2 << " segments in "
     << _games_with_case2 << " games";
  if (_tb_case2 > 0) {
    os << "  table-straight: " << _tb_case2_table_straight << " ("
       << 100.0 * _tb_case2_table_straight / _tb_case2 << "% of case2 segments)";
  }
  os << "\n";

  double games_per_sec =
      _run_elapsed_ms > 0 ? 1000.0 * num_games / _run_elapsed_ms : 0.0;
  os << "\nElapsed: " << _run_elapsed_ms << " ms  (" << static_cast<long>(games_per_sec)
     << " games/s)\n"
     << "  simulation:          " << _sim_elapsed_ms << " ms\n"
     << "  feature extract:     " << _extract_elapsed_ms << " ms\n"
     << "  parquet write:       " << _parquet_elapsed_ms << " ms\n\n";
}

void GameCoordinator::export_features(
    const std::vector<std::pair<int, GameRecord>> &indexed,
    const std::string &game_feature_out,
    const std::string &turn_feature_out) const {
  export_parquet_features(indexed, game_feature_out, turn_feature_out,
                          _game_level_features, _turn_level_features);
}

void GameCoordinator::run_all(const std::string &game_feature_out,
                                const std::string &turn_feature_out) {
  const bool skip_parquet = game_feature_out.empty() && turn_feature_out.empty();

  _p0_wins = 0;
  _total_turns = 0;
  _tb_total = 0;
  _tb_case1 = 0;
  _tb_case2 = 0;
  _tb_case1_seq_sum = 0;
  _tb_case1_seq_max = 0;
  _tb_case2_table_straight = 0;
  _games_with_tb = 0;
  _games_with_case1 = 0;
  _games_with_case2 = 0;
  _run_elapsed_ms = 0;
  _sim_elapsed_ms = 0;
  _extract_elapsed_ms = 0;
  _parquet_elapsed_ms = 0;
  _best_long.reset();
  _best_short.reset();
  _best_start.reset();
  _worst_start.reset();
  _anomaly_indexed.clear();
  for (auto &a : _game_feat_agg)
    a = FeatureAgg{};
  for (auto &a : _turn_feat_agg)
    a = FeatureAgg{};

  _games_completed.store(0);
  auto t_start = std::chrono::high_resolution_clock::now();

  constexpr int BATCH_SIZE = 200'000;
  const std::string game_tmp_prefix = "tmp_game_batch_";
  const std::string turn_tmp_prefix = "tmp_turn_batch_";

  int games_remaining = _num_games;
  int batch_idx = 0;

  std::vector<std::string> tmp_game_files;
  std::vector<std::string> tmp_turn_files;

  while (games_remaining > 0) {

    int this_batch = std::min(BATCH_SIZE, games_remaining);
    int global_start = _num_games - games_remaining;

    std::atomic<int> next_index{0};
    std::vector<std::thread> workers;
    std::vector<std::vector<std::pair<int, GameRecord>>> local_batch(_num_threads);
    workers.reserve(_num_threads);

    auto t_sim0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < _num_threads; ++t) {
      workers.emplace_back([&, t]() {
        std::mt19937 rng(_rng_seed + static_cast<unsigned>(t) +
                         static_cast<unsigned>(batch_idx * _num_threads));
        auto &out = local_batch[t];
        int idx;
        while ((idx = next_index.fetch_add(1)) < this_batch) {
          GameRecord rec = simulate_single_game(rng);
          _games_completed.fetch_add(1);
          out.emplace_back(global_start + idx, std::move(rec));
        }
      });
    }
    for (auto &th : workers)
      if (th.joinable())
        th.join();
    _sim_elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_sim0).count();

    std::vector<std::pair<int, GameRecord>> sorted_batch;
    sorted_batch.reserve(static_cast<size_t>(this_batch));
    for (auto &vec : local_batch) {
      for (auto &p : vec)
        sorted_batch.push_back(std::move(p));
    }
    std::sort(sorted_batch.begin(), sorted_batch.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    auto t_ext0 = std::chrono::high_resolution_clock::now();
    merge_batch_stats(sorted_batch);
    update_anomaly_candidates(sorted_batch);
    _extract_elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t_ext0).count();

    if (!skip_parquet) {
      auto t_pq0 = std::chrono::high_resolution_clock::now();
      std::string gfile = game_tmp_prefix + std::to_string(batch_idx) + ".parquet";
      std::string tfile = turn_tmp_prefix + std::to_string(batch_idx) + ".parquet";
      export_features(sorted_batch, gfile, tfile);
      tmp_game_files.push_back(gfile);
      tmp_turn_files.push_back(tfile);
      _parquet_elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - t_pq0).count();
    }

    games_remaining -= this_batch;
    ++batch_idx;
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  _run_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
                        .count();

  rebuild_anomaly_indexed_list();

  if (!skip_parquet) {
    std::cout << "[Coordinator] Concatenating " << tmp_game_files.size()
              << " game batches...\n";
    concat_parquet_files(tmp_game_files, game_feature_out);

    std::cout << "[Coordinator] Concatenating " << tmp_turn_files.size()
              << " turn batches...\n";
    concat_parquet_files(tmp_turn_files, turn_feature_out);

    for (auto &f : tmp_game_files)
      std::filesystem::remove(f);
    for (auto &f : tmp_turn_files)
      std::filesystem::remove(f);

    std::cout
        << "[Coordinator] All batches complete — final Parquet files written.\n";
  }
}

GameRecord GameCoordinator::simulate_single_game(std::mt19937 &rng) {
  auto player0 = _player_factory_p0->create_player();
  auto player1 = _player_factory_p1->create_player();
  GameSimulator sim(std::move(player0), std::move(player1), rng, _log_path);
  return sim.run();
}
