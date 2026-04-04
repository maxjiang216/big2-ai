// Build pass-vs-greedy training CSV from self-play games (default: greedy vs greedy).
//
// For each turn where voluntary pass is legal and tablebase did not fire,
// writes 60 tree features (pre-move mover view), label y=1 if mean rollout win
// with pass > greedy (tie -> 0), and mean win rates p_pass, p_greedy.
//
// Usage:
//   pass_greedy_datagen --games 500 --dets 20 --seed 42 --output data/pass_train.csv
//   pass_greedy_datagen --player pimc --player-param 20 ...   # slower, PIMC lines

#include "game_record.h"
#include "game_simulator.h"
#include "greedy/greedy_player.h"
#include "move.h"
#include "pimc/pimc_player.h"
#include "player_factory_registry.h"
#include "greedy/tree_evaluator.h"
#include "util.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "  --games <N>        Self-play games (required)\n"
      << "  --player <name>    Self-play policy: greedy (default) or pimc\n"
      << "  --player-param <f> PIMC samples N when --player pimc (default 20)\n"
      << "  --dets <T>         Determinizations per label (default 20)\n"
      << "  --seed <S>         Base seed (default 42)\n"
      << "  --output <path>    Output CSV (required)\n"
      << "  --max-rows <M>     Stop after M labeled rows (optional)\n"
      << "  --resample-redet   Use resample_opponent_each_turn in rollouts\n"
      << "  --quiet            No stderr progress line\n";
}

struct Row {
  std::array<float, TREE_N_FEATURES> f{};
  int y = 0;
  double p_pass = 0.0;
  double p_greedy = 0.0;
  int game_index = 0;
  int turn_idx = 0;
};

// Returns false if turn skipped (no voluntary pass or TB).
static bool label_one_turn(const TurnRecord &turn, int turn_idx, int game_idx,
                           int num_dets, bool resample_redet, std::mt19937 &rng_det,
                           Row *out) {
  int cp = turn.current_player;
  if (turn.tb_case != -1)
    return false;
  if (!voluntary_pass_legal(turn.legal_moves))
    return false;

  const PartialGame &view = turn.views[cp];
  Move greedy_mv = greedy_best(view, turn.legal_moves, greedy_hand_eval);

  const std::array<int, 13> my_hand = view.player_hand();
  const std::array<int, 13> discard = view.discard_pile();
  const int opp_count = view.opponent_hand_size();
  const Move last_mv = view.last_move();

  int pass_wins = 0, greedy_wins = 0;
  for (int d = 0; d < num_dets; ++d) {
    const std::array<int, 13> opp_hand =
        sample_opponent_hand(my_hand, discard, opp_count, rng_det);

    const std::array<int, 13> hand0 = (cp == 0) ? my_hand : opp_hand;
    const std::array<int, 13> hand1 = (cp == 0) ? opp_hand : my_hand;

    Game g_pass(hand0, hand1, discard, last_mv, cp);
    g_pass.apply_move(Move(kPASS));
    pass_wins += policy_rollout(g_pass, cp, greedy_hand_eval, resample_redet, rng_det);

    Game g_gr(hand0, hand1, discard, last_mv, cp);
    g_gr.apply_move(greedy_mv);
    greedy_wins +=
        policy_rollout(g_gr, cp, greedy_hand_eval, resample_redet, rng_det);
  }

  double pp = static_cast<double>(pass_wins) / static_cast<double>(num_dets);
  double p_gr = static_cast<double>(greedy_wins) / static_cast<double>(num_dets);
  int y = (pp > p_gr) ? 1 : 0;

  out->f = extract_tree_features(view);
  out->y = y;
  out->p_pass = pp;
  out->p_greedy = p_gr;
  out->game_index = game_idx;
  out->turn_idx = turn_idx;
  return true;
}

static void write_csv_header(std::ostream &os) {
  os << "y,p_pass,p_greedy,game_index,turn_idx";
  for (int i = 0; i < TREE_N_FEATURES; ++i)
    os << ",f" << i;
  os << "\n";
}

static void write_csv_row(std::ostream &os, const Row &r) {
  os << r.y << "," << std::setprecision(9) << r.p_pass << "," << r.p_greedy << ","
     << r.game_index << "," << r.turn_idx;
  for (int i = 0; i < TREE_N_FEATURES; ++i)
    os << "," << std::setprecision(9) << static_cast<double>(r.f[i]);
  os << "\n";
}

} // namespace

int main(int argc, char **argv) {
  int num_games = 0;
  int num_dets = 20;
  unsigned int seed = 42;
  std::string out_path;
  std::string player_name = "greedy";
  double player_param = 20.0;
  int max_rows = -1;
  bool resample_redet = false;
  bool quiet = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--games" && i + 1 < argc)
      num_games = std::stoi(argv[++i]);
    else if (arg == "--dets" && i + 1 < argc)
      num_dets = std::stoi(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      seed = static_cast<unsigned int>(std::stoul(argv[++i]));
    else if (arg == "--output" && i + 1 < argc)
      out_path = argv[++i];
    else if (arg == "--player" && i + 1 < argc)
      player_name = argv[++i];
    else if (arg == "--player-param" && i + 1 < argc)
      player_param = std::stod(argv[++i]);
    else if (arg == "--max-rows" && i + 1 < argc)
      max_rows = std::stoi(argv[++i]);
    else if (arg == "--resample-redet")
      resample_redet = true;
    else if (arg == "--quiet")
      quiet = true;
    else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (num_games <= 0 || out_path.empty()) {
    std::cerr << "Error: --games and --output are required\n";
    print_usage(argv[0]);
    return 1;
  }

  if (player_name != "greedy" && player_name != "pimc") {
    std::cerr << "Error: --player must be greedy or pimc\n";
    return 1;
  }

  auto factory = make_player_factory(player_name, player_param, seed);
  if (!factory) {
    std::cerr << "Error: could not create player factory for '" << player_name << "'\n";
    return 1;
  }

  std::vector<Row> rows;
  rows.reserve(static_cast<size_t>(num_games) * 8);

  std::atomic<int> games_done{0};
  std::atomic<long long> labeled_rows{0};
  std::atomic<bool> progress_done{false};

  auto t0 = std::chrono::steady_clock::now();

  std::thread progress_thread;
  if (!quiet) {
    progress_thread = std::thread([&, player_name]() {
      while (!progress_done.load()) {
        int g = games_done.load();
        long long r = labeled_rows.load();
        auto now = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration<double>(now - t0).count();
        double g_per_s = elapsed > 0 ? static_cast<double>(g) / elapsed : 0.0;
        int pct = num_games > 0 ? static_cast<int>(100.0 * g / num_games) : 0;
        std::fprintf(stderr,
                     "\r[pass_greedy_datagen] %s  games %d/%d (%d%%)  rows %lld  "
                     "%.1f games/s  %.0fs   ",
                     player_name.c_str(), g, num_games, pct, r, g_per_s, elapsed);
        std::fflush(stderr);
        if (g >= num_games)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    });
  }

  for (int gi = 0; gi < num_games; ++gi) {
    if (max_rows >= 0 && static_cast<int>(rows.size()) >= max_rows)
      break;

    std::mt19937 rng(seed + static_cast<unsigned int>(gi) * 7919u);
    auto p0 = factory->create_player();
    auto p1 = factory->create_player();
    GameSimulator sim(std::move(p0), std::move(p1), rng);
    GameRecord rec = sim.run();

    const auto &turns = rec.turns();
    for (size_t ti = 0; ti < turns.size(); ++ti) {
      if (max_rows >= 0 && static_cast<int>(rows.size()) >= max_rows)
        break;
      Row r{};
      std::mt19937 rng_det(seed + static_cast<unsigned int>(gi) * 10007u +
                           static_cast<unsigned int>(ti) * 17u + 13u);
      if (!label_one_turn(turns[ti], static_cast<int>(ti), gi, num_dets, resample_redet,
                          rng_det, &r))
        continue;
      rows.push_back(r);
      labeled_rows.store(static_cast<long long>(rows.size()));
    }
    games_done.store(gi + 1);
    labeled_rows.store(static_cast<long long>(rows.size()));
    if (max_rows >= 0 && static_cast<int>(rows.size()) >= max_rows)
      break;
  }

  progress_done = true;
  if (progress_thread.joinable())
    progress_thread.join();
  if (!quiet)
    std::fprintf(stderr, "\n");

  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "Error: cannot open " << out_path << "\n";
    return 1;
  }
  write_csv_header(out);
  for (const Row &r : rows)
    write_csv_row(out, r);
  out.close();

  std::cout << "Wrote " << rows.size() << " rows (" << player_name
            << " self-play) to " << out_path << " in " << std::fixed
            << std::setprecision(2) << sec << " s\n";
  return 0;
}
