// Compare greedy vs tree_greedy move choice at each non-tablebase decision.
//
// The game is advanced along either greedy's or the tree's chosen move (--advance)
// so the next position is well-defined; agreement stats count every non-TB turn.

#include "game.h"
#include "greedy/greedy_player.h"
#include "tablebase_peek.h"
#include "greedy/tree_evaluator.h"
#include "util.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <random>
#include <string>
#include <vector>

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --games <N>           Number of games (required)\n"
      << "  --seed <S>            RNG seed (default: random)\n"
      << "  --tree-model <path>   Tree text model (default: data/tree_model_d15.txt)\n"
      << "  --advance greedy|tree Which policy advances the game (default: greedy)\n"
      << "  --max-samples <K>     Print up to K disagreement samples (default: 8)\n"
      << "\n"
      << "For each non-tablebase turn, compares greedy_best vs tree greedy_best.\n"
      << "Skips turns where tablebase would move (same as Player::tablebase_move).\n";
}

static std::string hand_string(const std::array<int, 13> &h) {
  std::string s;
  for (int r = 0; r < 13; ++r) {
    for (int k = 0; k < h[r]; ++k)
      s += rankToChar(r + 3);
  }
  return s.empty() ? "-" : s;
}

int main(int argc, char **argv) {
  int num_games = 0;
  unsigned int seed = std::random_device{}();
  std::string tree_path = "data/tree_model_d15.txt";
  bool advance_greedy = true;
  int max_samples = 8;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--games" && i + 1 < argc)
      num_games = std::stoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc)
      seed = static_cast<unsigned int>(std::stoul(argv[++i]));
    else if (a == "--tree-model" && i + 1 < argc)
      tree_path = argv[++i];
    else if (a == "--advance" && i + 1 < argc) {
      std::string v = argv[++i];
      if (v == "greedy")
        advance_greedy = true;
      else if (v == "tree")
        advance_greedy = false;
      else {
        std::cerr << "Bad --advance (use greedy or tree)\n";
        return 1;
      }
    } else if (a == "--max-samples" && i + 1 < argc)
      max_samples = std::stoi(argv[++i]);
    else if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (num_games <= 0) {
    std::cerr << "Error: --games required\n";
    print_usage(argv[0]);
    return 1;
  }

  TreeEvaluator tree;
  try {
    tree.load(tree_path);
  } catch (const std::exception &e) {
    std::cerr << "Failed to load tree model: " << e.what() << "\n";
    return 1;
  }

  std::mt19937 rng(seed);

  long long total_compared = 0;
  long long agree = 0;
  long long tb_skipped = 0;

  struct Sample {
    int game_idx;
    int turn_idx;
    int player;
    std::string hand;
    int greedy_id;
    int tree_id;
    std::string greedy_desc;
    std::string tree_desc;
  };
  std::vector<Sample> samples;
  samples.reserve(static_cast<size_t>(max_samples));

  for (int g = 0; g < num_games; ++g) {
    Game game;
    game.shuffle_deal(rng);
    int turn_idx = 0;

    while (!game.is_over()) {
      int cp = game.current_player();
      PartialGame pg(game, cp);

      TablebasePeekResult tb = peek_tablebase_move(pg);
      if (tb.move) {
        ++tb_skipped;
        game.apply_move(*tb.move);
        ++turn_idx;
        continue;
      }

      std::vector<int> legal = pg.get_legal_moves();
      Move greedy_m = greedy_best(pg, legal, greedy_hand_eval);
      Move tree_m = greedy_best(pg, legal, [&](const PartialGame &sim) {
        return tree.predict(sim);
      });

      int gid = encodeMove(greedy_m);
      int tid = encodeMove(tree_m);
      ++total_compared;
      if (gid == tid)
        ++agree;
      else if (static_cast<int>(samples.size()) < max_samples) {
        Sample s;
        s.game_idx = g;
        s.turn_idx = turn_idx;
        s.player = cp;
        s.hand = hand_string(pg.player_hand());
        s.greedy_id = gid;
        s.tree_id = tid;
        {
          std::ostringstream os;
          os << greedy_m;
          s.greedy_desc = os.str();
        }
        {
          std::ostringstream os;
          os << tree_m;
          s.tree_desc = os.str();
        }
        samples.push_back(std::move(s));
      }

      Move adv = advance_greedy ? greedy_m : tree_m;
      game.apply_move(adv);
      ++turn_idx;
    }
  }

  long long disagree = total_compared - agree;
  double pct_agree =
      total_compared > 0 ? 100.0 * static_cast<double>(agree) / total_compared : 0.0;

  std::cout << "=== move_agreement ===\n";
  std::cout << "seed=" << seed << "  games=" << num_games << "\n";
  std::cout << "tree_model=" << tree_path << "\n";
  std::cout << "advance=" << (advance_greedy ? "greedy" : "tree") << "\n";
  std::cout << "tablebase_turns_skipped=" << tb_skipped << "\n";
  std::cout << "non_tb_decisions=" << total_compared << "\n";
  std::cout << "agree=" << agree << " (" << pct_agree << "%)\n";
  std::cout << "disagree=" << disagree;
  if (total_compared > 0)
    std::cout << " (" << (100.0 * disagree / total_compared) << "%)";
  std::cout << "\n\n";

  std::cout << "Sample disagreements (first " << samples.size() << "):\n";
  for (size_t i = 0; i < samples.size(); ++i) {
    const Sample &s = samples[i];
    std::cout << "--- sample " << (i + 1) << " (game " << s.game_idx
              << ", turn " << s.turn_idx << ", P" << s.player << ") ---\n";
    std::cout << "  hand: " << s.hand << "\n";
    std::cout << "  greedy: move_id=" << s.greedy_id << "  " << s.greedy_desc << "\n";
    std::cout << "  tree:   move_id=" << s.tree_id << "  " << s.tree_desc << "\n";
  }

  return 0;
}
