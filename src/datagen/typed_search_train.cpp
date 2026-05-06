// typed_search_train: in-process self-play that updates the four tabular
// files used by the typed-search player.
//
// Usage:
//   typed_search_train [options]
//     --policy {random|typed_search}   default: random for gen 0, typed_search otherwise
//     --games N                        number of games (default 10000)
//     --threads T                      parallelism (default hardware concurrency)
//     --alpha A                        decay applied to existing tables (default 0.7)
//     --seed S                         RNG seed for shuffles (default time-based)
//     --tables-dir D                   default "data/typed_search"
//     --eval-min-visits V              not used at train time; for query-side cfg
//
// Reads tables from D/{eval_main,eval_fallback,mp_main,mp_fallback}.bin
// (silently empty if absent). Applies decay, runs games with the chosen
// policy as both seats, accumulates new counts, writes updated tables.

#include "game.h"
#include "game_record.h"
#include "game_simulator.h"
#include "partial_game.h"
#include "player_factory.h"
#include "random/random_player_factory.h"
#include "typed_search/eval_features.h"
#include "typed_search/eval_table.h"
#include "typed_search/move_prob_table.h"
#include "typed_search/typed_search_player.h"
#include "typed_search/typed_search_player_factory.h"
#include "util.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
  std::string policy;  // "random" | "typed_search" | auto
  int games = 10000;
  int threads = 0;
  float alpha = 0.7f;
  std::uint64_t seed = 0;
  std::string tables_dir = "data/typed_search";
  // Optional: after the main run, play `sample_games` extra typed_search
  // self-play games with verbose annotated logs to `sample_dir`/game_NNNN.log
  // (top-10 moves with values per turn). Also writes a per-decision stats
  // CSV to `stats_csv` if non-empty.
  int sample_games = 0;
  std::string sample_dir;
  std::string stats_csv;
};

Args parse_args(int argc, char *argv[]) {
  Args a;
  a.seed = static_cast<std::uint64_t>(
      std::chrono::system_clock::now().time_since_epoch().count());
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char *flag) -> std::string {
      if (++i >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        std::exit(2);
      }
      return argv[i];
    };
    if (s == "--policy") a.policy = next("--policy");
    else if (s == "--games") a.games = std::stoi(next("--games"));
    else if (s == "--threads") a.threads = std::stoi(next("--threads"));
    else if (s == "--alpha") a.alpha = std::stof(next("--alpha"));
    else if (s == "--seed") a.seed = static_cast<std::uint64_t>(std::stoll(next("--seed")));
    else if (s == "--tables-dir") a.tables_dir = next("--tables-dir");
    else if (s == "--sample-games") a.sample_games = std::stoi(next("--sample-games"));
    else if (s == "--sample-dir") a.sample_dir = next("--sample-dir");
    else if (s == "--stats-csv") a.stats_csv = next("--stats-csv");
    else {
      std::cerr << "Unknown flag: " << s << "\n";
      std::exit(2);
    }
  }
  if (a.threads <= 0) {
    a.threads = static_cast<int>(std::thread::hardware_concurrency());
    if (a.threads <= 0) a.threads = 1;
  }
  return a;
}

// Per-thread accumulators (no contention).
struct EvalDelta {
  std::unordered_map<uint32_t, typed_search::EvalTable::Entry> main_;
  std::unordered_map<uint32_t, typed_search::EvalTable::Entry> fb_;
  void add_main(uint32_t sid, float winner, float w) {
    auto &e = main_[sid];
    e.total_wins += winner * w;
    e.visit_count += w;
  }
  void add_fb(uint32_t sid, float winner, float w) {
    auto &e = fb_[sid];
    e.total_wins += winner * w;
    e.visit_count += w;
  }
};

struct MpDelta {
  // (player_move, opp_count) -> 468 floats
  std::unordered_map<std::uint32_t, std::array<float, 468>> main_;
  std::unordered_map<std::uint32_t, std::array<float, 468>> fb_;
  static std::uint32_t main_key(int pm, int oc) {
    return static_cast<std::uint32_t>(pm) * 17u + static_cast<std::uint32_t>(oc);
  }
  static std::uint32_t fb_key(int pm) { return static_cast<std::uint32_t>(pm); }
  void add(int pm, int oc, int resp, float w) {
    if (resp < 0 || resp >= 468) return;
    main_[main_key(pm, oc)][resp] += w;
    fb_[fb_key(pm)][resp] += w;
  }
};

void process_game(const GameRecord &rec, int winner_seat, EvalDelta &eval_delta,
                  MpDelta &mp_delta) {
  const auto &turns = rec.turns();
  for (std::size_t i = 0; i < turns.size(); ++i) {
    const auto &t = turns[i];
    // Eval-table records at trick boundaries: when this turn's move was PASS,
    // the START of next turn (which equals END of this turn) is a leaf state.
    if (t.move.combination == Move::Combination::kPass &&
        i + 1 < turns.size()) {
      const auto &next = turns[i + 1];
      // Both players' views in `next` are the post-PASS leaf state.
      // For player P: initiative = (P == next.current_player) ? 0 : 1.
      for (int p = 0; p < 2; ++p) {
        const PartialGame &v = next.views[p];
        typed_search::LeafContext ctx{
            v.player_hand(),
            v.discard_pile(),
            v.opponent_hand_size(),
            (p == next.current_player) ? 0 : 1,
        };
        float winner = (p == winner_seat) ? 1.0f : 0.0f;
        uint32_t mid = typed_search::main_state_id(ctx);
        uint32_t fid = typed_search::fallback_state_id(ctx);
        eval_delta.add_main(mid, winner, 1.0f);
        eval_delta.add_fb(fid, winner, 1.0f);
      }
    }

    // Move-prob records: at every (non-pass move, response) consecutive pair.
    if (t.move.combination != Move::Combination::kPass &&
        i + 1 < turns.size()) {
      const auto &next = turns[i + 1];
      int player_move = encodeMove(t.move);
      int opp_count = t.views[t.current_player].opponent_hand_size();
      // Adjust opp_count for what opp has when responding: opp's current hand
      // before they move is the same as before our move (we don't change opp).
      // (no change needed)
      int response = encodeMove(next.move);
      mp_delta.add(player_move, opp_count, response, 1.0f);
    }
  }
}

void merge_deltas(typed_search::EvalTable &main_e, typed_search::EvalTable &fb_e,
                   typed_search::MoveProbTable &main_m,
                   typed_search::MoveProbTable &fb_m,
                   const std::vector<EvalDelta> &eds,
                   const std::vector<MpDelta> &mds) {
  for (const auto &ed : eds) {
    for (const auto &kv : ed.main_) {
      // add_observation(sid, winner, weight) does:
      //   total_wins += winner * weight; visit_count += weight.
      // We pass weight = vc and winner = tw/vc to land exactly delta totals.
      float vc = kv.second.visit_count;
      float winner = (vc > 0) ? (kv.second.total_wins / vc) : 0.0f;
      main_e.add_observation(kv.first, winner, vc);
    }
    for (const auto &kv : ed.fb_) {
      float vc = kv.second.visit_count;
      float winner = (vc > 0) ? (kv.second.total_wins / vc) : 0.0f;
      fb_e.add_observation(kv.first, winner, vc);
    }
  }
  for (const auto &md : mds) {
    for (const auto &kv : md.main_) {
      int pm = static_cast<int>(kv.first / 17u);
      int oc = static_cast<int>(kv.first % 17u);
      const auto &vec = kv.second;
      for (int m = 0; m < 468; ++m) {
        if (vec[m] != 0.0f) main_m.add_observation(pm, oc, m, vec[m]);
      }
    }
    for (const auto &kv : md.fb_) {
      int pm = static_cast<int>(kv.first);
      const auto &vec = kv.second;
      for (int m = 0; m < 468; ++m) {
        if (vec[m] != 0.0f) fb_m.add_observation(pm, /*opp_count=*/0, m, vec[m]);
      }
    }
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  Args args = parse_args(argc, argv);

  std::filesystem::create_directories(args.tables_dir);
  const std::string p_eval_main = args.tables_dir + "/eval_main.bin";
  const std::string p_eval_fb = args.tables_dir + "/eval_fallback.bin";
  const std::string p_mp_main = args.tables_dir + "/mp_main.bin";
  const std::string p_mp_fb = args.tables_dir + "/mp_fallback.bin";

  // Load + decay.
  typed_search::EvalTable eval_main, eval_fb;
  typed_search::MoveProbTable mp_main, mp_fb;
  mp_main.set_ignore_opp_count(false);
  mp_fb.set_ignore_opp_count(true);
  eval_main.load(p_eval_main);
  eval_fb.load(p_eval_fb);
  mp_main.load(p_mp_main);
  mp_fb.load(p_mp_fb);

  std::cout << "Loaded eval_main=" << eval_main.size()
            << " eval_fb=" << eval_fb.size() << " mp_main=" << mp_main.size()
            << " mp_fb=" << mp_fb.size() << "\n";

  if (args.alpha < 1.0f) {
    eval_main.decay(args.alpha);
    eval_fb.decay(args.alpha);
    mp_main.decay(args.alpha);
    mp_fb.decay(args.alpha);
    std::cout << "Decayed by alpha=" << args.alpha << "\n";
  }

  // Determine policy.
  std::string policy = args.policy;
  if (policy.empty()) {
    policy = (eval_fb.size() == 0 && mp_main.size() == 0) ? "random" : "typed_search";
  }
  std::cout << "Policy: " << policy << "\n";

  // Build a shared TypedSearchTables for the typed_search policy. We share the
  // tables we just decayed (post-decay snapshot is what gen N+1 plays with).
  std::shared_ptr<typed_search::TypedSearchTables> ts_tables;
  if (policy == "typed_search") {
    ts_tables = std::make_shared<typed_search::TypedSearchTables>();
    // Copy entries via save/load round-trip (simple, avoids exposing internals).
    eval_main.save(p_eval_main + ".tmp");
    eval_fb.save(p_eval_fb + ".tmp");
    mp_main.save(p_mp_main + ".tmp");
    mp_fb.save(p_mp_fb + ".tmp");
    ts_tables->eval_main.load(p_eval_main + ".tmp");
    ts_tables->eval_fallback.load(p_eval_fb + ".tmp");
    ts_tables->mp_main.load(p_mp_main + ".tmp");
    ts_tables->mp_fallback.load(p_mp_fb + ".tmp");
    std::filesystem::remove(p_eval_main + ".tmp");
    std::filesystem::remove(p_eval_fb + ".tmp");
    std::filesystem::remove(p_mp_main + ".tmp");
    std::filesystem::remove(p_mp_fb + ".tmp");
  }

  // Run games.
  std::vector<EvalDelta> eds(args.threads);
  std::vector<MpDelta> mds(args.threads);
  std::atomic<int> games_played{0};
  std::atomic<int> p0_wins{0};

  auto worker = [&](int tid) {
    std::mt19937 rng(args.seed + 0x9E3779B97F4A7C15ull * (tid + 1));
    while (true) {
      int idx = games_played.fetch_add(1);
      if (idx >= args.games) break;

      std::unique_ptr<Player> p0, p1;
      if (policy == "random") {
        RandomPlayerFactory f(static_cast<unsigned int>(rng()));
        p0 = f.create_player();
        p1 = f.create_player();
      } else {
        typed_search::TypedSearchPlayer *tp0 = new typed_search::TypedSearchPlayer(
            ts_tables, static_cast<std::uint64_t>(rng()));
        typed_search::TypedSearchPlayer *tp1 = new typed_search::TypedSearchPlayer(
            ts_tables, static_cast<std::uint64_t>(rng()));
        p0.reset(tp0);
        p1.reset(tp1);
      }
      GameSimulator sim(std::move(p0), std::move(p1), rng);
      GameRecord rec = sim.run();
      // Determine winner: in the Game model, winner is the player who emptied
      // their hand (i.e., the player whose hand_size hit 0).
      // After last move, _game.is_over() and current_player is the *next* to
      // move; the previous mover won. The last turn's current_player is the
      // winner.
      int winner = rec.turns().empty() ? 0
                                        : rec.turns().back().current_player;
      if (winner == 0) p0_wins.fetch_add(1);
      process_game(rec, winner, eds[tid], mds[tid]);
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  threads.reserve(args.threads);
  for (int t = 0; t < args.threads; ++t) threads.emplace_back(worker, t);
  for (auto &t : threads) t.join();
  auto t1 = std::chrono::steady_clock::now();

  // Print live-query hit rates from the play-time tables (only if we played
  // typed_search; the random policy doesn't query any tables).
  if (ts_tables) {
    auto pct = [](std::uint64_t n, std::uint64_t d) {
      return d == 0 ? 0.0 : (100.0 * static_cast<double>(n) / d);
    };
    auto &es = ts_tables->eval_main.stats();
    std::uint64_t eq = es.queries.load();
    std::uint64_t em = es.main_hits.load();
    std::uint64_t ef = es.fallback_hits.load();
    std::uint64_t ed = es.defaults.load();
    std::cout << "Eval-table query hit rates: total=" << eq
              << " main=" << em << " (" << pct(em, eq)
              << "%) fallback=" << ef << " (" << pct(ef, eq)
              << "%) default=" << ed << " (" << pct(ed, eq) << "%)\n";

    auto &ms = ts_tables->mp_main.stats();
    std::uint64_t mq = ms.queries.load();
    std::uint64_t mm = ms.main_hits.load();
    std::uint64_t mf = ms.fallback_hits.load();
    std::uint64_t mu = ms.uniform.load();
    std::cout << "MoveProb query hit rates:   total=" << mq
              << " main=" << mm << " (" << pct(mm, mq)
              << "%) fallback=" << mf << " (" << pct(mf, mq)
              << "%) uniform=" << mu << " (" << pct(mu, mq) << "%)\n";
  }

  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "Played " << args.games << " games in " << secs << "s ("
            << (args.games / secs) << " g/s); P0 win rate "
            << (100.0 * p0_wins.load() / args.games) << "%\n";

  // Merge deltas into the (already decayed) tables.
  merge_deltas(eval_main, eval_fb, mp_main, mp_fb, eds, mds);

  std::cout << "Post-merge: eval_main=" << eval_main.size()
            << " eval_fb=" << eval_fb.size() << " mp_main=" << mp_main.size()
            << " mp_fb=" << mp_fb.size() << "\n";

  // Visit-count histograms for the eval tables.
  auto eval_histogram = [](const typed_search::EvalTable &t,
                           const char *label, std::size_t total_states) {
    int b1 = 0, b5 = 0, b20 = 0, b100 = 0;
    double sum_v = 0.0;
    for (const auto &kv : t.entries()) {
      float v = kv.second.visit_count;
      sum_v += v;
      if (v >= 1) ++b1;
      if (v >= 5) ++b5;
      if (v >= 20) ++b20;
      if (v >= 100) ++b100;
    }
    std::cout << label << ": " << t.size() << " / " << total_states
              << " states (" << (100.0 * t.size() / total_states) << "%); "
              << "≥1=" << b1 << " ≥5=" << b5 << " ≥20=" << b20
              << " ≥100=" << b100 << "; total visits=" << sum_v << "\n";
  };
  eval_histogram(eval_main, "  eval_main", typed_search::kMainStateCount);
  eval_histogram(eval_fb, "  eval_fb  ", typed_search::kFallbackStateCount);

  auto mp_histogram = [](const typed_search::MoveProbTable &t,
                         const char *label, std::size_t total_states) {
    // We can't directly iterate MoveProbTable entries (private). Use
    // round-trip via save+load to a temp file? Simpler: just print size and
    // skip per-entry stats for now.
    (void)t; (void)label; (void)total_states;
  };
  mp_histogram(mp_main, "  mp_main  ", 468 * 17);
  mp_histogram(mp_fb, "  mp_fb   ", 468);
  std::cout << "  mp_main  : " << mp_main.size() << " / " << (468 * 17)
            << " states (" << (100.0 * mp_main.size() / (468.0 * 17.0))
            << "%)\n";
  std::cout << "  mp_fb    : " << mp_fb.size() << " / " << 468 << " states ("
            << (100.0 * mp_fb.size() / 468.0) << "%)\n";

  eval_main.save(p_eval_main);
  eval_fb.save(p_eval_fb);
  mp_main.save(p_mp_main);
  mp_fb.save(p_mp_fb);
  std::cout << "Saved to " << args.tables_dir << "\n";

  // Optionally play extra games with annotated logs (top-N move values per
  // turn) and write per-decision stats to a CSV for distribution analysis.
  if (args.sample_games > 0 || !args.stats_csv.empty()) {
    auto sample_tables = std::make_shared<typed_search::TypedSearchTables>();
    sample_tables->eval_main.load(p_eval_main);
    sample_tables->eval_fallback.load(p_eval_fb);
    sample_tables->mp_main.load(p_mp_main);
    sample_tables->mp_fallback.load(p_mp_fb);

    std::ofstream stats;
    if (!args.stats_csv.empty()) {
      stats.open(args.stats_csv);
      stats << "game_idx,turn_idx,player,tb_case,last_combo,last_rank,"
               "our_hand_size,opp_size,nodes_searched,n_legal,top1_move,"
               "top1_value,chosen_move,chosen_value\n";
    }
    if (args.sample_games > 0 && !args.sample_dir.empty()) {
      std::filesystem::create_directories(args.sample_dir);
    }

    auto fmt_move = [](const Move &m) {
      std::ostringstream os;
      os << m;
      return os.str();
    };
    auto fmt_hand = [](const std::array<int, 13> &h) {
      // rankToChar takes face values (3..14, or 15/2 for the deuce); convert
      // rank indices accordingly (matches the pattern in game.cpp).
      std::string s = "[";
      for (int r = 0; r < 13; ++r) {
        for (int k = 0; k < h[r]; ++k) s += rankToChar(r + 3);
      }
      s += "]";
      return s;
    };

    std::cout << "Sample/stats run: games=" << args.sample_games
              << " stats=" << (args.stats_csv.empty() ? "none" : args.stats_csv)
              << "\n";
    std::mt19937 rng(args.seed + 0xCAFEBABEull);
    for (int gi = 0; gi < std::max(args.sample_games, 1); ++gi) {
      auto p0 = std::make_unique<typed_search::TypedSearchPlayer>(
          sample_tables, static_cast<std::uint64_t>(rng()));
      auto p1 = std::make_unique<typed_search::TypedSearchPlayer>(
          sample_tables, static_cast<std::uint64_t>(rng()));
      Game game;
      game.shuffle_deal(rng);
      p0->accept_deal(game, 0);
      p1->accept_deal(game, 1);

      std::ofstream log;
      bool log_this = (gi < args.sample_games) && !args.sample_dir.empty();
      if (log_this) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%04d", gi);
        log.open(args.sample_dir + "/game_" + buf + ".log");
        log << "=== Game " << gi << " ===\n";
        log << "P0 hand: " << fmt_hand(game.player_hand(0)) << "\n";
        log << "P1 hand: " << fmt_hand(game.player_hand(1)) << "\n\n";
      }

      int turn_idx = 0;
      int last_mover = -1;
      while (!game.is_over()) {
        int cp = game.current_player();
        auto *curr = (cp == 0) ? p0.get() : p1.get();
        auto *other = (cp == 0) ? p1.get() : p0.get();

        Move last_move = game.last_move();
        auto our_hand = game.player_hand(cp);
        int our_size = game.get_player_hand_size(cp);
        int opp_size = game.get_player_hand_size(1 - cp);

        Move chosen = curr->select_move();
        int tb_case = curr->last_tb_case();
        bool used_search = (tb_case == -1);

        const auto &res = curr->last_result();
        std::size_t n_legal = res.top_moves.size();
        int top1_move = -1;
        float top1_val = 0.0f;
        if (used_search && !res.top_moves.empty()) {
          top1_move = res.top_moves.front().first;
          top1_val = res.top_moves.front().second;
        }

        if (log_this) {
          log << "Turn " << turn_idx << "  P" << cp
              << "  hand(" << our_size << ")=" << fmt_hand(our_hand)
              << "  opp=" << opp_size
              << "  last=" << fmt_move(last_move) << "\n";
          if (used_search) {
            log << "  search: " << res.nodes_searched
                << " nodes, " << n_legal << " legal moves\n";
            std::size_t shown = std::min<std::size_t>(10, res.top_moves.size());
            for (std::size_t i = 0; i < shown; ++i) {
              Move mv(res.top_moves[i].first);
              log << "    " << (i + 1) << ". " << fmt_move(mv)
                  << "  v=" << res.top_moves[i].second << "\n";
            }
          } else {
            log << "  tablebase: case=" << tb_case << "\n";
          }
          log << "  -> chosen: " << fmt_move(chosen) << "\n\n";
        }

        if (stats.is_open()) {
          stats << gi << "," << turn_idx << "," << cp << "," << tb_case
                << "," << static_cast<int>(last_move.combination)
                << "," << last_move.rank
                << "," << our_size << "," << opp_size
                << "," << (used_search ? res.nodes_searched : 0)
                << "," << n_legal
                << "," << top1_move
                << "," << top1_val
                << "," << encodeMove(chosen)
                << "," << (used_search ? res.value : 0.0f)
                << "\n";
        }

        game.apply_move(chosen);
        other->accept_opponent_move(chosen);
        last_mover = cp;
        ++turn_idx;
      }

      if (log_this) {
        log << "Winner: P" << last_mover << " after " << turn_idx
            << " turns\n";
      }
    }
    if (stats.is_open()) std::cout << "Stats CSV written.\n";
    if (args.sample_games > 0 && !args.sample_dir.empty())
      std::cout << "Annotated sample games written.\n";

    auto ps = typed_search::snapshot_prune_stats_thread_local();
    if (ps.calls > 0) {
      const std::uint64_t total_drops = ps.total_drops_single +
                                          ps.total_drops_bomb_aux +
                                          ps.total_drops_fh_aux;
      auto pct_of = [](std::uint64_t n, std::uint64_t d) {
        return d == 0 ? 0.0 : 100.0 * static_cast<double>(n) / d;
      };
      double avg_in = static_cast<double>(ps.total_input_moves) / ps.calls;
      double avg_drop = static_cast<double>(total_drops) / ps.calls;
      std::cout << "Prune stats over " << ps.calls << " visit_our calls ("
                << ps.calls_multi << " with >1 legal moves):\n";
      std::cout << "  calls_with_drop:    " << ps.calls_with_drop << " ("
                << pct_of(ps.calls_with_drop, ps.calls)
                << "% of all, "
                << pct_of(ps.calls_with_drop, ps.calls_multi)
                << "% of multi-move)\n";
      std::cout << "  total drops:        " << total_drops
                << " — single=" << ps.total_drops_single
                << " ("
                << pct_of(ps.total_drops_single, total_drops)
                << "%) bomb_aux=" << ps.total_drops_bomb_aux
                << " (" << pct_of(ps.total_drops_bomb_aux, total_drops)
                << "%) fh_aux=" << ps.total_drops_fh_aux
                << " (" << pct_of(ps.total_drops_fh_aux, total_drops)
                << "%)\n";
      std::cout << "  avg legal_moves in: " << avg_in << "\n";
      std::cout << "  avg dropped/call:   " << avg_drop << " ("
                << pct_of(static_cast<std::uint64_t>(avg_drop * 1000),
                           static_cast<std::uint64_t>(avg_in * 1000))
                << "% of input)\n";
    }
  }
  return 0;
}
