// az_pi_selfplay: perfect-information AlphaZero self-play data generation.
//
// Both hands are visible; one memoryless net guides a PUCT tree (src/players/
// az_pi). Many games run concurrently in a single-threaded pool: each round
// every active slot is pumped until it parks on one NN leaf eval (or finishes a
// decision/game with no eval needed, e.g. solver-proven lines); the parked
// leaves are flushed as ONE batched forward and distributed. No KV cache (the
// net is memoryless), no history.
//
// Output: one Parquet file matching the player schema in nn/dataset.py
//   game_id, turn_idx          int32
//   enc                        list<float32>(144) = exact(hand)|exact(opp)|exact(trick)
//   opp_size, our_size         int32
//   value                      float32  (1 if the mover won the game else 0)
//   legal                      list<int32>  legal engine move ids (mask)
//   visit_moves, visit_counts  list<int32>  MCTS root visit distribution (target)
//
//   make az_pi_selfplay
//   ./bin/az_pi_selfplay --model models/az_pi_gen0.pt --games 20000 \
//       --sims 400 --out data/az_pi_gen1.parquet --device cuda

#include "az_pi/pi_nn_eval.h"
#include "az_pi/pi_search.h"
#include "game.h"
#include "move.h"
#include "nn_encode.h"
#include "util.h"

#include <torch/cuda.h>

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>

#include <omp.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace az_pi;

static const char *arg(int argc, char **argv, const char *key, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}

struct Sample {
  int game_id, turn_idx, mover, opp_size, our_size;
  float value = 0.0f;   // backfilled at game end
  float root_v = 0.5f;  // search value at the root (exact 1/0 when proven)
  std::array<int, 13> hand, opp_hand, trick;
  std::vector<int> legal;
  std::vector<int> visit_moves, visit_counts;
};

struct Slot {
  bool active = false;
  Game game;
  PiNodePool pool;  // node storage persists across this slot's games
  std::unique_ptr<PiSearch> tree;
  int sims_done = 0;
  bool waiting = false;
  PiEvalFeatures pending{};
  int game_id = -1;
  int turn = 0;
  std::mt19937 rng;
  std::vector<int> sample_idx;  // indices into the slot's pending samples
  std::vector<Sample> samples;
};

int main(int argc, char **argv) {
  const std::string model_path = arg(argc, argv, "--model", "");
  const std::string out_path = arg(argc, argv, "--out", "data/az_pi.parquet");
  const int total_games = std::atoi(arg(argc, argv, "--games", "20000"));
  const int sims = std::atoi(arg(argc, argv, "--sims", "400"));
  const int n_slots = std::atoi(arg(argc, argv, "--slots", "1024"));
  const int temp_moves = std::atoi(arg(argc, argv, "--temp-moves", "8"));
  const float temp = std::atof(arg(argc, argv, "--temp", "1.0"));
  const float dir_alpha = std::atof(arg(argc, argv, "--dirichlet-alpha", "0.3"));
  const float dir_eps = std::atof(arg(argc, argv, "--dirichlet-eps", "0.25"));
  const float cpuct = std::atof(arg(argc, argv, "--cpuct", "1.5"));
  const int solve_cards = std::atoi(arg(argc, argv, "--solve-cards", "12"));
  const long solve_nodes = std::atol(arg(argc, argv, "--solve-nodes", "10000"));
  const unsigned seed = std::atoi(arg(argc, argv, "--seed", "0"));
  const std::string device_s = arg(argc, argv, "--device", "cuda");
  const int threads = std::atoi(arg(argc, argv, "--threads", "0"));  // shard count

  if (model_path.empty()) {
    std::fprintf(stderr, "error: --model is required\n");
    return 1;
  }
  torch::Device device(device_s == "cuda" && torch::cuda::is_available()
                           ? torch::kCUDA
                           : torch::kCPU);
  PiNNEvaluator nn(model_path, device);

  std::printf("az_pi self-play: games=%d sims=%d slots=%d device=%s\n",
              total_games, sims, n_slots, device.is_cuda() ? "cuda" : "cpu");

  std::mt19937 master(seed);
  std::vector<Slot> slots(n_slots);
  for (int i = 0; i < n_slots; ++i) slots[i].rng.seed(master());

  std::atomic<int> games_started{0}, games_done{0};
  std::vector<Sample> all_samples;
  std::mutex samples_mu;  // guards all_samples (hit once per finished game)

  auto make_config = [&](unsigned s) {
    PiSearchConfig cfg;
    cfg.c_puct = cpuct;
    cfg.sims = sims;
    cfg.seed = s;
    cfg.root_noise = true;
    cfg.dir_eps = dir_eps;
    cfg.dir_alpha = dir_alpha;
    cfg.solver.max_total_cards = solve_cards;
    cfg.solver.node_budget = solve_nodes;
    return cfg;
  };

  auto start_game = [&](Slot &s) -> bool {
    const int id = games_started.fetch_add(1, std::memory_order_relaxed);
    if (id >= total_games) return false;  // counter overshoot is harmless
    s.game = Game();
    s.game.shuffle_deal(s.rng);
    s.tree.reset();
    s.sims_done = 0;
    s.waiting = false;
    s.turn = 0;
    s.samples.clear();
    s.game_id = id;
    s.active = true;
    return true;
  };

  auto finalize_game = [&](Slot &s) {
    const int winner = s.game.get_winner();
    for (auto &smp : s.samples)
      smp.value = (smp.mover == winner) ? 1.0f : 0.0f;
    {
      std::lock_guard<std::mutex> lk(samples_mu);
      for (auto &smp : s.samples) all_samples.push_back(std::move(smp));
    }
    s.samples.clear();
    s.active = false;
    s.tree.reset();
    // Pools never shrink on their own; an unlucky game can leave a huge
    // high-water mark. Drop oversized pools between games (tree is dead) so
    // slot memory stays bounded — unbounded pools froze the box at high sims.
    if (s.pool.storage.size() > 1500) {
      s.pool.free_list.clear();
      s.pool.storage.clear();
    }
    games_done.fetch_add(1, std::memory_order_relaxed);
  };

  auto complete_decision = [&](Slot &s) {
    const int mover = s.game.current_player();
    auto visits = s.tree->root_visits();
    if (visits.empty()) visits.push_back({s.tree->best_move(), 1});

    Sample smp;
    smp.game_id = s.game_id;
    smp.turn_idx = s.turn;
    smp.mover = mover;
    smp.hand = s.game.player_hand(mover);
    smp.opp_hand = s.game.player_hand(1 - mover);
    const int lm = s.game.last_move_id();
    smp.trick.fill(0);
    if (lm != kPASS)
      for (int r = 0; r < 13; ++r) smp.trick[r] = MOVE_TO_CARDS[lm][r];
    smp.our_size = s.game.get_player_hand_size(mover);
    smp.opp_size = s.game.get_player_hand_size(1 - mover);
    smp.root_v = s.tree->root_value();
    smp.legal = s.game.get_legal_moves();
    for (auto &[m, n] : visits) {
      smp.visit_moves.push_back(m);
      smp.visit_counts.push_back(static_cast<int>(n));
    }
    s.samples.push_back(std::move(smp));

    const float t = (s.turn < temp_moves) ? temp : 0.0f;
    const int played = s.tree->sample_move(t, s.rng);
    s.game.apply_move(played);
    s.tree->advance_root(played);  // keep the played subtree (re-noised)
    s.sims_done = 0;
    ++s.turn;
  };

  // Pump one slot until it parks on an eval (waiting=true) or has no more work
  // this round. Returns true if it parked (its pending feat should be batched).
  auto advance = [&](Slot &s) -> bool {
    for (;;) {
      if (!s.active) {
        if (!start_game(s)) return false;
        continue;
      }
      if (s.game.is_over()) {
        finalize_game(s);
        continue;
      }
      {
        const auto legal = s.game.get_legal_moves();
        if (legal.size() == 1) {  // forced move: skip search + recording
          s.game.apply_move(legal[0]);
          if (s.tree) s.tree->advance_root(legal[0]);
          ++s.turn;
          continue;
        }
      }
      if (!s.tree) {
        s.tree = std::make_unique<PiSearch>(s.game, make_config(s.rng()),
                                            &s.pool);
        s.sims_done = 0;
      }
      while (s.sims_done < sims && !s.tree->root_proven()) {
        PiLeafRequest req = s.tree->select_leaf();
        if (req.needs_eval) {
          s.pending = req.feat;
          s.waiting = true;
          return true;
        }
        ++s.sims_done;
      }
      complete_decision(s);
    }
  };

  auto t0 = std::chrono::steady_clock::now();

  // ── Sharded pipeline ────────────────────────────────────────────────────
  // The slots are split into independent shards, each pumped by ONE thread
  // (no intra-shard synchronisation, no global barrier — a slot stuck on a
  // cold solver call only stalls its own shard). Shards submit their parked
  // leaves to a single inference thread which fuses all queued requests into
  // one forward pass, so GPU batches stay fat while CPU and GPU overlap.
  struct EvalReq {
    std::vector<PiEvalFeatures> feats;
    std::vector<PiNetEval> res;
    bool ready = false;
    std::mutex mu;
    std::condition_variable cv;
  };
  std::deque<EvalReq *> queue;
  std::mutex qmu;
  std::condition_variable qcv;
  const int n_shards =
      std::max(1, std::min(threads > 0 ? threads : 12, n_slots));
  std::atomic<int> shards_live{n_shards};

  std::thread gpu_thread([&] {
    std::vector<EvalReq *> reqs;
    std::vector<PiEvalFeatures> fused;
    for (;;) {
      reqs.clear();
      {
        std::unique_lock<std::mutex> lk(qmu);
        qcv.wait(lk, [&] {
          return !queue.empty() || shards_live.load() == 0;
        });
        if (queue.empty() && shards_live.load() == 0) return;
        while (!queue.empty()) {
          reqs.push_back(queue.front());
          queue.pop_front();
        }
      }
      fused.clear();
      for (EvalReq *r : reqs)
        fused.insert(fused.end(), r->feats.begin(), r->feats.end());
      auto res = nn.eval_batch(fused);
      std::size_t off = 0;
      for (EvalReq *r : reqs) {
        r->res.assign(res.begin() + off, res.begin() + off + r->feats.size());
        off += r->feats.size();
        {
          std::lock_guard<std::mutex> lk(r->mu);
          r->ready = true;
        }
        r->cv.notify_one();
      }
    }
  });

  std::atomic<int> last_print{0};
  auto shard_fn = [&](int sh) {
    const int lo = sh * n_slots / n_shards;
    const int hi = (sh + 1) * n_slots / n_shards;
    EvalReq req;
    std::vector<int> fslot;
    for (;;) {
      req.feats.clear();
      fslot.clear();
      for (int i = lo; i < hi; ++i) {
        if (!slots[i].waiting) advance(slots[i]);
        if (slots[i].waiting) {
          req.feats.push_back(slots[i].pending);
          fslot.push_back(i);
        }
      }
      if (req.feats.empty()) break;  // every game in this shard finished

      req.ready = false;
      {
        std::lock_guard<std::mutex> lk(qmu);
        queue.push_back(&req);
      }
      qcv.notify_one();
      {
        std::unique_lock<std::mutex> lk(req.mu);
        req.cv.wait(lk, [&] { return req.ready; });
      }
      for (std::size_t k = 0; k < fslot.size(); ++k) {
        Slot &s = slots[fslot[k]];
        s.tree->apply_eval(req.res[k]);
        ++s.sims_done;
        s.waiting = false;
      }

      const int done_now = games_done.load(std::memory_order_relaxed);
      if (done_now / 1000 > last_print.load(std::memory_order_relaxed)) {
        last_print.store(done_now / 1000, std::memory_order_relaxed);
        double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        std::size_t ns;
        {
          std::lock_guard<std::mutex> lk(samples_mu);
          ns = all_samples.size();
        }
        std::printf("  games %d/%d  samples %zu  %.1fs\n", done_now,
                    total_games, ns, el);
        std::fflush(stdout);
      }
    }
    if (shards_live.fetch_sub(1) == 1) qcv.notify_one();
  };

  std::vector<std::thread> shard_threads;
  for (int sh = 0; sh < n_shards; ++sh) shard_threads.emplace_back(shard_fn, sh);
  for (auto &t : shard_threads) t.join();
  gpu_thread.join();

  std::printf("self-play done: %d games, %zu samples. writing %s\n",
              games_done.load(), all_samples.size(), out_path.c_str());

  // ---- Parquet write (player schema: enc list<float32>(144)) ----
  auto pool = arrow::default_memory_pool();
  arrow::Int32Builder game_id, turn_idx, opp_size, our_size;
  arrow::FloatBuilder value, root_value;
  arrow::ListBuilder enc(pool, std::make_shared<arrow::FloatBuilder>(pool));
  arrow::ListBuilder legal(pool, std::make_shared<arrow::Int32Builder>(pool));
  arrow::ListBuilder vmoves(pool, std::make_shared<arrow::Int32Builder>(pool));
  arrow::ListBuilder vcounts(pool, std::make_shared<arrow::Int32Builder>(pool));
  auto *enc_v = static_cast<arrow::FloatBuilder *>(enc.value_builder());
  auto *legal_v = static_cast<arrow::Int32Builder *>(legal.value_builder());
  auto *vmoves_v = static_cast<arrow::Int32Builder *>(vmoves.value_builder());
  auto *vcounts_v = static_cast<arrow::Int32Builder *>(vcounts.value_builder());

  std::array<float, 3 * ENCODING_DIM> buf;
  for (const auto &s : all_samples) {
    game_id.Append(s.game_id);
    turn_idx.Append(s.turn_idx);
    opp_size.Append(s.opp_size);
    our_size.Append(s.our_size);
    value.Append(s.value);
    root_value.Append(s.root_v);
    encode_exact(s.hand, buf.data());
    encode_exact(s.opp_hand, buf.data() + ENCODING_DIM);
    encode_exact(s.trick, buf.data() + 2 * ENCODING_DIM);
    enc.Append();
    enc_v->AppendValues(buf.data(), static_cast<int64_t>(buf.size()));
    legal.Append();
    legal_v->AppendValues(s.legal.data(), static_cast<int64_t>(s.legal.size()));
    vmoves.Append();
    vmoves_v->AppendValues(s.visit_moves.data(),
                           static_cast<int64_t>(s.visit_moves.size()));
    vcounts.Append();
    vcounts_v->AppendValues(s.visit_counts.data(),
                            static_cast<int64_t>(s.visit_counts.size()));
  }

  auto finish_i32 = [](arrow::Int32Builder &b) {
    std::shared_ptr<arrow::Array> a;
    b.Finish(&a);
    return a;
  };
  auto finish_list = [](arrow::ListBuilder &b) {
    std::shared_ptr<arrow::Array> a;
    b.Finish(&a);
    return a;
  };
  std::shared_ptr<arrow::Array> value_arr, root_value_arr;
  value.Finish(&value_arr);
  root_value.Finish(&root_value_arr);

  std::vector<std::shared_ptr<arrow::Field>> fields = {
      arrow::field("game_id", arrow::int32()),
      arrow::field("turn_idx", arrow::int32()),
      arrow::field("enc", arrow::list(arrow::float32())),
      arrow::field("opp_size", arrow::int32()),
      arrow::field("our_size", arrow::int32()),
      arrow::field("value", arrow::float32()),
      arrow::field("root_value", arrow::float32()),
      arrow::field("legal", arrow::list(arrow::int32())),
      arrow::field("visit_moves", arrow::list(arrow::int32())),
      arrow::field("visit_counts", arrow::list(arrow::int32())),
  };
  std::vector<std::shared_ptr<arrow::Array>> arrays = {
      finish_i32(game_id),   finish_i32(turn_idx), finish_list(enc),
      finish_i32(opp_size),  finish_i32(our_size), value_arr,
      root_value_arr,        finish_list(legal),   finish_list(vmoves),
      finish_list(vcounts),
  };

  auto table = arrow::Table::Make(arrow::schema(fields), arrays);
  auto outfile = arrow::io::FileOutputStream::Open(out_path).ValueOrDie();
  parquet::WriterProperties::Builder pb;
  pb.compression(parquet::Compression::SNAPPY);
  auto st = parquet::arrow::WriteTable(*table, pool, outfile, 1 << 20, pb.build());
  if (!st.ok()) {
    std::fprintf(stderr, "parquet write failed: %s\n", st.ToString().c_str());
    return 1;
  }
  std::printf("✓ wrote %s\n", out_path.c_str());
  return 0;
}
