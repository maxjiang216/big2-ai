// generate_nn_selfplay: NN-guided self-play data generation.
//
// Uses two independent game pools on two CPU threads, interleaved against a
// single GPU inference stream.  While the GPU runs inference for pool A, the
// CPU thread for pool B advances its games and encodes the next batch —
// hiding CPU work inside GPU latency for ~50% throughput improvement over a
// serial loop.
//
// Output format: identical to generate_nn_data (same Parquet schema).

#include "nn_game_runner.h"
#include "nn_encode.h"
#include "parquet_export.hpp"

#include "game.h"
#include "hint_table.h"
#include "move.h"
#include "tablebase_opp1.h"
#include "util.h"

#include <torch/script.h>
#include <torch/cuda.h>

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int MAX_BATCH    = 200000;  // upper bound on positions per round
static const char* kAtNames[4]    = { "at1", "at2", "at3", "at4" };

// ---------------------------------------------------------------------------
// Tablebase detection (mirror of nn_game_runner.cpp)
// ---------------------------------------------------------------------------

static int tablebase_move_id(const Game& game, int cp,
                              const std::vector<int>& legal) {
    auto hand      = game.player_hand(cp);
    int  hand_size = game.get_player_hand_size(cp);
    int  opp_size  = game.get_player_hand_size(1 - cp);

    for (int mid : legal) {
        if (mid == kPASS) continue;
        if (MOVE_TO_CARDS[mid][13] == hand_size) return mid;
    }
    if (game.last_move().combination != Move::Combination::kPass) return -1;

    auto discard = game.discard_pile();
    if (opp_size == 1) {
        int best_rank = -1; bool all_singles = true;
        for (int mid : legal) {
            if (mid == kPASS) continue;
            Move m(mid);
            if (m.combination != Move::Combination::kSingle) { all_singles = false; break; }
            if (m.rank > best_rank) best_rank = m.rank;
        }
        if (all_singles && best_rank != -1)
            for (int mid : legal)
                if (mid != kPASS && Move(mid).rank == best_rank) return mid;
        Opp1Result opp1 = lookup_opp1(hand);
        if (opp1.first_move_id != 0)
            for (int mid : legal) if (mid == opp1.first_move_id) return mid;
        if (auto def = opp1_default_strategy_move(hand)) return *def;
    }
    if (auto seq = find_forced_win(hand, discard, opp_size)) return (*seq)[0];
    return -1;
}

// Advance game through forced/tablebase moves until a real decision point.
// Returns legal moves (non-empty) or empty vector if game ended.
static std::vector<int> advance_to_decision(Game& game, std::mt19937& rng) {
    while (!game.is_over()) {
        auto legal = game.get_legal_moves();
        int tb = tablebase_move_id(game, game.current_player(), legal);
        if (tb >= 0) {
            game.apply_move(tb);
            while (!game.is_over()) {
                auto rem = game.get_legal_moves();
                int tb2 = tablebase_move_id(game, game.current_player(), rem);
                if (tb2 >= 0) game.apply_move(tb2);
                else {
                    int idx = std::uniform_int_distribution<int>(
                        0, (int)rem.size()-1)(rng);
                    game.apply_move(rem[idx]);
                }
            }
            return {};
        }
        if ((int)legal.size() == 1) { game.apply_move(legal[0]); continue; }
        return legal;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Per-position metadata needed to record training data
// ---------------------------------------------------------------------------

struct PosRecord {
    int   game_idx;    // index into pool
    int   move_idx;    // index into legal move list
    int   cp;          // current player
    float hint;
    bool  flag;
    bool  pass_;
    std::array<int,13> move_enc;
    uint16_t hand_at[4];
    uint16_t opp_at[4];
};

// ---------------------------------------------------------------------------
// Game pool
// ---------------------------------------------------------------------------

struct GameSlot {
    Game             game{};
    std::vector<int> legal{};   // legal moves at current decision point
    NNGameData       data{};    // accumulates turns for this game
    bool             active{false};
    int              global_id{-1};
};

// ---------------------------------------------------------------------------
// Inference buffer: CPU-side inputs + results for one pool's batch
// ---------------------------------------------------------------------------

struct InferBuf {
    // Flat CPU tensors (pinned) — shape [n_pos, ENCODING_DIM] or [n_pos]
    torch::Tensor hand, opp, move_t, hint, flag, pass_;
    // Outputs (written by GPU, read by CPU)
    torch::Tensor v_init, v_no_init, p_trick;
    // Mapping: which slot/move each row corresponds to
    std::vector<PosRecord> pos_map;
    // game_row_start[g] = first row index for game g; game_row_end[g] = one past last
    std::vector<int> game_row_start, game_row_end;
    int n_pos{0};

    void reset(int max_n, int pool_size) {
        auto cpu_pin = torch::TensorOptions().dtype(torch::kFloat32)
                           .device(torch::kCPU).pinned_memory(true);
        auto cpu_bool_pin = torch::TensorOptions().dtype(torch::kBool)
                                .device(torch::kCPU).pinned_memory(true);
        hand   = torch::zeros({max_n, ENCODING_DIM}, cpu_pin);
        opp    = torch::zeros({max_n, ENCODING_DIM}, cpu_pin);
        move_t = torch::zeros({max_n, ENCODING_DIM}, cpu_pin);
        hint   = torch::zeros({max_n},               cpu_pin);
        flag   = torch::zeros({max_n},               cpu_bool_pin);
        game_row_start.resize(pool_size, 0);
        game_row_end.resize(pool_size, 0);
        pass_  = torch::zeros({max_n},               cpu_bool_pin);
        pos_map.resize(max_n);
        n_pos = 0;
    }
};

// ---------------------------------------------------------------------------
// Encode one pool into an InferBuf
// ---------------------------------------------------------------------------

static void encode_pool(const std::vector<GameSlot>& pool, InferBuf& buf) {
    float* h_hand = buf.hand.data_ptr<float>();
    float* h_opp  = buf.opp.data_ptr<float>();
    float* h_move = buf.move_t.data_ptr<float>();
    float* h_hint = buf.hint.data_ptr<float>();
    bool*  h_flag = buf.flag.data_ptr<bool>();
    bool*  h_pass = buf.pass_.data_ptr<bool>();

    int row = 0;
    for (int g = 0; g < (int)pool.size(); ++g) {
        const auto& slot = pool[g];
        buf.game_row_start[g] = row;
        if (!slot.active) { buf.game_row_end[g] = row; continue; }

        int cp = slot.game.current_player();
        HandBits hb = slot.game.player_hand_bits(cp);
        auto hand_counts = hand_bits_to_counts(hb);
        auto discard     = slot.game.discard_pile();
        int  opp_count   = slot.game.get_player_hand_size(1 - cp);

        // Opponent upper-bound counts + HandBits (constant across all moves for this slot)
        std::array<int,13> opp_counts{};
        HandBits opp_bits{};
        for (int r = 0; r < 13; ++r) {
            int hand_r = ((hb.at1>>r)&1)+((hb.at2>>r)&1)+((hb.at3>>r)&1)+((hb.at4>>r)&1);
            int om = max_cards_in_deck_for_rank(r) - hand_r - discard[r];
            opp_counts[r] = om;
            if (om >= 1) opp_bits.at1 |= static_cast<uint16_t>(1u << r);
            if (om >= 2) opp_bits.at2 |= static_cast<uint16_t>(1u << r);
            if (om >= 3) opp_bits.at3 |= static_cast<uint16_t>(1u << r);
            if (om >= 4) opp_bits.at4 |= static_cast<uint16_t>(1u << r);
        }

        for (int mi = 0; mi < (int)slot.legal.size(); ++mi) {
            int move_id = slot.legal[mi];
            std::array<int,13> mc{};
            for (int r = 0; r < 13; ++r) mc[r] = MOVE_TO_CARDS[move_id][r];

            // Post-move hand (model was trained with hand-after-playing-move as input)
            std::array<int,13> hand_after{};
            for (int r = 0; r < 13; ++r) hand_after[r] = hand_counts[r] - mc[r];
            encode_exact(hand_after,   h_hand + row*ENCODING_DIM);
            encode_thermo(opp_counts,  h_opp  + row*ENCODING_DIM);
            encode_exact(mc,           h_move + row*ENCODING_DIM);

            bool is_pass = (move_id == kPASS);
            bool is_flag = !is_pass && !opponent_can_respond(move_id, opp_bits, opp_count);
            float hint_val = is_pass ? 0.0f
                           : (is_flag ? 1.0f : (1.0f - kHintTable[move_id]));

            h_hint[row] = hint_val;
            h_flag[row] = is_flag;
            h_pass[row] = is_pass;

            // Post-move hand bits (after applying this move)
            HandBits post_hb = hb;
            for (int r = 0; r < 13; ++r)
                if (mc[r]) hand_bits_remove(post_hb, r, mc[r]);

            buf.pos_map[row] = {g, mi, cp, hint_val, is_flag, is_pass,
                                mc,
                                {post_hb.at1, post_hb.at2, post_hb.at3, post_hb.at4},
                                {opp_bits.at1, opp_bits.at2, opp_bits.at3, opp_bits.at4}};
            ++row;
        }
        buf.game_row_end[g] = row;
    }
    buf.n_pos = row;
}

// ---------------------------------------------------------------------------
// Run inference and dispatch best moves back to pool
// ---------------------------------------------------------------------------

static void run_inference(torch::jit::Module& model,
                          InferBuf& buf,
                          const torch::Device& dev) {
    if (buf.n_pos == 0) return;

    auto slice = [&](torch::Tensor& t) {
        return t.slice(0, 0, buf.n_pos).to(dev, /*non_blocking=*/true);
    };
    auto d_hand  = slice(buf.hand);
    auto d_opp   = slice(buf.opp);
    auto d_move  = slice(buf.move_t);
    auto d_hint  = slice(buf.hint);
    auto d_flag  = slice(buf.flag);
    auto d_pass  = slice(buf.pass_);

    torch::NoGradGuard ng;
    auto out = model.forward({d_hand, d_opp, d_move, d_hint, d_flag, d_pass}).toTuple();
    buf.v_init    = out->elements()[0].toTensor().to(torch::kCPU);
    buf.v_no_init = out->elements()[1].toTensor().to(torch::kCPU);
    buf.p_trick   = out->elements()[2].toTensor().to(torch::kCPU);
}

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

static std::string format_eta(double s) {
    if (s < 0 || !std::isfinite(s)) return "?";
    int si = (int)(s+0.5);
    if (si < 60) return std::to_string(si) + "s";
    int m = si/60; si %= 60;
    if (m < 60) return std::to_string(m) + "m " + std::to_string(si) + "s";
    int h = m/60; m %= 60;
    return std::to_string(h) + "h " + std::to_string(m) + "m";
}

static void print_progress(int done, int total, double elapsed) {
    if (total <= 0) return;
    int pct = (int)(100.0*done/total);
    int bar_w = 24, filled = (int)(bar_w*done/(double)total);
    std::string bar(bar_w, '.'); for (int i=0;i<filled;++i) bar[i]='#';
    double rate = elapsed>0 ? done/elapsed : 0;
    double eta  = (done>0&&rate>0) ? (total-done)/rate : 0.0;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "\rPlaying games: [%s] %d/%d (%d%%) | %.0f games/s | ETA %s   ",
        bar.c_str(), done, total, pct, rate, format_eta(eta).c_str());
    std::cerr << buf << std::flush;
}

// ---------------------------------------------------------------------------
// Parquet writer (identical schema to generate_nn_data)
// ---------------------------------------------------------------------------

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

static void write_nn_parquet(
    const std::vector<std::pair<int, NNGameData>>& sorted,
    const std::string& path) {

    size_t total = 0;
    for (const auto& pr : sorted) total += pr.second.turns.size();

    arrow::Int32Builder   game_id_b, turn_idx_b, cp_b, winner_b;
    arrow::Int32Builder   move_b[13], hand_at_b[4], opp_at_b[4];
    arrow::FloatBuilder   hint_b;
    arrow::BooleanBuilder flag_b, pass_b;

    auto reserve = [&](int64_t n) {
        game_id_b.Reserve(n); turn_idx_b.Reserve(n); cp_b.Reserve(n); winner_b.Reserve(n);
        hint_b.Reserve(n); flag_b.Reserve(n); pass_b.Reserve(n);
        for (int r=0;r<13;++r) move_b[r].Reserve(n);
        for (int i=0;i<4;++i) { hand_at_b[i].Reserve(n); opp_at_b[i].Reserve(n); }
    };
    reserve((int64_t)total);

    for (const auto& pr : sorted) {
        int gid = pr.first; int tidx = 0;
        for (const auto& t : pr.second.turns) {
            game_id_b.UnsafeAppend(gid); turn_idx_b.UnsafeAppend(tidx++);
            cp_b.UnsafeAppend(t.current_player); winner_b.UnsafeAppend(pr.second.winner);
            hint_b.UnsafeAppend(t.hint); flag_b.UnsafeAppend(t.flag); pass_b.UnsafeAppend(t.pass_);
            for (int r=0;r<13;++r) move_b[r].UnsafeAppend(t.move_enc[r]);
            for (int i=0;i<4;++i) {
                hand_at_b[i].UnsafeAppend((int32_t)t.hand_at[i]);
                opp_at_b[i].UnsafeAppend((int32_t)t.opp_at[i]);
            }
        }
    }

    auto fi = [](arrow::Int32Builder&   b){ std::shared_ptr<arrow::Array> a; b.Finish(&a); return a; };
    auto ff = [](arrow::FloatBuilder&   b){ std::shared_ptr<arrow::Array> a; b.Finish(&a); return a; };
    auto fb = [](arrow::BooleanBuilder& b){ std::shared_ptr<arrow::Array> a; b.Finish(&a); return a; };

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    auto push = [&](auto f, auto a){ fields.push_back(f); arrays.push_back(a); };

    push(arrow::field("game_id",        arrow::int32()), fi(game_id_b));
    push(arrow::field("turn_idx",       arrow::int32()), fi(turn_idx_b));
    push(arrow::field("current_player", arrow::int32()), fi(cp_b));
    for (int r=0;r<13;++r)
        push(arrow::field("move_at"+std::to_string(r), arrow::int32()), fi(move_b[r]));
    for (int i=0;i<4;++i)
        push(arrow::field(std::string("hand_")+kAtNames[i], arrow::int32()), fi(hand_at_b[i]));
    for (int i=0;i<4;++i)
        push(arrow::field(std::string("opp_")+kAtNames[i], arrow::int32()), fi(opp_at_b[i]));
    push(arrow::field("hint",   arrow::float32()), ff(hint_b));
    push(arrow::field("flag",   arrow::boolean()), fb(flag_b));
    push(arrow::field("pass_",  arrow::boolean()), fb(pass_b));
    push(arrow::field("winner", arrow::int32()),   fi(winner_b));

    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    parquet::WriterProperties::Builder pb; pb.compression(parquet::Compression::SNAPPY);
    parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1<<20, pb.build());
    out->Close();
}

#pragma GCC diagnostic pop

// ---------------------------------------------------------------------------
// Usage / main
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --model <path>    TorchScript model (.pt) (required)\n"
        << "  --games <N>       Number of games (required)\n"
        << "  --out <path>      Output .parquet file (required)\n"
        << "  --pool <P>        Games per pool (default: 900, total 2P in flight)\n"
        << "  --seed <S>        RNG seed (default: random)\n"
        << "  --device <d>      cuda or cpu (default: cuda if available)\n"
        << "\nRuns NN self-play with double-buffer CPU/GPU pipeline.\n"
        << "Two independent pools of P games each; while GPU runs inference\n"
        << "for pool A, CPU advances pool B (and vice versa).\n";
}

int main(int argc, char** argv) {
    int         num_games  = 0;
    int         pool_size  = 900;
    std::string out_path;
    std::string model_path;
    std::string device_str = "auto";
    unsigned    seed       = std::random_device{}();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg=="--help"||arg=="-h")          { print_usage(argv[0]); return 0; }
        else if (arg=="--model"  && i+1<argc) model_path  = argv[++i];
        else if (arg=="--games"  && i+1<argc) num_games   = std::stoi(argv[++i]);
        else if (arg=="--out"    && i+1<argc) out_path    = argv[++i];
        else if (arg=="--pool"   && i+1<argc) pool_size   = std::stoi(argv[++i]);
        else if (arg=="--seed"   && i+1<argc) seed        = std::stoul(argv[++i]);
        else if (arg=="--device" && i+1<argc) device_str  = argv[++i];
        else { std::cerr << "Unknown: " << arg << "\n"; print_usage(argv[0]); return 1; }
    }

    if (num_games<=0 || out_path.empty() || model_path.empty()) {
        std::cerr << "Error: --model, --games and --out are required\n\n";
        print_usage(argv[0]); return 1;
    }

    torch::Device dev = torch::kCPU;
    if (device_str=="auto") dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    else if (device_str=="cuda") dev = torch::kCUDA;

    std::cout << "=== generate_nn_selfplay ===\n"
              << "Model:   " << model_path  << "\n"
              << "Games:   " << num_games   << "\n"
              << "Out:     " << out_path    << "\n"
              << "Seed:    " << seed        << "\n"
              << "Pool:    " << pool_size   << " per pool (" << 2*pool_size << " total in flight)\n"
              << "Device:  " << (dev==torch::kCUDA ? "cuda" : "cpu") << "\n\n";

    torch::jit::Module model = torch::jit::load(model_path);
    model.eval();
    model.to(dev);

    std::atomic<int> games_done{0};
    auto t_start = std::chrono::high_resolution_clock::now();

    // Progress thread
    std::atomic<bool> prog_stop{false};
    std::thread prog_thread([&]{
        while (!prog_stop.load()) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now()-t_start).count();
            print_progress(games_done.load(), num_games, elapsed);
            if (games_done.load() >= num_games) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    // Completed game data (protected by mutex)
    std::mutex data_mutex;
    std::vector<std::pair<int, NNGameData>> completed;
    completed.reserve(num_games);

    // Global game id counter
    std::atomic<int> global_id_counter{0};

    // GPU mutex — only one pool runs inference at a time
    std::mutex gpu_mutex;

    // ---------------------------------------------------------------------------
    // Worker: runs one pool, pings GPU when ready, writes completed games
    // ---------------------------------------------------------------------------
    auto worker = [&](int pool_id) {
        std::mt19937 rng(seed + static_cast<unsigned>(pool_id) * 1000007u);

        std::vector<GameSlot> pool(pool_size);
        for (auto& s : pool) {
            s.global_id = global_id_counter.fetch_add(1);
            s.game.shuffle_deal(rng);
            s.legal = advance_to_decision(s.game, rng);
            s.active = !s.legal.empty();
        }

        // Allocate inference buffer (pinned)
        InferBuf buf;
        buf.reset(pool_size * 80, pool_size);  // 80 = generous upper bound on K

        int local_games_done = 0;

        while (games_done.load() < num_games) {
            // Encode this pool's current decision points
            encode_pool(pool, buf);

            // Run inference (serialised through GPU mutex)
            {
                std::lock_guard<std::mutex> lock(gpu_mutex);
                run_inference(model, buf, dev);
            }

            // Dispatch results: apply best move, advance games
            for (int g = 0; g < pool_size; ++g) {
                auto& slot = pool[g];
                if (!slot.active) continue;

                // Find best move for this slot
                float best_w = -1e9f;
                int   best_row = -1;
                const float* vi = buf.v_init.data_ptr<float>();
                const float* vn = buf.v_no_init.data_ptr<float>();
                const float* pt = buf.p_trick.data_ptr<float>();

                for (int row = buf.game_row_start[g]; row < buf.game_row_end[g]; ++row) {
                    float w = pt[row]*vi[row] + (1.0f-pt[row])*vn[row];
                    if (w > best_w) { best_w = w; best_row = row; }
                }
                if (best_row < 0) continue;

                const auto& pr = buf.pos_map[best_row];
                int move_id = slot.legal[pr.move_idx];

                slot.data.turns.push_back({
                    pr.cp, pr.move_enc,
                    {pr.hand_at[0],pr.hand_at[1],pr.hand_at[2],pr.hand_at[3]},
                    {pr.opp_at[0], pr.opp_at[1], pr.opp_at[2], pr.opp_at[3]},
                    pr.hint, pr.flag, pr.pass_
                });

                slot.game.apply_move(move_id);
                slot.legal = advance_to_decision(slot.game, rng);

                if (slot.legal.empty()) {
                    slot.data.winner = slot.game.get_winner();
                    int gid = slot.global_id;
                    {
                        std::lock_guard<std::mutex> lk(data_mutex);
                        completed.emplace_back(gid, std::move(slot.data));
                    }
                    games_done.fetch_add(1);
                    ++local_games_done;

                    if (games_done.load() < num_games) {
                        slot.global_id = global_id_counter.fetch_add(1);
                        slot.game.shuffle_deal(rng);
                        slot.data = {};
                        slot.legal = advance_to_decision(slot.game, rng);
                        slot.active = !slot.legal.empty();
                    } else {
                        slot.active = false;
                    }
                }
            }
        }
    };

    // Launch two pool workers
    std::thread t0(worker, 0);
    std::thread t1(worker, 1);
    t0.join(); t1.join();

    prog_stop = true;
    prog_thread.join();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cerr << "\n";

    // Sort by game_id and write
    std::sort(completed.begin(), completed.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    // Trim to exactly num_games
    if ((int)completed.size() > num_games)
        completed.resize(num_games);

    std::cout << "Writing " << completed.size() << " games to " << out_path << "...\n";
    write_nn_parquet(completed, out_path);

    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
    std::cout << "Done. " << ms << " ms  ("
              << (ms>0 ? (long)(1000.0*num_games/ms) : 0) << " games/s)\n"
              << "Output: " << out_path << "\n";
    return 0;
}
