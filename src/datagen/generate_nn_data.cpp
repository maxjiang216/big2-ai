#include "nn_game_runner.h"
#include "parquet_export.hpp"

#include <arrow/builder.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Parquet export
// ---------------------------------------------------------------------------

static const char* kAtNames[4] = { "at1", "at2", "at3", "at4" };

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

static void write_nn_parquet(
    const std::vector<std::pair<int, NNGameData>>& sorted,
    const std::string& path) {

    size_t total_rows = 0;
    for (const auto& pr : sorted)
        total_rows += pr.second.turns.size();

    // --- Builders ---
    arrow::Int32Builder   game_id_b, turn_idx_b, cp_b, winner_b;
    arrow::Int32Builder   move_b[13], hand_at_b[4], opp_at_b[4];
    arrow::FloatBuilder   hint_b;
    arrow::BooleanBuilder flag_b, pass_b;

    auto reserve_all = [&](int64_t n) {
        game_id_b.Reserve(n); turn_idx_b.Reserve(n); cp_b.Reserve(n); winner_b.Reserve(n);
        hint_b.Reserve(n); flag_b.Reserve(n); pass_b.Reserve(n);
        for (int r = 0; r < 13; ++r) move_b[r].Reserve(n);
        for (int i = 0; i < 4; ++i) { hand_at_b[i].Reserve(n); opp_at_b[i].Reserve(n); }
    };
    reserve_all(static_cast<int64_t>(total_rows));

    // --- Fill ---
    for (const auto& pr : sorted) {
        int gid = pr.first;
        const auto& gd = pr.second;
        int tidx = 0;
        for (const auto& t : gd.turns) {
            game_id_b.UnsafeAppend(gid);
            turn_idx_b.UnsafeAppend(tidx++);
            cp_b.UnsafeAppend(t.current_player);
            winner_b.UnsafeAppend(gd.winner);
            hint_b.UnsafeAppend(t.hint);
            flag_b.UnsafeAppend(t.flag);
            pass_b.UnsafeAppend(t.pass_);
            for (int r = 0; r < 13; ++r)
                move_b[r].UnsafeAppend(t.move_enc[r]);
            for (int i = 0; i < 4; ++i) {
                hand_at_b[i].UnsafeAppend(static_cast<int32_t>(t.hand_at[i]));
                opp_at_b[i].UnsafeAppend(static_cast<int32_t>(t.opp_at[i]));
            }
        }
    }

    // --- Finish into arrays ---
    auto finish_int  = [](arrow::Int32Builder&   b) { std::shared_ptr<arrow::Array> a; b.Finish(&a); return a; };
    auto finish_flt  = [](arrow::FloatBuilder&   b) { std::shared_ptr<arrow::Array> a; b.Finish(&a); return a; };
    auto finish_bool = [](arrow::BooleanBuilder& b) { std::shared_ptr<arrow::Array> a; b.Finish(&a); return a; };

    // Schema + arrays in declaration order
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    auto push = [&](std::shared_ptr<arrow::Field> f, std::shared_ptr<arrow::Array> a) {
        fields.push_back(std::move(f));
        arrays.push_back(std::move(a));
    };

    push(arrow::field("game_id",        arrow::int32()),   finish_int(game_id_b));
    push(arrow::field("turn_idx",       arrow::int32()),   finish_int(turn_idx_b));
    push(arrow::field("current_player", arrow::int32()),   finish_int(cp_b));

    for (int r = 0; r < 13; ++r)
        push(arrow::field(std::string("move_at") + std::to_string(r), arrow::int32()), finish_int(move_b[r]));
    for (int i = 0; i < 4; ++i)
        push(arrow::field(std::string("hand_") + kAtNames[i], arrow::int32()), finish_int(hand_at_b[i]));
    for (int i = 0; i < 4; ++i)
        push(arrow::field(std::string("opp_") + kAtNames[i], arrow::int32()), finish_int(opp_at_b[i]));

    push(arrow::field("hint",   arrow::float32()),  finish_flt(hint_b));
    push(arrow::field("flag",   arrow::boolean()),  finish_bool(flag_b));
    push(arrow::field("pass_",  arrow::boolean()),  finish_bool(pass_b));
    push(arrow::field("winner", arrow::int32()),    finish_int(winner_b));

    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
    parquet::WriterProperties::Builder pb;
    pb.compression(parquet::Compression::SNAPPY);
    parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1 << 20, pb.build());
    out->Close();
}

#pragma GCC diagnostic pop

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

static std::string format_eta(double secs) {
    if (secs < 0 || !std::isfinite(secs)) return "?";
    int s = static_cast<int>(secs + 0.5);
    if (s < 60) return std::to_string(s) + "s";
    int m = s / 60; s %= 60;
    if (m < 60) return std::to_string(m) + "m " + std::to_string(s) + "s";
    int h = m / 60; m %= 60;
    return std::to_string(h) + "h " + std::to_string(m) + "m";
}

static void print_progress(int done, int total, double elapsed) {
    if (total <= 0) return;
    int pct = static_cast<int>(100.0 * done / total);
    int bar_w = 24;
    int filled = static_cast<int>(bar_w * done / static_cast<double>(total));
    std::string bar(bar_w, '.');
    for (int i = 0; i < filled; ++i) bar[i] = '#';
    double rate = elapsed > 0 ? done / elapsed : 0;
    double eta  = (done > 0 && rate > 0) ? (total - done) / rate : 0.0;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "\rPlaying games: [%s] %d/%d (%d%%) | %.1f games/s | ETA %s   ",
        bar.c_str(), done, total, pct, rate, format_eta(eta).c_str());
    std::cerr << buf << std::flush;
}

// ---------------------------------------------------------------------------
// Usage / main
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --games <N>      Number of games (required)\n"
        << "  --out <path>     Output .parquet file (required)\n"
        << "  --threads <T>    Workers (default: hardware - 2)\n"
        << "  --seed <S>       RNG seed (default: random)\n"
        << "\nRuns random-vs-random self-play. Each row = one recorded turn.\n"
        << "Tablebase and forced-singleton turns are excluded.\n"
        << "Columns: game_id, turn_idx, current_player, "
           "move_at0..12(×13), hand_at1..at4(×4), opp_at1..at4(×4), "
           "hint, flag, pass_, winner\n";
}

int main(int argc, char** argv) {
    int num_games  = 0;
    std::string out_path;
    unsigned int seed   = std::random_device{}();
    int num_threads     = std::max(1u, std::thread::hardware_concurrency() - 2);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--help"    || arg == "-h")      { print_usage(argv[0]); return 0; }
        else if (arg == "--games"   && i+1 < argc) num_games   = std::stoi(argv[++i]);
        else if (arg == "--out"     && i+1 < argc) out_path    = argv[++i];
        else if (arg == "--threads" && i+1 < argc) num_threads = std::stoi(argv[++i]);
        else if (arg == "--seed"    && i+1 < argc) seed        = std::stoul(argv[++i]);
        else { std::cerr << "Unknown: " << arg << "\n"; print_usage(argv[0]); return 1; }
    }

    if (num_games <= 0 || out_path.empty()) {
        std::cerr << "Error: --games and --out required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== generate_nn_data ===\n"
              << "Games:   " << num_games   << "\n"
              << "Out:     " << out_path    << "\n"
              << "Seed:    " << seed        << "\n"
              << "Threads: " << num_threads << "\n\n";

    std::atomic<int> games_done{0};
    auto t_start = std::chrono::high_resolution_clock::now();

    // Progress reporter thread
    std::atomic<bool> prog_stop{false};
    std::thread prog_thread([&] {
        while (!prog_stop.load()) {
            int done = games_done.load();
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_start).count();
            print_progress(done, num_games, elapsed);
            if (done >= num_games) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    constexpr int BATCH_SIZE = 200'000;
    std::vector<std::string> tmp_files;
    int remaining = num_games;
    int batch_idx = 0;

    while (remaining > 0) {
        int this_batch   = std::min(BATCH_SIZE, remaining);
        int global_start = num_games - remaining;

        std::atomic<int> next_idx{0};
        std::vector<std::thread> workers;
        std::vector<std::vector<std::pair<int, NNGameData>>> local(num_threads);
        workers.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&, t]() {
                std::mt19937 rng(seed + static_cast<unsigned>(t)
                                 + static_cast<unsigned>(batch_idx * num_threads));
                auto p0 = make_random_policy(rng);
                auto p1 = make_random_policy(rng);
                NNGameRunner runner(p0, p1);
                auto& out = local[t];
                int idx;
                while ((idx = next_idx.fetch_add(1)) < this_batch) {
                    out.emplace_back(global_start + idx, runner.run_game(rng));
                    games_done.fetch_add(1);
                }
            });
        }
        for (auto& th : workers) if (th.joinable()) th.join();

        std::vector<std::pair<int, NNGameData>> batch;
        batch.reserve(static_cast<size_t>(this_batch));
        for (auto& v : local)
            for (auto& p : v)
                batch.push_back(std::move(p));
        std::sort(batch.begin(), batch.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::string tmp = "tmp_nn_batch_" + std::to_string(batch_idx) + ".parquet";
        write_nn_parquet(batch, tmp);
        tmp_files.push_back(tmp);
        remaining -= this_batch;
        ++batch_idx;
    }

    prog_stop = true;
    if (prog_thread.joinable()) prog_thread.join();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cerr << "\n";

    if (tmp_files.size() == 1) {
        std::filesystem::rename(tmp_files[0], out_path);
    } else {
        std::cout << "Concatenating " << tmp_files.size() << " batches...\n";
        concat_parquet_files(tmp_files, out_path);
        for (const auto& f : tmp_files) std::filesystem::remove(f);
    }

    long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_end - t_start).count();
    std::cout << "Done. " << total_ms << " ms  ("
              << static_cast<long>(total_ms > 0 ? 1000.0 * num_games / total_ms : 0)
              << " games/s)\n"
              << "Output: " << out_path << "\n";
    return 0;
}
