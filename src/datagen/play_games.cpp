// play_games: play fresh games with a NN model and print per-turn decisions.
//
// Both players use the same model. For each position the binary scores all
// legal moves, picks the best (argmax w), and prints the top-2 with scores so
// you can see how each player is reasoning.
//
// Usage:
//   play_games --model PATH --games N [--out FILE] [--seed S]
//              [--device auto|cpu|cuda]

#include "nn_encode.h"

#include "game.h"
#include "hint_table.h"
#include "util.h"

#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Formatting helpers
// ============================================================================

// rankToChar takes actual rank values (3=three, ..., 14=ace, 2=two).
// MOVE_TO_CARDS and player_hand() use array indices 0-12:
//   idx 0..11 → ranks 3..14; idx 12 → rank 2 (the "two" high card).
static inline char idx_to_char(int idx) {
    return rankToChar(idx < 12 ? idx + 3 : 2);
}

static std::string fmt_hand(const std::array<int,13>& hand) {
    std::string s;
    for (int r = 0; r < 13; ++r)
        for (int i = 0; i < hand[r]; ++i)
            s += idx_to_char(r);
    if (s.empty()) s = "(empty)";
    return s;
}

static std::string fmt_move(int mid) {
    if (mid == kPASS) return "pass";
    const auto& mc = MOVE_TO_CARDS[mid];
    int total = mc[13];

    std::string cards;
    int nz_count = 0, max_count = 0;
    for (int r = 0; r < 13; ++r) {
        if (mc[r]) { nz_count++; if (mc[r] > max_count) max_count = mc[r]; }
        for (int i = 0; i < mc[r]; ++i) cards += idx_to_char(r);
    }

    std::string t;
    if      (total == 1) t = "single";
    else if (total == 2 && nz_count == 1) t = "pair";
    else if (total == 3 && nz_count == 1) t = "trips";
    else if (total == 4 && nz_count == 1 && max_count == 4) t = "quad";
    else if (total == 5 && nz_count == 2 && max_count == 3) t = "FH";
    else if (total >= 5 && nz_count == total) t = "str";
    else if (max_count == 4) {
        int extra = total - 4;
        t = extra ? "bomb+" + std::to_string(extra) : "bomb";
    }
    else t = std::to_string(total) + "c";

    return t + "(" + cards + ")";
}

// ============================================================================
// Per-move scoring
// ============================================================================

struct ScoredMove { int mid; float vi, vn, pt, w; };

static std::vector<ScoredMove> score_moves(
    torch::jit::Module& model, torch::Device device,
    const Game& game, int player_num,
    const std::vector<int>& legal_moves)
{
    const int K = (int)legal_moves.size();

    HandBits hb      = game.player_hand_bits(player_num);
    auto discard     = game.discard_pile();
    auto hand_counts = hand_bits_to_counts(hb);
    int  opp_count   = game.get_player_hand_size(1 - player_num);

    std::array<int,13> opp_counts{};
    HandBits opp_bits{};
    for (int r = 0; r < 13; ++r) {
        int hand_r = ((hb.at1>>r)&1)+((hb.at2>>r)&1)+((hb.at3>>r)&1)+((hb.at4>>r)&1);
        int om = std::max(0, max_cards_in_deck_for_rank(r) - hand_r - discard[r]);
        opp_counts[r] = om;
        if (om >= 1) opp_bits.at1 |= static_cast<uint16_t>(1u << r);
        if (om >= 2) opp_bits.at2 |= static_cast<uint16_t>(1u << r);
        if (om >= 3) opp_bits.at3 |= static_cast<uint16_t>(1u << r);
        if (om >= 4) opp_bits.at4 |= static_cast<uint16_t>(1u << r);
    }

    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto bool_opts  = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor t_hand = torch::zeros({K, ENCODING_DIM}, float_opts);
    torch::Tensor t_opp  = torch::zeros({K, ENCODING_DIM}, float_opts);
    torch::Tensor t_move = torch::zeros({K, ENCODING_DIM}, float_opts);
    torch::Tensor t_hint = torch::zeros({K}, float_opts);
    torch::Tensor t_flag = torch::zeros({K}, bool_opts);
    torch::Tensor t_pass = torch::zeros({K}, bool_opts);

    float* h_hand = t_hand.data_ptr<float>();
    float* h_opp  = t_opp.data_ptr<float>();
    float* h_move = t_move.data_ptr<float>();
    float* h_hint = t_hint.data_ptr<float>();
    bool*  h_flag = t_flag.data_ptr<bool>();
    bool*  h_pass = t_pass.data_ptr<bool>();

    for (int i = 0; i < K; ++i) {
        int mid = legal_moves[i];
        bool is_pass_move = (mid == kPASS);

        std::array<int,13> mc{};
        if (!is_pass_move)
            for (int r = 0; r < 13; ++r) mc[r] = MOVE_TO_CARDS[mid][r];

        std::array<int,13> hand_after{};
        for (int r = 0; r < 13; ++r) hand_after[r] = hand_counts[r] - mc[r];

        encode_exact(hand_after,  h_hand + i * ENCODING_DIM);
        encode_thermo(opp_counts, h_opp  + i * ENCODING_DIM);
        encode_exact(mc,          h_move + i * ENCODING_DIM);

        bool flag_val = !is_pass_move && !opponent_can_respond(mid, opp_bits, opp_count);
        h_hint[i] = is_pass_move ? 0.0f : (flag_val ? 1.0f : (1.0f - kHintTable[mid]));
        h_flag[i] = flag_val;
        h_pass[i] = is_pass_move;
    }

    torch::NoGradGuard no_grad;
    auto out = model.forward({
        t_hand.to(device), t_opp.to(device), t_move.to(device),
        t_hint.to(device), t_flag.to(device), t_pass.to(device)
    }).toTuple();

    auto vi_t = out->elements()[0].toTensor().to(torch::kCPU);
    auto vn_t = out->elements()[1].toTensor().to(torch::kCPU);
    auto pt_t = out->elements()[2].toTensor().to(torch::kCPU);
    auto* vi_p = vi_t.data_ptr<float>();
    auto* vn_p = vn_t.data_ptr<float>();
    auto* pt_p = pt_t.data_ptr<float>();

    std::vector<ScoredMove> result;
    result.reserve(K);
    for (int i = 0; i < K; ++i) {
        float w = pt_p[i] * vi_p[i] + (1.0f - pt_p[i]) * vn_p[i];
        result.push_back({legal_moves[i], vi_p[i], vn_p[i], pt_p[i], w});
    }
    std::stable_sort(result.begin(), result.end(),
                     [](const ScoredMove& a, const ScoredMove& b){ return a.w > b.w; });
    return result;
}

// ============================================================================
// Game play + printing
// ============================================================================

static constexpr int kMoveWidth = 20;

static void play_and_print_game(torch::jit::Module& model, torch::Device device,
                                 std::mt19937& rng, std::ostream& out, int game_num)
{
    Game game;
    game.shuffle_deal(rng);

    int trick = 1;
    int winner = -1;

    // Collect turn output into a buffer so we can prepend winner header
    std::vector<std::string> lines;
    lines.reserve(128);

    while (!game.is_over()) {
        int cp        = game.current_player();
        int last_mid  = game.last_move_id();
        bool init     = (last_mid == kPASS);
        auto legal    = game.get_legal_moves();
        auto hand     = game.player_hand(cp);
        auto discard  = game.discard_pile();
        int  hand_sz  = game.get_player_hand_size(cp);
        int  opp_sz   = game.get_player_hand_size(1 - cp);
        bool forced   = (legal.size() == 1);
        bool endgame  = (opp_sz == 1);

        // Context string
        std::string ctx = init ? "Initiative" : ("vs " + fmt_move(last_mid));
        std::string eg  = endgame ? "  [endgame]" : "";

        std::string hdr = "  ── T" + std::to_string(trick) +
                          "  P" + std::to_string(cp) +
                          "  " + std::to_string(hand_sz) + " cards  " +
                          ctx + eg + " ─";
        lines.push_back(hdr);

        // Discard: fmt_hand returns "(empty)" for an empty hand; show "—" instead
        std::string dh = fmt_hand(discard);
        std::string disc_s = (dh == "(empty)") ? "\xe2\x80\x94" : dh;  // — (em dash)
        lines.push_back("  Hand: " + fmt_hand(hand));
        lines.push_back("  Opp:  " + std::to_string(opp_sz) + " cards   Disc: " + disc_s);

        int chosen_mid;
        std::vector<ScoredMove> topN;

        if (forced) {
            chosen_mid = legal[0];
        } else {
            auto scored = score_moves(model, device, game, cp, legal);
            chosen_mid  = scored[0].mid;
            int n = std::min((int)scored.size(), 10);
            for (int i = 0; i < n; ++i) topN.push_back(scored[i]);
        }

        if (forced) {
            std::string mv = fmt_move(chosen_mid);
            while ((int)mv.size() < kMoveWidth) mv += ' ';
            lines.push_back("  ** " + mv + "  [forced]");
        } else {
            for (int i = 0; i < (int)topN.size(); ++i) {
                const auto& sm = topN[i];
                char label[8]; std::snprintf(label, sizeof(label), "#%-2d", i + 1);
                std::string mv = fmt_move(sm.mid);
                while ((int)mv.size() < kMoveWidth) mv += ' ';
                char score_buf[64];
                std::snprintf(score_buf, sizeof(score_buf),
                              "pt=%.2f vi=%.2f vn=%.2f w=%.2f",
                              sm.pt, sm.vi, sm.vn, sm.w);
                std::string chosen_mark = (sm.mid == chosen_mid) ? "  ←" : "   ";
                lines.push_back("  " + std::string(label) + " " + mv + "  " + score_buf + chosen_mark);
            }
        }

        // Check for win before applying
        int cards_played = (chosen_mid == kPASS) ? 0 : MOVE_TO_CARDS[chosen_mid][13];
        if (hand_sz - cards_played == 0 && chosen_mid != kPASS)
            winner = cp;

        game.apply_move(chosen_mid);

        if (chosen_mid == kPASS) trick++;
    }

    if (winner < 0) winner = game.get_winner();

    // Write header + separator
    out << "\n  game " << game_num
        << "  (winner: P" << winner << ")\n";
    out << "  " << std::string(72, '-') << "\n";
    for (const auto& l : lines) out << l << "\n";
    out << "\n  " << std::string(72, '-') << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " --model PATH --games N [--out FILE] [--seed S]"
                 " [--device auto|cpu|cuda]\n";
}

int main(int argc, char* argv[]) {
    std::string model_path, out_path;
    int n_games = 3;
    unsigned seed = 42;
    std::string device_str = "auto";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"  && i+1 < argc) { model_path  = argv[++i]; }
        else if (a == "--games"  && i+1 < argc) { n_games     = std::stoi(argv[++i]); }
        else if (a == "--out"    && i+1 < argc) { out_path    = argv[++i]; }
        else if (a == "--seed"   && i+1 < argc) { seed        = (unsigned)std::stoul(argv[++i]); }
        else if (a == "--device" && i+1 < argc) { device_str  = argv[++i]; }
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else { std::cerr << "Unknown argument: " << a << "\n"; print_usage(argv[0]); return 1; }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage(argv[0]); return 1;
    }

    // Device
    torch::Device device(torch::kCPU);
    if (device_str == "auto")
        device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    else if (device_str == "cuda")
        device = torch::Device(torch::kCUDA);

    torch::jit::Module model = torch::jit::load(model_path, device);
    model.eval();

    std::cerr << "Loaded: " << model_path << " on "
              << (device.is_cuda() ? "cuda" : "cpu") << "\n";

    // Output stream
    std::ofstream file_out;
    std::ostream* out_ptr = &std::cout;
    if (!out_path.empty()) {
        // Create parent directories if needed (best-effort)
        auto slash = out_path.rfind('/');
        if (slash != std::string::npos) {
            std::string dir = out_path.substr(0, slash);
            (void)system(("mkdir -p " + dir).c_str());
        }
        file_out.open(out_path);
        if (!file_out) {
            std::cerr << "Error: cannot open " << out_path << "\n";
            return 1;
        }
        out_ptr = &file_out;
    }
    std::ostream& out = *out_ptr;

    // Extract model stem for header
    std::string model_stem = model_path;
    auto sl = model_stem.rfind('/');
    if (sl != std::string::npos) model_stem = model_stem.substr(sl + 1);
    auto dot = model_stem.rfind('.');
    if (dot != std::string::npos) model_stem = model_stem.substr(0, dot);

    out << "\n" << std::string(72, '=') << "\n"
        << "  Model: " << model_stem
        << "  (" << n_games << " games, seed=" << seed << ")\n"
        << std::string(72, '=') << "\n";

    std::mt19937 rng(seed);
    for (int i = 1; i <= n_games; ++i)
        play_and_print_game(model, device, rng, out, i);

    if (!out_path.empty())
        std::cerr << "Wrote " << out_path << "\n";

    return 0;
}
