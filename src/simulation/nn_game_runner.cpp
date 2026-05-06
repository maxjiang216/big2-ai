#include "nn_game_runner.h"

#include "hint_table.h"
#include "move.h"
#include "tablebase_opp1.h"
#include "util.h"

#include <algorithm>
#include <optional>

// ---------------------------------------------------------------------------
// Tablebase detection (mirrors tablebase_peek.h but uses Game directly)
// ---------------------------------------------------------------------------

// Returns the forced/winning move_id if the current position is a tablebase
// case, or -1 if normal policy should choose.
// Cases:
//   1. A legal move empties the player's hand → guaranteed win
//   2. Lead + opponent has 1 card → opp1 tablebase / all-singles heuristic
//   3. Lead + find_forced_win succeeds → forced win sequence
static int tablebase_move_id(const Game& game, int cp,
                              const std::vector<int>& legal) {
    auto hand = game.player_hand(cp);
    int hand_size = game.get_player_hand_size(cp);
    int opp_size = game.get_player_hand_size(1 - cp);

    // Case 1: play entire hand at once
    for (int mid : legal) {
        if (mid == kPASS) continue;
        if (MOVE_TO_CARDS[mid][13] == hand_size)
            return mid;
    }

    // Cases 2 & 3 only apply when we have the lead (last move was pass)
    if (game.last_move().combination != Move::Combination::kPass)
        return -1;

    auto discard = game.discard_pile();

    // Case 2: opponent has exactly 1 card
    if (opp_size == 1) {
        // All-singles optimisation: play highest single
        int best_rank = -1;
        bool all_singles = true;
        for (int mid : legal) {
            if (mid == kPASS) continue;
            Move m(mid);
            if (m.combination != Move::Combination::kSingle) {
                all_singles = false;
                break;
            }
            if (m.rank > best_rank) best_rank = m.rank;
        }
        if (all_singles && best_rank != -1) {
            for (int mid : legal) {
                if (mid == kPASS) continue;
                if (Move(mid).rank == best_rank) return mid;
            }
        }

        // Opp1 tablebase lookup
        Opp1Result opp1 = lookup_opp1(hand);
        if (opp1.first_move_id != 0) {
            for (int mid : legal) {
                if (mid == opp1.first_move_id) return mid;
            }
        }

        // Default endgame strategy
        if (auto def = opp1_default_strategy_move(hand))
            return *def;
    }

    // Case 3: forced win sequence
    auto seq = find_forced_win(hand, discard, opp_size);
    if (seq) return (*seq)[0];

    return -1;
}

// ---------------------------------------------------------------------------
// NNGameRunner
// ---------------------------------------------------------------------------

PolicyFn make_random_policy(std::mt19937& rng) {
    return [&rng](const Game&, int, const std::vector<int>& legal) -> int {
        int idx = std::uniform_int_distribution<int>(
            0, static_cast<int>(legal.size()) - 1)(rng);
        return legal[idx];
    };
}

NNGameRunner::NNGameRunner(PolicyFn p0_fn, PolicyFn p1_fn)
    : _p0_fn(std::move(p0_fn)), _p1_fn(std::move(p1_fn)) {}

NNGameData NNGameRunner::run_game(std::mt19937& rng) const {
    Game game;
    game.shuffle_deal(rng);

    std::vector<NNTurnData> turns;
    turns.reserve(64);

    while (!game.is_over()) {
        int cp = game.current_player();
        auto legal = game.get_legal_moves();

        // Tablebase: play out without recording turns; both players use
        // tablebase when applicable, policy otherwise (no random fallback).
        int tb_mid = tablebase_move_id(game, cp, legal);
        if (tb_mid >= 0) {
            game.apply_move(tb_mid);
            while (!game.is_over()) {
                int cp2 = game.current_player();
                auto rem = game.get_legal_moves();
                int tb2 = tablebase_move_id(game, cp2, rem);
                if (tb2 >= 0) {
                    game.apply_move(tb2);
                } else if (rem.size() == 1) {
                    game.apply_move(rem[0]);
                } else {
                    const auto& fn = (cp2 == 0) ? _p0_fn : _p1_fn;
                    game.apply_move(fn(game, cp2, rem));
                }
            }
            break;
        }

        // Single legal move: forced play, no training value
        if (legal.size() == 1) {
            game.apply_move(legal[0]);
            continue;
        }

        // Policy chooses move
        const auto& fn = (cp == 0) ? _p0_fn : _p1_fn;
        int move_id = fn(game, cp, legal);

        // Apply move, then snapshot post-move state
        game.apply_move(move_id);

        const HandBits hb = game.player_hand_bits(cp);
        const auto discard_after = game.discard_pile();
        const int opp_count = game.get_player_hand_size(1 - cp);

        // Build opp_at: for each rank, opp_max[r] = max_cards - count(hb,r) - discard[r]
        // then encode as HandBits.
        HandBits opp_bits{};
        for (int r = 0; r < 13; ++r) {
            const int hand_r = ((hb.at1>>r)&1) + ((hb.at2>>r)&1) +
                               ((hb.at3>>r)&1) + ((hb.at4>>r)&1);
            const int om = max_cards_in_deck_for_rank(r) - hand_r - discard_after[r];
            if (om >= 1) opp_bits.at1 |= static_cast<uint16_t>(1u << r);
            if (om >= 2) opp_bits.at2 |= static_cast<uint16_t>(1u << r);
            if (om >= 3) opp_bits.at3 |= static_cast<uint16_t>(1u << r);
            if (om >= 4) opp_bits.at4 |= static_cast<uint16_t>(1u << r);
        }

        std::array<int, 13> move_enc;
        for (int r = 0; r < 13; ++r)
            move_enc[r] = MOVE_TO_CARDS[move_id][r];

        const bool pass_ = (move_id == kPASS);
        const bool flag = !pass_ && !opponent_can_respond(move_id, opp_bits, opp_count);
        const float hint = pass_ ? 0.0f : (flag ? 1.0f : (1.0f - kHintTable[move_id]));

        turns.push_back({cp, move_enc,
                         {hb.at1, hb.at2, hb.at3, hb.at4},
                         {opp_bits.at1, opp_bits.at2, opp_bits.at3, opp_bits.at4},
                         hint, flag, pass_});
    }

    return {game.get_winner(), std::move(turns)};
}
