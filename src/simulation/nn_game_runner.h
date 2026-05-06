#ifndef NN_GAME_RUNNER_H
#define NN_GAME_RUNNER_H

#include "game.h"

#include <array>
#include <functional>
#include <random>
#include <vector>

// Per-turn record for NN training. All fields reflect game state AFTER the move.
// hand_after / opp_counts: rank counts [0..12], ready for Python encode_exact/thermo.
// trick_winner: 1 if this player won the trick, 0 if not (computed in C++ during sim).
struct NNTurnData {
    int current_player;               // 0 or 1 (who played this move)
    std::array<int, 13> move_enc;     // move rank counts (MOVE_TO_CARDS[move_id][0..12])
    std::array<int, 13> hand_after;   // post-move player rank counts
    std::array<int, 13> opp_counts;   // opponent max-possible rank counts
    float hint;                       // P(we win the trick) = 1-kHintTable; 0=pass, 1=flag
    bool flag;                        // !opponent_can_respond() post-move (logical deduction)
    bool pass_;                       // move_id == kPASS
    bool vi_forced;                   // find_forced_win succeeds from hand_after w/ initiative
    bool vn_forced;                   // vn=0: 1 card left, opp has <=1 card strictly below it
    int8_t trick_winner;              // 1 if current_player won this trick, 0 otherwise
};

struct NNGameData {
    int winner;                       // 0 or 1
    std::vector<NNTurnData> turns;    // excludes tablebase and forced-singleton turns
};

// Returns chosen move_id given the full Game state, which player is moving,
// and the pre-computed legal move list.
using PolicyFn = std::function<int(const Game&, int player_num,
                                    const std::vector<int>& legal_moves)>;

// Random policy: picks uniformly at random.
// Captures rng by reference — the RNG must outlive the returned PolicyFn.
PolicyFn make_random_policy(std::mt19937& rng);

class NNGameRunner {
public:
    NNGameRunner(PolicyFn p0_fn, PolicyFn p1_fn);

    // Runs one complete game. Tablebase and forced-singleton turns are not
    // recorded in NNGameData; the winner is determined from game.get_winner()
    // unless a tablebase early-exit fires (in which case winner = current player).
    NNGameData run_game(std::mt19937& rng) const;

private:
    PolicyFn _p0_fn;
    PolicyFn _p1_fn;
};

#endif
