#ifndef NN_GAME_RUNNER_H
#define NN_GAME_RUNNER_H

#include "game.h"

#include <array>
#include <functional>
#include <random>
#include <vector>

// Compact per-turn record for NN training. All fields reflect the game state
// AFTER the move was applied (post-move hand, discard includes move cards).
//
// hand_at[n] / opp_at[n]:  bit r set iff the player/opponent-max has ≥(n+1) of rank r.
// Python: count[r] = (at[0]>>r&1) + (at[1]>>r&1) + (at[2]>>r&1) + (at[3]>>r&1)
struct NNTurnData {
    int current_player;               // 0 or 1 (who played this move)
    std::array<int, 13> move_enc;     // MOVE_TO_CARDS[move_id][0..12]
    uint16_t hand_at[4];              // post-move player hand as HandBits (at1..at4)
    uint16_t opp_at[4];              // post-move opponent max counts as HandBits
    float hint;                       // P(we win the trick) = 1-kHintTable; 0=pass, 1=flag
    bool flag;                        // !opponent_can_respond() post-move (logical deduction)
    bool pass_;                       // move_id == kPASS
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
