// game_simulator.cpp
#include "game_simulator.h"
#include "player.h"
#include "game.h"

GameSimulator::GameSimulator(
    std::unique_ptr<Player> player0,
    std::unique_ptr<Player> player1,
    std::mt19937 &rng
) :
    _player0(std::move(player0)),
    _player1(std::move(player1)),
    _rng(rng),
    _game()
{}

GameRecord GameSimulator::run() {
    GameRecord record;
    // Initialize game state and inform players
    initialize_game();

    // Record initial deal
    _record.set_initial_state(_game);

    // Main play loop
    play_loop();

    // Finalize record with outcome
    _record.set_result(_game.get_winner(), _game.get_cards_remaining());
    return _record;
}

void GameSimulator::initialize_game() {
    // Shuffle and deal
    _game.shuffle_deal(_rng);

    // Inform players of new game and their hands
    auto hand0 = _game.get_player_hand(0);
    auto hand1 = _game.get_player_hand(1);
    _player0->accept_deal(hand0);
    _player1->accept_deal(hand1);
}

void GameSimulator::play_loop() {
    // Continue until someone wins
    while(!_game.is_over()) {
        play_turn();
    }
}

void GameSimulator::play_turn() {
    int current = _game.current_player();
    Player *curr_player = (current == 0 ? _player0.get() : _player1.get());
    Player *other_player = (current == 0 ? _player1.get() : _player0.get());

    // Ask the current player for a move
    auto move = curr_player->select_move();

    // Apply move to game
    _game.apply_move(move);

    // Record the move
    _record.add_move(current, move);

    // Notify opponent of the move
    other_player->accept_opponent_move(move);
}
