use big2_core::{Game, GameRecord};
use big2_players::player::AnyPlayer;
use rand::rngs::SmallRng;

/// Play a single game between two players and return the full record.
///
/// `game` must already be dealt (hands assigned).  Both players receive
/// `accept_deal`, then turns alternate until the game ends.
pub fn play_game(
    game: &Game,
    p0: &mut dyn AnyPlayer,
    p1: &mut dyn AnyPlayer,
) -> GameRecord {
    p0.accept_deal(game, 0);
    p1.accept_deal(game, 1);

    let mut record = GameRecord::new(game);
    let mut g = game.clone();

    while !g.is_over() {
        let cp = g.current_player();
        let players: [&mut dyn AnyPlayer; 2] = [p0, p1];

        let mv = players[cp].select_move();
        let tb_case = players[cp].last_tb_case();
        let tb_seq = players[cp].last_tb_forced_seq_len();
        let tb_straight = players[cp].last_tb_opp1_table_straight();

        // record.add_move expects the pre-move game state
        record.add_move(mv, &g, tb_case, tb_seq, tb_straight);
        g.apply_move(mv);
        players[1 - cp].accept_opponent_move(mv);
    }

    record
}

/// Deal and play a game with a fresh shuffle.
pub fn play_game_shuffle(
    p0: &mut dyn AnyPlayer,
    p1: &mut dyn AnyPlayer,
    rng: &mut SmallRng,
) -> GameRecord {
    let mut game = Game::new();
    game.shuffle_deal(rng);
    play_game(&game, p0, p1)
}

/// Play `n_games` and return the win counts for player 0 and player 1.
pub fn eval_match(
    p0: &mut dyn AnyPlayer,
    p1: &mut dyn AnyPlayer,
    n_games: u32,
    rng: &mut SmallRng,
) -> (u32, u32) {
    let mut wins = [0u32; 2];
    for _ in 0..n_games {
        let record = play_game_shuffle(p0, p1, rng);
        let winner = record.turns.last()
            .map(|t| 1 - t.current_player)  // the player who just emptied their hand
            .unwrap_or(0);
        wins[winner] += 1;
    }
    (wins[0], wins[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use big2_players::Player;
    use big2_players::random::RandomStrategy;
    use rand::SeedableRng;

    #[test]
    fn test_random_vs_random_completes() {
        let mut rng = SmallRng::seed_from_u64(99);
        let mut p0 = Player::new(RandomStrategy::new(SmallRng::seed_from_u64(1)));
        let mut p1 = Player::new(RandomStrategy::new(SmallRng::seed_from_u64(2)));
        let record = play_game_shuffle(&mut p0, &mut p1, &mut rng);
        assert!(!record.turns.is_empty());
    }

    #[test]
    fn test_card_conservation() {
        let mut rng = SmallRng::seed_from_u64(777);
        for _ in 0..100 {
            use rand::Rng;
            let mut p0 = Player::new(RandomStrategy::new(SmallRng::seed_from_u64(rng.gen())));
            let mut p1 = Player::new(RandomStrategy::new(SmallRng::seed_from_u64(rng.gen())));
            let record = play_game_shuffle(&mut p0, &mut p1, &mut rng);

            let initial = record.initial_game();
            let total: u8 = (0..13).map(|r| {
                initial.player_hand(0)[r] + initial.player_hand(1)[r]
            }).sum();
            assert_eq!(total, 32, "32 cards must be dealt (48 card deck, 16 each)");

            // After the game the winner has 0 cards; sum of last game state
            let last_turn = record.turns.last().unwrap();
            let g = &last_turn.game;
            // Apply last move to see final state.
            let mut final_g = g.clone();
            final_g.apply_move(last_turn.mv);
            let remaining: u8 = (0..13).map(|r| {
                final_g.player_hand(0)[r] + final_g.player_hand(1)[r] + final_g.discard_pile()[r]
            }).sum();
            assert_eq!(remaining, 32, "cards must be conserved");
        }
    }
}
