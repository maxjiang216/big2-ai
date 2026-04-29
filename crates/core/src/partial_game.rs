use crate::consts::max_copies_in_deck;
use crate::game::Game;
use crate::moves::{Combination, Move, MOVE_TO_CARDS};
use crate::util::{compute_legal_moves, compute_possible_moves};

/// A player's imperfect-information view of the game state.
/// Tracks only our hand, the discard pile, opponent card count, and the last move.
/// `turn` = 0 means it is our turn; 1 means it is the opponent's turn.
#[derive(Debug, Clone)]
pub struct PartialGame {
    turn: u8,
    player_hand: [u8; 13],
    opponent_card_count: u8,
    discard: [u8; 13],
    last_move: Move,
}

impl PartialGame {
    /// Construct from a full `Game` from the perspective of `player_num`.
    pub fn from_game(game: &Game, player_num: usize) -> Self {
        let turn = if game.current_player() == player_num { 0 } else { 1 };
        PartialGame {
            turn: turn as u8,
            player_hand: game.player_hand(player_num),
            opponent_card_count: game.hand_size(1 - player_num),
            discard: game.discard_pile(),
            last_move: game.last_move(),
        }
    }

    /// Apply a move (may be ours or the opponent's — determined by `turn`).
    pub fn apply_move(&mut self, mv: Move) {
        let move_id = mv.encode();
        let cost = &MOVE_TO_CARDS[move_id];
        if self.turn == 0 {
            // Our move: remove cards from our hand.
            for i in 0..13 {
                self.player_hand[i] -= cost[i];
                self.discard[i] += cost[i];
            }
        } else {
            // Opponent's move: only update discard and their count.
            for i in 0..13 {
                self.discard[i] += cost[i];
            }
            self.opponent_card_count -= cost[13];
        }
        self.last_move = mv;
        self.turn = 1 - self.turn;
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        compute_legal_moves(&self.player_hand, self.last_move)
    }

    pub fn possible_moves(&self) -> Vec<usize> {
        compute_possible_moves(
            &self.player_hand,
            &self.discard,
            self.opponent_card_count,
            self.last_move,
            false,
        )
    }

    pub fn possible_moves_not_bomb(&self) -> Vec<usize> {
        compute_possible_moves(
            &self.player_hand,
            &self.discard,
            self.opponent_card_count,
            self.last_move,
            true,
        )
    }

    pub fn count_bombs(&self) -> u8 {
        let mut n = 0;
        if self.player_hand[11] == 3 {
            n += 1;
        }
        for i in 0..11 {
            if self.player_hand[i] == 4 {
                n += 1;
            }
        }
        n
    }

    pub fn trick_rank(&self) -> Option<u8> {
        if self.last_move.combination != Combination::Pass {
            Some(self.last_move.rank)
        } else {
            None
        }
    }

    #[inline] pub fn turn(&self) -> usize { self.turn as usize }
    #[inline] pub fn player_hand(&self) -> [u8; 13] { self.player_hand }
    #[inline] pub fn discard_pile(&self) -> [u8; 13] { self.discard }
    #[inline] pub fn opponent_hand_size(&self) -> u8 { self.opponent_card_count }
    #[inline] pub fn last_move(&self) -> Move { self.last_move }

    #[inline]
    pub fn num_cards(&self) -> u8 {
        self.player_hand.iter().sum()
    }

    /// True if every rank count in our hand is at most 1.
    pub fn hand_is_only_singles(&self) -> bool {
        self.player_hand.iter().all(|&c| c <= 1)
    }

    /// Available ranks in opponent's unknown pool (not in our hand, not discarded).
    pub fn unseen_count(&self, rank_idx: usize) -> u8 {
        max_copies_in_deck(rank_idx) - self.player_hand[rank_idx] - self.discard[rank_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::consts::PASS;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    #[test]
    fn test_construction_from_game() {
        let mut rng = seeded(17);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        let p0 = PartialGame::from_game(&g, 0);
        let p1 = PartialGame::from_game(&g, 1);

        // current_player is 0, so p0.turn=0 (my turn), p1.turn=1 (opponent's turn)
        assert_eq!(p0.turn(), 0);
        assert_eq!(p1.turn(), 1);

        // Hands match game.
        assert_eq!(p0.player_hand(), g.player_hand(0));
        assert_eq!(p1.player_hand(), g.player_hand(1));
    }

    #[test]
    fn test_partial_legal_equals_game_legal() {
        let mut rng = seeded(9);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        let p0 = PartialGame::from_game(&g, 0);

        let mut game_legal = g.legal_moves();
        let mut part_legal = p0.legal_moves();
        game_legal.sort_unstable();
        part_legal.sort_unstable();
        assert_eq!(part_legal, game_legal);
    }

    #[test]
    fn test_apply_move_flips_turn() {
        let mut rng = seeded(5);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        let mut p0 = PartialGame::from_game(&g, 0);
        assert_eq!(p0.turn(), 0);

        let legal = p0.legal_moves();
        p0.apply_move(Move::decode(legal[0]));
        assert_eq!(p0.turn(), 1);
    }

    #[test]
    fn test_possible_moves_not_bomb_has_no_bombs() {
        let mut rng = seeded(77);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);
        let p0 = PartialGame::from_game(&g, 0);

        for mid in p0.possible_moves_not_bomb() {
            if mid == PASS { continue; }
            let m = Move::decode(mid);
            assert_ne!(m.combination, Combination::Bomb);
        }
    }
}
