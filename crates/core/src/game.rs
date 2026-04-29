use crate::consts::*;
use crate::moves::{Move, MOVE_TO_CARDS};
use rand::Rng;

/// Full game state: both hands, discard pile, whose turn, last move played.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Game {
    hands: [[u8; 13]; 2],
    discard: [u8; 13],
    hand_size: [u8; 2],
    current_player: u8,
    last_move: Move,
}

impl Game {
    pub fn new() -> Self {
        Game {
            hands: [[0; 13]; 2],
            discard: [0; 13],
            hand_size: [0; 2],
            current_player: 0,
            last_move: Move::PASS,
        }
    }

    /// Construct from explicit full state (used by PIMC rollouts).
    pub fn from_state(
        hand0: [u8; 13],
        hand1: [u8; 13],
        discard: [u8; 13],
        last_move: Move,
        current_player: usize,
    ) -> Self {
        let size0: u8 = hand0.iter().sum();
        let size1: u8 = hand1.iter().sum();
        Game {
            hands: [hand0, hand1],
            discard,
            hand_size: [size0, size1],
            current_player: current_player as u8,
            last_move,
        }
    }

    /// Shuffle and deal 16 cards to each player; 16 cards left unseen.
    pub fn shuffle_deal<R: Rng>(&mut self, rng: &mut R) {
        // Build the 48-card deck
        let mut deck = Vec::with_capacity(48);
        for rank_idx in 0..13usize {
            let copies = max_copies_in_deck(rank_idx);
            for _ in 0..copies {
                deck.push(rank_idx as u8);
            }
        }
        // Fisher-Yates shuffle
        for i in (1..deck.len()).rev() {
            let j = rng.gen_range(0..=i);
            deck.swap(i, j);
        }

        self.hands = [[0; 13]; 2];
        self.discard = [0; 13];
        for i in 0..16 {
            self.hands[0][deck[i] as usize] += 1;
        }
        for i in 16..32 {
            self.hands[1][deck[i] as usize] += 1;
        }
        self.hand_size = [16, 16];
        self.current_player = 0;
        self.last_move = Move::PASS;
    }

    #[inline]
    pub fn current_player(&self) -> usize {
        self.current_player as usize
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        self.hand_size[0] == 0 || self.hand_size[1] == 0
    }

    #[inline]
    pub fn winner(&self) -> usize {
        if self.hand_size[0] == 0 { 0 } else { 1 }
    }

    #[inline]
    pub fn player_hand(&self, player: usize) -> [u8; 13] {
        self.hands[player]
    }

    #[inline]
    pub fn hand_size(&self, player: usize) -> u8 {
        self.hand_size[player]
    }

    #[inline]
    pub fn discard_pile(&self) -> [u8; 13] {
        self.discard
    }

    #[inline]
    pub fn last_move(&self) -> Move {
        self.last_move
    }

    /// Apply a move given as a move ID.
    pub fn apply_move_id(&mut self, move_id: usize) {
        debug_assert!(move_id < LEGAL_MOVES_SIZE);
        let cost = &MOVE_TO_CARDS[move_id];
        let cp = self.current_player as usize;
        for rank in 0..13 {
            let c = cost[rank];
            debug_assert!(self.hands[cp][rank] >= c);
            self.hands[cp][rank] -= c;
            self.discard[rank] += c;
        }
        let total = cost[13];
        self.hand_size[cp] -= total;
        self.last_move = Move::decode(move_id);
        self.current_player = 1 - self.current_player;
    }

    /// Apply a move given as a Move struct.
    #[inline]
    pub fn apply_move(&mut self, mv: Move) {
        self.apply_move_id(mv.encode());
    }

    /// Legal moves for the current player.
    pub fn legal_moves(&self) -> Vec<usize> {
        crate::util::compute_legal_moves(&self.hands[self.current_player as usize], self.last_move)
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use crate::consts::PASS;

    fn seeded(seed: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed)
    }

    #[test]
    fn test_deal_invariants() {
        let mut rng = seeded(42);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        assert_eq!(g.hand_size(0), 16);
        assert_eq!(g.hand_size(1), 16);
        assert!(!g.is_over());

        let h0 = g.player_hand(0);
        let h1 = g.player_hand(1);
        let dp = g.discard_pile();

        for rank in 0..13 {
            let total = h0[rank] + h1[rank] + dp[rank];
            let limit = max_copies_in_deck(rank);
            assert!(h0[rank] <= limit);
            assert!(h1[rank] <= limit);
            assert_eq!(dp[rank], 0);
            assert!(total <= limit);
        }
    }

    #[test]
    fn test_player_alternates() {
        let mut rng = seeded(1);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        let before = g.current_player();
        let legal = g.legal_moves();
        assert!(!legal.is_empty());
        g.apply_move_id(legal[0]);
        assert_eq!(g.current_player(), 1 - before);
    }

    #[test]
    fn test_legal_moves_are_affordable() {
        let mut rng = seeded(99);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        let hand = g.player_hand(g.current_player());
        let legal = g.legal_moves();
        assert!(!legal.is_empty());

        for mid in legal {
            let cost = MOVE_TO_CARDS[mid];
            for rank in 0..13 {
                assert!(hand[rank] >= cost[rank]);
            }
        }
    }

    #[test]
    fn test_apply_move_id_and_move_agree() {
        let mut rng = seeded(7);
        let mut g1 = Game::new();
        g1.shuffle_deal(&mut rng);
        let mut rng2 = seeded(7);
        let mut g2 = Game::new();
        g2.shuffle_deal(&mut rng2);

        let legal = g1.legal_moves();
        assert!(!legal.is_empty());
        let mid = legal[0];
        let mv = Move::decode(mid);

        g1.apply_move(mv);
        g2.apply_move_id(mid);

        for p in 0..2 {
            assert_eq!(g1.hand_size(p), g2.hand_size(p));
            assert_eq!(g1.player_hand(p), g2.player_hand(p));
        }
        assert_eq!(g1.current_player(), g2.current_player());
    }

    #[test]
    fn test_pass_legality() {
        let mut rng = seeded(3);
        let mut g = Game::new();
        g.shuffle_deal(&mut rng);

        // Start of game: no active trick, pass must NOT be legal.
        let legal_start = g.legal_moves();
        assert!(!legal_start.contains(&PASS));

        // After a non-pass move, pass must appear in opponent's legal set.
        let first_move = *legal_start.iter().find(|&&m| m != PASS).unwrap();
        g.apply_move_id(first_move);

        let legal_after = g.legal_moves();
        assert!(legal_after.contains(&PASS));
    }
}
