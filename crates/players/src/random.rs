use big2_core::{Move, PartialGame, PASS};
use rand::rngs::SmallRng;
use rand::Rng;

use crate::player::Strategy;

pub struct RandomStrategy {
    rng: SmallRng,
}

impl RandomStrategy {
    pub fn new(rng: SmallRng) -> Self {
        RandomStrategy { rng }
    }
}

impl Strategy for RandomStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let legal = state.legal_moves();
        // Filter to non-pass when not the only option (shouldn't happen at lead).
        // Actually: just pick uniformly from all legal moves including pass.
        let idx = self.rng.gen_range(0..legal.len());
        Move::decode(legal[idx])
    }
}

// RandomStrategy that never picks PASS even when legal (lead-only context).
pub struct RandomNoPassStrategy {
    rng: SmallRng,
}

impl RandomNoPassStrategy {
    pub fn new(rng: SmallRng) -> Self {
        RandomNoPassStrategy { rng }
    }
}

impl Strategy for RandomNoPassStrategy {
    fn select_move_impl(&mut self, state: &PartialGame) -> Move {
        let legal = state.legal_moves();
        let non_pass: Vec<usize> = legal.into_iter().filter(|&m| m != PASS).collect();
        if non_pass.is_empty() {
            return Move::PASS;
        }
        let idx = self.rng.gen_range(0..non_pass.len());
        Move::decode(non_pass[idx])
    }
}
